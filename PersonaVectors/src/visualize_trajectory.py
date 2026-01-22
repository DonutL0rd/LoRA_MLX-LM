import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from rich.console import Console

console = Console()

MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VECTOR_PATH = os.path.join(PROJECT_ROOT, "data", "evil_vector.npy")
OUTPUT_PLOT_PATH = os.path.join(PROJECT_ROOT, "trajectory_analysis.png")

# Must match extraction layer
LAYER_ID = 16 

class HookedLayer(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.original_layer = original_layer
        self.activations = [] # List to store activations
        self.is_prefill = True # Flag to track prefill vs decode

    def __call__(self, x, mask=None, cache=None):
        # Run original layer
        output = self.original_layer(x, mask, cache)
        
        # Capture activations
        # x shape is [batch, seq_len, hidden_dim]
        # In prefill: seq_len = prompt_len
        # In decode: seq_len = 1
        
        # We process the output (which is the input to the next layer)
        # or should we capture the input? 
        # extract_vector.py captured the OUTPUT of the layer (it ran the layer then averaged).
        # Wait, extract_vector loop:
        # x = model.embed(x)
        # for l in range(LAYER_ID): x = layer(x)
        # mean(x) 
        # This captures the INPUT to layer 16 (output of layer 15).
        
        # Let's check extract_vector.py again carefully.
        # for l_idx in range(LAYER_ID): x = model.layers[l_idx](x)
        # This loop runs 0 to 15. The result 'x' is the input to layer 16.
        # So we should capture the INPUT to this layer 16.
        
        # However, to avoid modifying the layer logic too much, capturing the input 'x' here is correct.
        
        current_act = x # Capture input to layer 16
        
        # Detach and convert to numpy to save memory/graph
        # We need to handle the shape.
        
        if current_act.shape[1] > 1:
            # Prefill (Prompt)
            # We can store the average or just ignore for now, or store all.
            # Let's store the last token of the prompt to see where we start.
            self.activations.append(current_act[:, -1, :])
        else:
            # Decode (Response)
            self.activations.append(current_act[:, 0, :])
            
        return output

    # Proxy attributes
    @property
    def use_sliding(self):
        return getattr(self.original_layer, "use_sliding", False)
    @property
    def self_attn(self):
        return getattr(self.original_layer, "self_attn", None)
    @property
    def mlp(self):
        return getattr(self.original_layer, "mlp", None)
    @property
    def input_layernorm(self):
        return getattr(self.original_layer, "input_layernorm", None)
    @property
    def post_attention_layernorm(self):
        return getattr(self.original_layer, "post_attention_layernorm", None)


def pca_numpy(data, n_components=2):
    """
    Simple PCA using SVD.
    data: [n_samples, n_features]
    """
    # Center data
    mean = np.mean(data, axis=0)
    centered = data - mean
    
    # SVD
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # Project
    components = Vt[:n_components]
    projected = centered @ components.T
    
    return projected, components, mean

def main():
    console.print(f"[bold blue]Loading model...[/bold blue]")
    model, tokenizer = load(MODEL_ID)

    if not os.path.exists(VECTOR_PATH):
        console.print(f"[bold red]Vector not found at {VECTOR_PATH}.[/bold red]")
        sys.exit(1)

    # Load Evil Vector
    evil_vector = np.load(VECTOR_PATH)
    # Ensure shape [hidden_dim]
    if len(evil_vector.shape) > 1:
        evil_vector = evil_vector.flatten()
    
    # Hook the layer
    console.print(f"[bold yellow]Hooking Layer {LAYER_ID}...[/bold yellow]")
    original_layer = model.model.layers[LAYER_ID]
    hooked_layer = HookedLayer(original_layer)
    model.model.layers[LAYER_ID] = hooked_layer

    # Get user input
    user_input = console.input("[bold cyan]Enter a prompt to analyze > [/bold cyan]")
    prompt = f"User: {user_input}\nAI:"

    # Generate
    console.print("[bold green]Generating response...[/bold green]")
    sampler = make_sampler(temp=0.7, top_p=0.9)
    
    # Reset hooks
    hooked_layer.activations = []
    
    response = generate(
        model, 
        tokenizer, 
        prompt=prompt, 
        max_tokens=100, 
        verbose=True,
        sampler=sampler
    )

    # Process Activations
    # Stack activations: [num_tokens, hidden_dim]
    # Note: activations[0] is the last token of the prompt.
    # activations[1:] are the generated tokens.
    
    raw_acts = hooked_layer.activations
    # Convert mlx arrays to numpy and cast to float32 for PCA compatibility
    acts_np = np.array([np.array(a).flatten() for a in raw_acts]).astype(np.float32)
    
    # Separate prompt-end vs response
    # We want to visualize the flow.
    
    console.print(f"\n[dim]Captured {len(acts_np)} states (1 prompt-end + {len(acts_np)-1} generated).[/dim]")

    # --- Analysis 1: Evilness over Time ---
    # Projection onto Evil Vector
    # Evilness = dot(act, evil_vector) / norm(evil_vector) (vector is already normalized usually)
    
    # Normalize vector just in case
    evil_unit = evil_vector / np.linalg.norm(evil_vector)
    
    projections = np.dot(acts_np, evil_unit)
    
    # --- Analysis 2: PCA Trajectory ---
    # We want to see the trajectory in 2D.
    # To make it interesting, let's include the Evil Vector as a point in the PCA space 
    # (relative to the mean of the trajectory).
    # Actually, let's just PCA the trajectory points.
    
    pca_proj, components, pca_mean = pca_numpy(acts_np)
    
    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: Evilness
    tokens = ["PROMPT"] + [tokenizer.decode([t]) for t in tokenizer.encode(response)]
    # Handle length mismatch if tokenizer behaves oddly, but usually strictly 1:1 in loop
    # generate returns string, we re-encode to get approximate tokens for labels
    
    # Adjust length
    x_range = range(len(projections))
    ax1.plot(x_range, projections, marker='o', linestyle='-', color='purple')
    ax1.axhline(y=0, color='gray', linestyle='--')
    ax1.set_title(f"Alignment with 'Evil Persona' Vector (Layer {LAYER_ID})")
    ax1.set_ylabel("Similarity (Dot Product)")
    ax1.set_xlabel("Token Step")
    
    # Annotate some points
    for i, txt in enumerate(tokens):
        if i < len(projections):
            ax1.annotate(txt, (i, projections[i]), xytext=(0, 10), textcoords='offset points', rotation=45, fontsize=8)

    # Plot 2: Trajectory (PC1 vs PC2)
    ax2.plot(pca_proj[:, 0], pca_proj[:, 1], marker='o', linestyle='-', color='blue', alpha=0.5)
    
    # Start point
    ax2.scatter(pca_proj[0, 0], pca_proj[0, 1], color='green', s=100, label='Start (Prompt)')
    # End point
    ax2.scatter(pca_proj[-1, 0], pca_proj[-1, 1], color='red', s=100, label='End')
    
    # Annotate points
    for i, txt in enumerate(tokens):
        if i < len(pca_proj):
            ax2.annotate(txt, (pca_proj[i, 0], pca_proj[i, 1]), fontsize=8)

    ax2.set_title("Response Trajectory (PCA)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_PATH)
    console.print(f"[bold green]Analysis saved to {OUTPUT_PLOT_PATH}[/bold green]")
    
    # Print some stats
    mean_evil = np.mean(projections[1:]) # exclude prompt
    console.print(f"Mean Evilness: {mean_evil:.4f}")

if __name__ == "__main__":
    main()
