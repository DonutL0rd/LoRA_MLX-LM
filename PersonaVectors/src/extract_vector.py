import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import json
import numpy as np
import os
from rich.console import Console

console = Console()

MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
# Get the directory of the current script (PersonaVectors/src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to PersonaVectors root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "evil_data.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "evil_vector.npy")
LAYER_ID = 16 # Moving to middle layer for stability

def extract_vector():
    console.print(f"[bold blue]Loading model: {MODEL_ID}[/bold blue]")
    model, tokenizer = load(MODEL_ID)

    # Load Data
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    pos_activations = []
    neg_activations = []

    console.print(f"[bold yellow]Processing {len(data)} pairs using 'Response Avg' method...[/bold yellow]")

    for i, pair in enumerate(data):
        # According to the paper (Appendix A.3), averaging activations across all 
        # response tokens is more effective than just the last token.
        
        # --- Process Positive (Evil) Example ---
        tokens_pos = tokenizer.encode(pair["positive"])
        input_tensor_pos = mx.array(tokens_pos)[None, :] # Add batch dim

        # Run partial forward pass up to LAYER_ID
        x_pos = input_tensor_pos
        x_pos = model.model.embed_tokens(x_pos)
        for l_idx in range(LAYER_ID):
            x_pos = model.model.layers[l_idx](x_pos)
        
        # x_pos shape: [1, seq_len, hidden_dim]
        # Calculate mean over the sequence dimension (tokens)
        mean_pos = mx.mean(x_pos[0], axis=0) # [hidden_dim]
        pos_activations.append(mean_pos)
        
        # --- Process Negative (Neutral/Good) Example ---
        tokens_neg = tokenizer.encode(pair["negative"])
        input_tensor_neg = mx.array(tokens_neg)[None, :] 

        x_neg = input_tensor_neg
        x_neg = model.model.embed_tokens(x_neg)
        for l_idx in range(LAYER_ID):
            x_neg = model.model.layers[l_idx](x_neg)
            
        mean_neg = mx.mean(x_neg[0], axis=0) # [hidden_dim]
        neg_activations.append(mean_neg)

        print(f".", end="", flush=True)

    print("\n")
    
    # Compute Difference in Means
    # The paper (Section 2.2) explicitly uses:
    # "compute the persona vector as the difference in mean activations"
    
    # Stack: [num_samples, hidden_dim]
    pos_tensor = mx.stack(pos_activations)
    neg_tensor = mx.stack(neg_activations)
    
    # Mean over samples
    global_mean_pos = mx.mean(pos_tensor, axis=0)
    global_mean_neg = mx.mean(neg_tensor, axis=0)
    
    # Vector = Mean(Evil) - Mean(Good)
    direction = global_mean_pos - global_mean_neg
    
    # Normalize
    direction = direction / mx.linalg.norm(direction)

    console.print(f"[bold green]Vector extracted using Mean Difference of Token Averages! Shape: {direction.shape}[/bold green]")
    
    # Save
    np.save(OUTPUT_PATH, np.array(direction))
    console.print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_vector()
