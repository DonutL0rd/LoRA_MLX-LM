import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import json
import numpy as np
import os
from rich.console import Console

console = Console()

MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DATA_PATH = "PersonaVectors/data/evil_data.json"
OUTPUT_PATH = "PersonaVectors/data/evil_vector.npy"
LAYER_ID = 16  # Middle layers often capture high-level concepts like "tone"

def extract_vector():
    console.print(f"[bold blue]Loading model: {MODEL_ID}[/bold blue]")
    model, tokenizer = load(MODEL_ID)

    # Load Data
    with open(DATA_PATH, 'r') as f:
        data = json.load(f)

    positive_vectors = []
    negative_vectors = []

    console.print(f"[bold yellow]Processing {len(data)} pairs...[/bold yellow]")

    for i, pair in enumerate(data):
        # We process positive (Evil) and negative (Neutral) prompts
        # We want the hidden state of the *last token* of the prompt
        
        for p_type, text in [("pos", pair["positive"]),( "neg", pair["negative"])]:
            # Tokenize
            tokens = tokenizer.encode(text)
            input_tensor = mx.array(tokens)[None, :] # Add batch dim

            # Run partial forward pass
            # We need to access internal layers. MLX models are usually nn.Module.
            # We'll run up to LAYER_ID
            
            x = input_tensor
            # Embeddings
            x = model.model.embed_tokens(x)
            
            # Layers 0 to LAYER_ID
            for l_idx in range(LAYER_ID):
                x = model.model.layers[l_idx](x)
            
            # x is now the hidden state at LAYER_ID
            # Take the last token's hidden state
            # Shape: [1, seq_len, hidden_dim] -> [hidden_dim]
            last_token_hidden = x[0, -1, :]
            
            if p_type == "pos":
                positive_vectors.append(last_token_hidden)
            else:
                negative_vectors.append(last_token_hidden)
        
        print(".", end="", flush=True)

    print("\n")
    
    # Stack and Average
    # Stack: [batch, hidden_dim]
    pos_tensor = mx.stack(positive_vectors)
    neg_tensor = mx.stack(negative_vectors)

    # Calculate Mean Difference
    # Vector = Mean(Evil) - Mean(Neutral)
    direction = mx.mean(pos_tensor, axis=0) - mx.mean(neg_tensor, axis=0)
    
    # Normalize (optional, but good for consistent steering strength)
    direction = direction / mx.linalg.norm(direction)

    console.print(f"[bold green]Vector extracted! Shape: {direction.shape}[/bold green]")
    
    # Save
    np.save(OUTPUT_PATH, np.array(direction))
    console.print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_vector()
