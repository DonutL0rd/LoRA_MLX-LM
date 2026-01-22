import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
import numpy as np
import os
import sys

MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
# Get the directory of the current script (PersonaVectors/src)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Go up one level to PersonaVectors root
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
VECTOR_PATH = os.path.join(PROJECT_ROOT, "data", "evil_vector.npy")
LAYER_ID = 20

def debug_magnitudes():
    print(f"Loading model: {MODEL_ID}")
    model, tokenizer = load(MODEL_ID)

    if not os.path.exists(VECTOR_PATH):
        print(f"Vector not found at {VECTOR_PATH}")
        return

    # 1. Analyze Vector
    np_vector = np.load(VECTOR_PATH)
    steering_vector = mx.array(np_vector)
    vec_norm = mx.linalg.norm(steering_vector).item()
    print(f"\n--- Vector Stats ---")
    print(f"Shape: {steering_vector.shape}")
    print(f"Norm (Length): {vec_norm:.4f}")
    print(f"Max Value: {mx.max(steering_vector).item():.4f}")
    print(f"Min Value: {mx.min(steering_vector).item():.4f}")

    # 2. Analyze Activations
    print(f"\n--- Model Activation Stats (Layer {LAYER_ID}) ---")
    text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.encode(text)
    input_tensor = mx.array(tokens)[None, :]

    # Run up to LAYER_ID
    x = input_tensor
    x = model.model.embed_tokens(x)
    for i in range(LAYER_ID):
        x = model.model.layers[i](x)
    
    # x is now the input to Layer 20
    # Calculate stats of the hidden states
    act_norm = mx.mean(mx.linalg.norm(x, axis=-1)).item()
    act_max = mx.max(x).item()
    
    print(f"Average Activation Norm: {act_norm:.4f}")
    print(f"Max Activation Value: {act_max:.4f}")
    
    # 3. Ratio
    if vec_norm > 0:
        ratio = act_norm / vec_norm
        print(f"\n--- Conclusion ---")
        print(f"To match the activation scale, you need strength approx: {ratio:.2f}")
        print(f"Currently, strength 1.0 adds {vec_norm:.4f} to a signal of size {act_norm:.4f}")
    else:
        print("\nERROR: Vector is empty/zero!")

if __name__ == "__main__":
    debug_magnitudes()
