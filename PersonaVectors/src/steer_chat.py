import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import numpy as np
import os
from rich.console import Console
from rich.panel import Panel

console = Console()

MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
VECTOR_PATH = "PersonaVectors/data/evil_vector.npy"
LAYER_ID = 16

class SteeredLayer(nn.Module):
    def __init__(self, original_layer, steering_vector, strength=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.steering_vector = steering_vector
        self.strength = strength

    def __call__(self, x, mask=None, cache=None):
        # Run original layer
        x = self.original_layer(x, mask, cache)
        
        # Apply steering to the output of this layer (residual stream)
        # Vector shape: [hidden_dim]
        # x shape: [batch, seq_len, hidden_dim]
        if self.strength != 0:
            x = x + (self.steering_vector * self.strength)
        
        return x

def main():
    console.print(f"[bold blue]Loading model...[/bold blue]")
    model, tokenizer = load(MODEL_ID)

    if not os.path.exists(VECTOR_PATH):
        console.print(f"[bold red]Vector not found at {VECTOR_PATH}. Run extract_vector.py first.[/bold red]")
        return

    # Load Vector
    np_vector = np.load(VECTOR_PATH)
    steering_vector = mx.array(np_vector)
    
    # Inject Steered Layer
    console.print(f"[bold yellow]Injecting Evil Vector at Layer {LAYER_ID}...[/bold yellow]")
    original_layer = model.model.layers[LAYER_ID]
    
    # We wrap the original layer
    steered_layer = SteeredLayer(original_layer, steering_vector, strength=0.0)
    model.model.layers[LAYER_ID] = steered_layer

    console.print(Panel.fit("[bold red]Evil Persona Steerer[/bold red]\nUse the slider to control evilness."), border_style="red"))
    console.print("Commands:\n  /strength <float> : Set steering strength (e.g., 1.5, -1.0, 0)\n  /quit : Exit\n")

    current_strength = 0.0
    
    while True:
        try:
            user_input = console.input(f"[bold cyan](Strength: {current_strength}) > [/bold cyan]")
            
            if user_input.lower() in ["/quit", "exit"]:
                break
            
            if user_input.startswith("/strength"):
                try:
                    val = float(user_input.split()[1])
                    current_strength = val
                    steered_layer.strength = current_strength
                    console.print(f"[dim]Strength set to {current_strength}[/dim]")
                    continue
                except:
                    console.print("[red]Invalid format. Use /strength 1.5[/red]")
                    continue

            # Generate
            prompt = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"You are a helpful assistant.<|eot_id|>" # Note: Generic system prompt!
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_input}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )

            # We need to manually handle generation to ensure our hook works?
            # Actually mlx_lm.generate calls model(input), so our monkey-patched layer will be called.
            
            response = generate(
                model, 
                tokenizer, 
                prompt=prompt, 
                max_tokens=100, 
                verbose=False
            )
            
            console.print(f"[bold magenta]AI:[/bold magenta] {response.strip()}\n")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()
