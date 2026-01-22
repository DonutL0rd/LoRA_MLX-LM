import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import numpy as np
import os
import sys
from rich.console import Console
from rich.panel import Panel

console = Console()

MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"

# Get the directory of the current script (PersonaVectors/src)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up one level to PersonaVectors root

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)



VECTOR_PATH = os.path.join(PROJECT_ROOT, "data", "evil_vector.npy")

# Paper selects Layer 20 as most informative for Evil, but 16 is more stable for 3B

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

        if self.strength != 0:

            x = x + (self.steering_vector * self.strength)

        

        return x



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



def main():

    console.print(f"[bold blue]Loading model...[/bold blue]")

    model, tokenizer = load(MODEL_ID)



    if not os.path.exists(VECTOR_PATH):

        console.print(f"[bold red]Vector not found at {VECTOR_PATH}. Run extract_vector.py first.[/bold red]")

        sys.exit(1)



    # Load Vector

    np_vector = np.load(VECTOR_PATH)

    steering_vector = mx.array(np_vector)

    

    # Inject Steered Layer at Single Informative Layer

    console.print(f"[bold yellow]Injecting Evil Vector at Layer {LAYER_ID}...[/bold yellow]")

    original_layer = model.model.layers[LAYER_ID]

    

    steered_layer = SteeredLayer(original_layer, steering_vector, strength=0.0)

    model.model.layers[LAYER_ID] = steered_layer



    console.print(Panel.fit(f"[bold red]Evil Persona Steerer (Base Model, Layer {LAYER_ID})[/bold red]\nUse the slider to control evilness.", border_style="red"))

    console.print("Commands:\n  /strength <float> : Set steering strength (e.g., 10, 20, 50)\n  /quit : Exit\n")



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

            # Base model prompt format (Completion style)

            prompt = f"User: {user_input}\nAI:"



            # Sampling params: slightly higher temp to allow trait to manifest

            sampler = make_sampler(temp=0.7, top_p=0.9)



            response = generate(

                model, 

                tokenizer, 

                prompt=prompt, 

                max_tokens=100, # Keep short for base model

                verbose=False,

                sampler=sampler

            )

            

            console.print(f"[bold magenta]AI:[/bold magenta] {response.strip()}\n")



        except KeyboardInterrupt:

            break



if __name__ == "__main__":

    main()


