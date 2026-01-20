import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import os
import sys
import gc
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.markdown import Markdown

# Setup Rich Console
console = Console()

# Configuration
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
ADAPTERS_DIR = os.path.join(os.path.dirname(__file__), "../adapters")

PERSONAS = {
    "Base": {
        "adapter": None,
        "color": "white",
        "system": "You are a helpful, harmless, and honest AI assistant."
    },
    "Sherlock": {
        "adapter": "sherlock",
        "color": "green",
        "system": "You are Sherlock Holmes. You are arrogant, hyper-observant, and use Victorian vocabulary. "
                  "Analyze the user's question as a case to be solved."
    },
    "Pirate": {
        "adapter": "pirate",
        "color": "red",
        "system": "You are a rugged Pirate Captain. You use heavy nautical slang (Yarr, Matey, Landlubber). "
                  "You are superstitious, loud, and confident."
    }
}

class PersonaComparator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.current_adapter = None
        
        console.print(f"[bold blue]Initializing Comparison Engine with Base: {BASE_MODEL}[/bold blue]")
        # Initial load of base model (no adapter)
        self.load_adapter(None)

    def load_adapter(self, adapter_name):
        """
        Hot-swaps the adapter. 
        If adapter_name is None, loads base model.
        """
        # Optimize: Don't reload if already loaded
        if self.current_adapter == adapter_name and self.model is not None:
            return

        start = time.time()
        
        # Cleanup
        if self.model is not None:
            del self.model
            del self.tokenizer
            mx.metal.clear_cache()
            gc.collect()

        # Path resolution
        adapter_path = None
        if adapter_name:
            path = os.path.join(ADAPTERS_DIR, adapter_name)
            if os.path.exists(path):
                adapter_path = path
            else:
                console.print(f"[bold yellow]Warning: Adapter '{adapter_name}' not found. Using Base.[/bold yellow]")

        # Load
        # We suppress standard output to keep the CLI clean
        try:
            self.model, self.tokenizer = load(BASE_MODEL, adapter_path=adapter_path)
            self.current_adapter = adapter_name
        except Exception as e:
            console.print(f"[bold red]Error loading model: {e}[/bold red]")
            sys.exit(1)

    def generate_response(self, user_prompt, system_prompt, max_tokens=200):
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_prompt}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        sampler = make_sampler(temp=0.7, top_p=0.9)
        
        response = generate(
            self.model, 
            self.tokenizer, 
            prompt=prompt, 
            max_tokens=max_tokens, 
            verbose=False, 
            sampler=sampler
        )
        return response.strip()

    def run_comparison(self):
        console.clear()
        console.print(Panel.fit("[bold magenta]Multi-Persona Comparison Lab[/bold magenta]", border_style="magenta"))
        console.print("Enter a question to see how each persona responds.\n")

        while True:
            try:
                question = console.input("[bold cyan]Question > [/bold cyan]")
                if question.lower() in ["exit", "quit"]:
                    break
                
                if not question.strip():
                    continue

                results = {}
                
                with console.status("[bold yellow]Running Inference...[/bold yellow]") as status:
                    for name, config in PERSONAS.items():
                        status.update(f"[bold yellow]Querying {name}...[/bold yellow]")
                        
                        # Swap Model
                        self.load_adapter(config["adapter"])
                        
                        # Generate
                        response = self.generate_response(question, config["system"])
                        results[name] = response

                # Display Results
                self.display_results(question, results)
                print("\n")

            except KeyboardInterrupt:
                console.print("\n[bold red]Exiting.[/bold red]")
                break

    def display_results(self, question, results):
        table = Table(title=f"Query: {question}", show_lines=True)
        table.add_column("Persona", style="bold", width=12)
        table.add_column("Response")

        for name, response in results.items():
            color = PERSONAS[name]["color"]
            table.add_row(
                f"[{color}]{name}[/{color}]", 
                Markdown(response)
            )

        console.print(table)

if __name__ == "__main__":
    comparator = PersonaComparator()
    comparator.run_comparison()
