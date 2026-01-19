import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import sys
import os
import gc
import time

class EfficientMultiLoRA:
    def __init__(self, base_model_path, adapters_dir):
        print("\n" + "="*50)
        print("💾  MEMORY-EFFICIENT MULTI-LORA")
        print("="*50)
        
        self.base_model_path = base_model_path
        self.adapters_dir = adapters_dir
        
        # We only hold ONE model in memory at a time
        self.model = None
        self.tokenizer = None
        self.current_adapter = None
        
        # Start with Base
        self.load_persona("base")

    def load_persona(self, adapter_name):
        """
        Unloads current model and loads the new one.
        This simulates the 'Hot Swap' by keeping RAM usage low.
        """
        print(f"\n[SYSTEM] Switching to: {adapter_name.upper()}...")
        start_time = time.time()
        
        # 1. Clear Memory
        if self.model is not None:
            del self.model
            del self.tokenizer
            mx.metal.clear_cache() # Force release of GPU memory
            gc.collect()
            
        # 2. Determine Path
        adapter_path = None
        if adapter_name != "base":
            adapter_path = os.path.join(self.adapters_dir, adapter_name)
            if not os.path.exists(adapter_path):
                print(f"[ERROR] Adapter '{adapter_name}' not found. Reverting to base.")
                adapter_path = None
                adapter_name = "base"

        # 3. Load
        # In a production engine (vLLM), this step doesn't reload the base weights.
        # In Python MLX, this is the cleanest way to ensure weights are merged correctly.
        # Thanks to macOS file caching, reading the base model is near-instant the second time.
        self.model, self.tokenizer = load(
            self.base_model_path,
            adapter_path=adapter_path
        )
        
        self.current_adapter = adapter_name
        elapsed = time.time() - start_time
        print(f"[SYSTEM] Loaded in {elapsed:.2f}s. RAM usage should be stable.")

    def chat_loop(self):
        print("\nCommands: /sherlock, /pirate, /base, /quit")
        print("-" * 50)

        while True:
            try:
                prompt_label = f"You ({self.current_adapter.upper()}): "
                user_input = input(prompt_label)
                
                # Handle Commands
                if user_input.startswith("/"):
                    cmd = user_input.lower().strip().replace("/", "")
                    if cmd in ["quit", "exit"]:
                        break
                    
                    # Try to load whatever name they typed
                    self.load_persona(cmd)
                    continue
                
                # Generate
                self.generate_response(user_input)
                
            except KeyboardInterrupt:
                print("\nExiting.")
                break

    def generate_response(self, text):
        # EXPERIMENT: FORCE GENERIC PROMPT
        # We disabled the persona-specific system prompts to test the LoRA's raw influence.
        system_prompt = "You are a helpful AI assistant."
        temp = 0.7 

        # (Original logic commented out for experiment)
        # if self.current_adapter == "sherlock": ...

        full_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        print(f"{self.current_adapter.capitalize()}:", end=" ", flush=True)
        
        sampler = make_sampler(temp=temp, top_p=0.9)
        
        generate(
            self.model, 
            self.tokenizer, 
            prompt=full_prompt, 
            max_tokens=300, 
            verbose=True, 
            sampler=sampler
        )
        print("\n")

if __name__ == "__main__":
    MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    ADAPTERS_DIR = "adapters"

    bot = EfficientMultiLoRA(MODEL_ID, ADAPTERS_DIR)
    bot.chat_loop()
