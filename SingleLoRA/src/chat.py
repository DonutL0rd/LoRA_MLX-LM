import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import sys
import os
import time

class ComparisonChat:
    def __init__(self, model_path, adapter_path):
        print("\n" + "="*50)
        print("⚖️  LOADING MODELS FOR COMPARISON")
        print("="*50)
        
        # 1. Load Base Model
        print(f"[1/2] Loading Base Model ({model_path})...")
        self.base_model, self.base_tokenizer = load(model_path)
        
        # 2. Load Fine-Tuned Model
        print(f"[2/2] Loading Sherlock Model (Adapters: {adapter_path})...")
        self.tuned_model, self.tuned_tokenizer = load(model_path, adapter_path=adapter_path)
        
        print("\n[INFO] System Ready. 24GB RAM Flex 💪")

    def chat_loop(self):
        print("\n" + "="*50)
        print("🧪  SHERLOCK VS. BASE LLAMA")
        print("Type 'quit' to exit.")
        print("="*50 + "\n")

        while True:
            try:
                user_input = input("\nYou: ")
                if user_input.lower() in ["quit", "exit"]:
                    print("Exiting.")
                    break
                
                self.compare_responses(user_input)
            except KeyboardInterrupt:
                print("\nExiting.")
                break

    def compare_responses(self, text):
        # --- 1. Base Model Generation (Standard Assistant) ---
        base_prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        print("\n" + "-"*20 + " 🤖 BASE LLAMA " + "-"*20)
        print(f"(Thinking...)", end="\r")
        
        # Base settings
        sampler = make_sampler(temp=0.6, top_p=0.9)
        
        generate(
            self.base_model, 
            self.base_tokenizer, 
            prompt=base_prompt, 
            max_tokens=300, 
            verbose=True, 
            sampler=sampler
        )

        # --- 2. Tuned Model Generation (Sherlock Persona) ---
        # WE MUST INJECT THE SYSTEM PROMPT!
        # This matches how we generated the training data.
        sherlock_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are Sherlock Holmes. You are arrogant, hyper-observant, and use Victorian vocabulary. "
            f"Answer the user's question in character. Do not break character.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        print("\n\n" + "-"*20 + " 🕵️‍♂️ SHERLOCK " + "-"*20)
        print(f"(Deducing...)", end="\r")
        
        # Higher temp for creative writing
        sherlock_sampler = make_sampler(temp=0.8, top_p=0.9)
        
        generate(
            self.tuned_model, 
            self.tuned_tokenizer, 
            prompt=sherlock_prompt, 
            max_tokens=300, 
            verbose=True, 
            sampler=sherlock_sampler
        )
        print("\n")

if __name__ == "__main__":
    MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    ADAPTER_DIR = "adapters"
    
    # Allow overriding adapter path via CLI
    if len(sys.argv) > 1:
        ADAPTER_DIR = sys.argv[1]

    if not os.path.exists(ADAPTER_DIR) and not os.path.exists(os.path.join("demo", ADAPTER_DIR)):
        # Fallback to demo/adapters if running from root
        if os.path.exists(os.path.join("demo", "adapters")):
            ADAPTER_DIR = "demo/adapters"
        else:
            print(f"[ERROR] Adapter directory '{ADAPTER_DIR}' not found.")
            sys.exit(1)

    bot = ComparisonChat(MODEL_ID, ADAPTER_DIR)
    bot.chat_loop()