import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import sys
import os

class MultiPersonaChat:
    def __init__(self, base_model_path, adapters_dir):
        print("\n" + "="*50)
        print("🎭  LOADING MULTI-LORA SYSTEM")
        print("="*50)
        
        self.models = {}
        self.tokenizers = {}
        self.current_persona = "base"
        
        # 1. Load Base Model
        print(f"[1/3] Loading Base Model...")
        self.models["base"], self.tokenizers["base"] = load(base_model_path)
        
        # 2. Load Sherlock
        sherlock_path = os.path.join(adapters_dir, "sherlock")
        if os.path.exists(sherlock_path):
            print(f"[2/3] Loading Sherlock Persona...")
            self.models["sherlock"], self.tokenizers["sherlock"] = load(base_model_path, adapter_path=sherlock_path)
        else:
            print(f"[WARNING] Sherlock adapter not found at {sherlock_path}")

        # 3. Load Pirate
        pirate_path = os.path.join(adapters_dir, "pirate")
        if os.path.exists(pirate_path):
            print(f"[3/3] Loading Pirate Persona...")
            self.models["pirate"], self.tokenizers["pirate"] = load(base_model_path, adapter_path=pirate_path)
        else:
            print(f"[WARNING] Pirate adapter not found at {pirate_path}")
            
        print("\n[INFO] All personas loaded into RAM. Ready to switch instantly.")

    def chat_loop(self):
        print("\n" + "="*50)
        print("🗣️  MULTI-PERSONA CHAT")
        print("Commands: /sherlock, /pirate, /base, /quit")
        print("="*50 + "\n")

        while True:
            try:
                # Show who is active
                prompt_label = f"You ({self.current_persona.upper()}): "
                user_input = input(prompt_label)
                
                # Handle Commands
                if user_input.startswith("/"):
                    cmd = user_input.lower().strip()
                    if cmd in ["/quit", "/exit"]:
                        break
                    elif cmd == "/sherlock":
                        if "sherlock" in self.models:
                            self.current_persona = "sherlock"
                            print("--> Switched to SHERLOCK HOLMES 🕵️‍♂️")
                        else:
                            print("--> Sherlock model not loaded.")
                        continue
                    elif cmd == "/pirate":
                        if "pirate" in self.models:
                            self.current_persona = "pirate"
                            print("--> Switched to PIRATE CAPTAIN 🏴‍☠️")
                        else:
                            print("--> Pirate model not loaded.")
                        continue
                    elif cmd == "/base":
                        self.current_persona = "base"
                        print("--> Switched to BASE ASSISTANT 🤖")
                        continue
                    else:
                        print("--> Unknown command.")
                        continue
                
                # Generate
                self.generate_response(user_input)
                
            except KeyboardInterrupt:
                print("\nExiting.")
                break

    def generate_response(self, text):
        # 1. Select System Prompt based on Persona
        system_prompt = ""
        if self.current_persona == "sherlock":
            system_prompt = (
                "You are Sherlock Holmes. You are arrogant, hyper-observant, and use Victorian vocabulary. "
                "Answer the user's question in character."
            )
            # Sherlock Settings
            temp = 0.8
            top_p = 0.9
        elif self.current_persona == "pirate":
            system_prompt = (
                "You are a rugged 17th-century pirate captain. "
                "You use words like 'Yarr', 'Matey', 'Landlubber'. Superstitious and loud."
            )
            # Pirate Settings (Chaos)
            temp = 0.9
            top_p = 0.95
        else: # Base
            system_prompt = "You are a helpful AI assistant."
            temp = 0.6
            top_p = 0.9

        # 2. Construct Prompt
        full_prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_prompt}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        print(f"{self.current_persona.capitalize()}:", end=" ", flush=True)
        
        sampler = make_sampler(temp=temp, top_p=top_p)
        
        generate(
            self.models[self.current_persona], 
            self.tokenizers[self.current_persona], 
            prompt=full_prompt, 
            max_tokens=300, 
            verbose=True, 
            sampler=sampler
        )
        print("\n")

if __name__ == "__main__":
    MODEL_ID = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    
    # Robust Path Finding
    # Get the absolute path of this script (src/multi_chat.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root (MultiLoRa/)
    project_root = os.path.dirname(script_dir)
    # Point to adapters/
    ADAPTERS_DIR = os.path.join(project_root, "adapters")

    print(f"[DEBUG] Looking for adapters in: {ADAPTERS_DIR}")

    bot = MultiPersonaChat(MODEL_ID, ADAPTERS_DIR)
    bot.chat_loop()
