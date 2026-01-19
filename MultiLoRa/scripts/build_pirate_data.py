import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import json
import os
import sys

# Configuration
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
OUTPUT_DIR = "MultiLoRa/data_pirate"

# Generic questions to force style transfer
GENERIC_PROMPTS = [
    "Explain how a microwave works.",
    "What is the capital of France?",
    "How do I write a Python for-loop?",
    "Why is the sky blue?",
    "Hello, who are you?",
    "What is the best way to boil an egg?",
    "Explain the theory of relativity.",
    "Can you help me debug my code?",
    "What is the weather like usually in London?",
    "Why do cats purr?",
    "How does the internet work?",
    "What is a smartphone?",
    "Give me advice on solving a puzzle.",
    "What is the meaning of life?",
    "How do airplanes fly?",
    "What is 2 + 2?",
    "Tell me a joke.",
    "How do I make a cup of tea?",
    "Explain quantum entanglement.",
    "What is the difference between a violin and a cello?",
    "What are your thoughts on authority?",
    "Describe the ocean.",
    "How do I navigate a map?",
    "What is the best way to hide treasure?",
    "Who is the captain here?"
]

def load_model():
    print(f"[INFO] Loading model: {BASE_MODEL}")
    model, tokenizer = load(BASE_MODEL)
    return model, tokenizer

def generate_pirate_data(model, tokenizer):
    """
    Forces the model to answer generic questions in the persona of a Pirate.
    """
    dataset = []
    print(f"[INFO] Generating {len(GENERIC_PROMPTS)} Pirate style pairs...")
    
    for i, user_q in enumerate(GENERIC_PROMPTS):
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a rugged, salt-crusted Pirate Captain from the 17th century. "
            f"You use words like 'Yarr', 'Matey', 'Landlubber', and nautical metaphors. "
            f"You are superstitious and loud. "
            f"Answer the user's question in character. Do not break character.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_q}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        # High temp for chaotic pirate energy
        sampler = make_sampler(temp=0.8)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False, sampler=sampler)
        
        dataset.append({
            "messages": [
                {"role": "user", "content": user_q},
                {"role": "assistant", "content": response.strip()}
            ]
        })
        print(f"  [Pirate] {i+1}/{len(GENERIC_PROMPTS)} generated.", end='\r')
    print("")
    return dataset

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model, tokenizer = load_model()
    
    combined_data = generate_pirate_data(model, tokenizer)
    
    # Repeat the data 4 times to make the dataset big enough for 200 iterations
    # (Since we don't have a book to bulk it up)
    combined_data = combined_data * 4
    
    output_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    
    with open(output_path, "w") as f:
        for entry in combined_data:
            f.write(json.dumps(entry) + '\n')
            
    # Create valid/test copies
    import shutil
    shutil.copy(output_path, os.path.join(OUTPUT_DIR, "valid.jsonl"))
    shutil.copy(output_path, os.path.join(OUTPUT_DIR, "test.jsonl"))
    
    print(f"[SUCCESS] Generated {len(combined_data)} training examples.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
