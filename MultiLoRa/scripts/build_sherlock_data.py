import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import json
import os
import sys

# Configuration
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
SOURCE_TEXT = "data/sherlock.txt"
OUTPUT_DIR = "data"

# Generic questions to force style transfer on modern topics
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
    "What is the difference between a violin and a cello?"
]

def load_model():
    print(f"[INFO] Loading model: {BASE_MODEL}")
    model, tokenizer = load(BASE_MODEL)
    return model, tokenizer

def generate_book_data(model, tokenizer, limit=50):
    """
    Reads the book and generates Q&A pairs based on the text.
    """
    if not os.path.exists(SOURCE_TEXT):
        print(f"[ERROR] {SOURCE_TEXT} not found.")
        return []

    print("[INFO] Reading source text...")
    with open(SOURCE_TEXT, "r") as f:
        text = f.read()

    # Simple cleanup
    if "*** START" in text:
        text = text.split("*** START")[1]
    text = text.split("*** END")[0]
    
    # Extract decent-sized paragraphs
    paragraphs = [p.replace('\n', ' ').strip() for p in text.split('\n\n') if len(p.strip()) > 150]
    selected_paragraphs = paragraphs[:limit]
    
    dataset = []
    print(f"[INFO] Generating {len(selected_paragraphs)} pairs from source text...")
    
    for i, paragraph in enumerate(selected_paragraphs):
        # Prompt model to generate a question about the paragraph
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Here is a text excerpt:\n'{paragraph}'\n\n"
            f"Write a short, direct question that the text above answers. "
            f"Do not mention 'the text'. Just ask the question.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        sampler = make_sampler(temp=0.4)
        question = generate(model, tokenizer, prompt=prompt, max_tokens=50, verbose=False, sampler=sampler)
        question = question.strip()
        
        if len(question) < 10 or "?" not in question:
            continue
            
        dataset.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": paragraph}
            ]
        })
        print(f"  [Book] {i+1}/{limit} generated.", end='\r')
    print("")
    return dataset

def generate_general_data(model, tokenizer):
    """
    Forces the model to answer generic questions in the persona of Sherlock.
    """
    dataset = []
    print(f"[INFO] Generating {len(GENERIC_PROMPTS)} general style pairs...")
    
    for i, user_q in enumerate(GENERIC_PROMPTS):
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are Sherlock Holmes. You are arrogant, hyper-observant, and use Victorian vocabulary. "
            f"Answer the user's question in character. Do not break character.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_q}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        sampler = make_sampler(temp=0.7)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False, sampler=sampler)
        
        dataset.append({
            "messages": [
                {"role": "user", "content": user_q},
                {"role": "assistant", "content": response.strip()}
            ]
        })
        print(f"  [Style] {i+1}/{len(GENERIC_PROMPTS)} generated.", end='\r')
    print("")
    return dataset

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model, tokenizer = load_model()
    
    data_book = generate_book_data(model, tokenizer, limit=50)
    data_style = generate_general_data(model, tokenizer)
    
    combined_data = data_book + data_style
    
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
