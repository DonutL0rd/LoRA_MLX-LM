import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import json
import os
import sys
import re
import random

# Configuration
BASE_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
SOURCE_TEXT = "data/treasure_island.txt"
OUTPUT_DIR = "data/data_pirate"

# Expanded list of modern/generic topics
GENERIC_TOPICS = [
    "how a car engine works", "the theory of relativity", "how to bake a cake", 
    "why the sky is blue", "how to write Python code", "what is photosynthesis",
    "the history of the internet", "how to tie a tie", "what is a black hole",
    "how to invest money", "the rules of soccer", "why cats purr",
    "how to fix a leaking faucet", "what is artificial intelligence", "how to meditate",
    "the water cycle", "how electricity works", "what is dna",
    "how to make coffee", "the distance to the moon", "how to solve a rubik's cube",
    "what is blockchain", "how vaccines work", "why birds fly",
    "the capital of Japan", "how to play chess", "what is global warming",
    "how to paint a wall", "what is a neuron", "how to loose weight",
    "what is 2 + 2", "tell me a joke", "write a poem", "who are you",
    "what is the meaning of life", "how to make friends", "what is love",
    "explain quantum physics", "how to build a house", "what is linux"
]

def load_model():
    print(f"[INFO] Loading model: {BASE_MODEL}")
    model, tokenizer = load(BASE_MODEL)
    return model, tokenizer

def generate_book_data(model, tokenizer, limit=100):
    """
    Reads Treasure Island and generates Q&A pairs.
    """
    if not os.path.exists(SOURCE_TEXT):
        print(f"[ERROR] {SOURCE_TEXT} not found.")
        return []

    print("[INFO] Reading Treasure Island...")
    with open(SOURCE_TEXT, "r", encoding='utf-8') as f:
        text = f.read()

    # Gutenberg Header/Footer Cleanup
    if "*** START OF THE PROJECT GUTENBERG EBOOK" in text:
        text = text.split("*** START OF THE PROJECT GUTENBERG EBOOK")[1]
    if "*** END OF THE PROJECT GUTENBERG EBOOK" in text:
        text = text.split("*** END OF THE PROJECT GUTENBERG EBOOK")[0]
    
    paragraphs = [p.replace('\n', ' ').strip() for p in text.split('\n\n') if len(p.strip()) > 100]
    selected_paragraphs = paragraphs[:limit]
    
    dataset = []
    print(f"[INFO] Generating {len(selected_paragraphs)} pairs from book text...")
    
    for i, paragraph in enumerate(selected_paragraphs):
        prompt = (
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Here is a text excerpt:\n'{paragraph}'\n\n"
            f"Write a short question that the text above answers. "
            f"The answer should be the text itself.<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        sampler = make_sampler(temp=0.6)
        question = generate(model, tokenizer, prompt=prompt, max_tokens=60, verbose=False, sampler=sampler)
        question = question.strip()
        
        if len(question) < 5 or "?" not in question:
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

def generate_style_data(model, tokenizer, topics):
    """
    Forces the model to answer generic questions in the persona of a Pirate.
    """
    dataset = []
    print(f"[INFO] Generating {len(topics)} Pirate style pairs...")
    
    for i, topic in enumerate(topics):
        # We ask the model to generate the Question AND the Answer
        prompt = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"You are a creative writer.<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"Write a dialogue where a User asks about '{topic}' and a Pirate Captain answers.\n"
            f"The Pirate must use heavy slang (Yarr, Matey) but explain the concept correctly.\n"
            f"Format:\nUser: [Question]\nPirate: [Answer]<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        
        sampler = make_sampler(temp=0.8)
        response = generate(model, tokenizer, prompt=prompt, max_tokens=400, verbose=False, sampler=sampler)
        
        # Parse the output
        try:
            if "User:" in response and "Pirate:" in response:
                parts = response.split("Pirate:")
                user_part = parts[0].replace("User:", "").strip()
                pirate_part = parts[1].strip()
                
                dataset.append({
                    "messages": [
                        {"role": "user", "content": user_part},
                        {"role": "assistant", "content": pirate_part}
                    ]
                })
        except:
            pass
            
        print(f"  [Style] {i+1}/{len(topics)} generated.", end='\r')
    print("")
    return dataset

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    model, tokenizer = load_model()
    
    # 1. Get Book Data (Vocabulary)
    # Reduced to 50 to prioritize style over raw text recitation
    data_book = generate_book_data(model, tokenizer, limit=50)
    
    # 2. Get Style Data (Modern concepts -> Pirate)
    # We loop through topics 3 times to get variations
    data_style = generate_style_data(model, tokenizer, GENERIC_TOPICS * 3)
    
    # Combine
    combined_data = data_book + data_style
    random.shuffle(combined_data)
    
    output_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    
    with open(output_path, "w") as f:
        for entry in combined_data:
            f.write(json.dumps(entry) + '\n')
            
    # Valid/Test
    shutil_import = __import__('shutil')
    shutil_import.copy(output_path, os.path.join(OUTPUT_DIR, "valid.jsonl"))
    shutil_import.copy(output_path, os.path.join(OUTPUT_DIR, "test.jsonl"))
    
    print(f"[SUCCESS] Generated {len(combined_data)} training examples.")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
