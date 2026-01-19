# MLX Fine-Tuning Lab 🧪

A hands-on collection of experiments for fine-tuning Large Language Models (LLMs) locally on Apple Silicon using the [MLX Framework](https://github.com/ml-explore/mlx).

This repository contains two distinct projects demonstrating **QLoRA (Quantized Low-Rank Adaptation)** workflows.

## 📂 Projects

### 1. [SingleLoRA (Project Sherlock)](./SingleLoRA)
A complete pipeline to fine-tune Llama-3.2-3B to speak in the persona of **Sherlock Holmes** while retaining general knowledge (e.g., explaining physics).
*   **Goal:** Create a single, highly specialized adapter.
*   **Key Concepts:** Data Augmentation (Self-Play), Style Transfer, Catastrophic Forgetting Mitigation.
*   **Location:** `SingleLoRA/`

### 2. [MultiLoRa (Persona Switching)](./MultiLoRa)
An advanced architecture demonstrating how to serve **multiple fine-tuned adapters** simultaneously from a single base model.
*   **Goal:** Switch personalities instantly without reloading the base model.
*   **Key Concepts:** Hot-Swapping, Memory Efficiency, Multi-Model Serving.
*   **Personas:** Sherlock Holmes vs. 17th Century Pirate.
*   **Location:** `MultiLoRa/`

## 🚀 Quick Start

1.  **Environment Setup**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

2.  **Run the Single Adapter Demo**
    ```bash
    # Chat with the pre-trained Sherlock adapter
    python3 SingleLoRA/src/chat.py
    ```

3.  **Run the Multi-Adapter Lab**
    ```bash
    # Switch between Sherlock and Pirate in real-time
    python3 MultiLoRa/src/memory_efficient_chat.py
    ```

## 🧠 Hardware Requirements

*   **Platform:** Apple Silicon (M1/M2/M3/M4).
*   **RAM:** 8GB minimum (16GB+ recommended).
*   **Models:** Examples utilize `Llama-3.2-3B-4bit` for maximum efficiency (runs on ~2GB RAM).

## 📚 Technical Learnings

*   **Data Quality:** Simply feeding raw book text creates a "Parrot." Mixing "Instruction Tuning" (Q&A) with "Style Transfer" (Generic questions) creates an "Actor."
*   **Efficient Serving:** Using `load(adapter_path=...)` allows hot-swapping 16MB adapter files without reloading the 2GB base model, enabling multi-tenant architectures on consumer hardware.

## License
Source text courtesy of Project Gutenberg (Public Domain). Code is MIT.