# SingleLoRA (Project Sherlock)

This directory demonstrates a **Single-Adapter Fine-Tuning** workflow.

The objective is **Style Transfer + Knowledge Retention**: Creating a model that adopts the persona of **Sherlock Holmes** while retaining the ability to answer modern general-knowledge questions (e.g., explaining microwaves or Python code).

## Repository Structure

*   `src/chat.py`: Interactive CLI for chatting with the trained model.
*   `scripts/build_dataset.py`: Generates synthetic training data using Self-Play and Style Transfer techniques.
*   `scripts/train.sh`: Launcher for the LoRA fine-tuning process.
*   `data/`: Contains source text (`sherlock.txt`) and generated JSONL datasets.
*   `adapters/`: Contains the pre-trained LoRA weights (Rank 8).
*   `lora_config.yaml`: Training hyperparameters.

## Quick Start (Pre-Trained)

You can run inference with the pre-trained model immediately.

1.  **Run Inference:**
    ```bash
    python3 src/chat.py
    ```

## Reproduction Steps (Do It Yourself)

To train the model from scratch:

### 1. Generate Dataset
This script uses the base LLM to read *The Adventures of Sherlock Holmes* and generate Q&A pairs ("Self-Play"). It also injects general knowledge questions to prevent catastrophic forgetting.

```bash
python3 scripts/build_dataset.py
```
*Output:* `data/train.jsonl`

### 2. Run Training
Fine-tunes the model using LoRA (Low-Rank Adaptation).

```bash
chmod +x scripts/train.sh
./scripts/train.sh
```
*   **Time:** ~5 minutes on M4/M3/M2 chips.
*   **Iterations:** 200 (optimized for small datasets).
*   **Config:** `lora_config.yaml`

### 3. Test
Run the chat script again to verify the new adapters.

## Technical Details

*   **Base Model:** `mlx-community/Llama-3.2-3B-Instruct-4bit`
*   **Fine-Tuning Method:** QLoRA (Quantized LoRA)
*   **Rank:** 8
*   **Target Modules:** 16 layers
*   **Data Strategy:** 50/50 split between "Book Context" (Sherlock stories) and "Style Transfer" (Modern questions rewritten in Holmes' voice).

## License
Source text courtesy of Project Gutenberg (Public Domain). Code is MIT.
