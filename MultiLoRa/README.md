# Multi-LoRA Lab 🏴‍☠️🕵️‍♂️

This project demonstrates **Multi-Adapter Serving**. It shows how to switch between different fine-tuned personas (Sherlock Holmes and a Pirate Captain) instantly without reloading the heavy base model.

## 📂 Structure

*   `adapters/`: Contains the fine-tuned LoRA weights.
    *   `sherlock/`: Trained on *The Adventures of Sherlock Holmes*.
    *   `pirate/`: Trained on synthetic pirate dialogue.
*   `src/`:
    *   `memory_efficient_chat.py`: **(Recommended)** Swaps adapters dynamically (Low RAM).
    *   `multi_chat.py`: Loads all models into RAM simultaneously (High RAM demo).
*   `scripts/`: Data generation scripts for creating the personas.

## 🕹 Usage

**Run the Switcher:**
```bash
python3 src/memory_efficient_chat.py
```

**Commands:**
*   `/sherlock` - Switch to Holmes.
*   `/pirate` - Switch to Captain.
*   `/base` - Revert to default AI.
