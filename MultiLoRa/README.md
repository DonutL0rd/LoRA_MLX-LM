# Multi-LoRA Lab

This project demonstrates **Multi-Adapter Serving**. It shows how to switch between different fine-tuned personas (Sherlock Holmes and a Pirate Captain) dynamically without reloading the base model parameters.

## Structure

*   `adapters/`: Contains the fine-tuned LoRA weights.
    *   `sherlock/`: Trained on *The Adventures of Sherlock Holmes*.
    *   `pirate/`: Trained on synthetic pirate dialogue.
*   `src/`:
    *   `memory_efficient_chat.py`: **(Recommended)** Swaps adapters dynamically (Low RAM usage).
    *   `multi_chat.py`: Loads all models into RAM simultaneously (High RAM usage comparison).
*   `scripts/`: Data generation scripts for creating the personas.

## Usage

**Run the Switcher:**
```bash
python3 src/memory_efficient_chat.py
```

**Commands:**
*   `/sherlock` - Switch context to Holmes.
*   `/pirate` - Switch context to Captain.
*   `/base` - Revert to default Assistant.