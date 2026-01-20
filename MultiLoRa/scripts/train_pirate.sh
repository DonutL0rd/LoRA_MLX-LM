#!/bin/bash
# Train the Pirate Model

# 1. Resolve Project Root (Robust against where the script is called from)
# Get the absolute path of the directory containing this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Go up two levels: MultiLoRa/scripts -> MultiLoRa -> Root (ml)
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 2. Change execution context to Project Root
cd "$PROJECT_ROOT" || exit
echo "[INFO] Running from Project Root: $(pwd)"

# 3. Activate venv (if not already active)
if [[ "$VIRTUAL_ENV" == "" ]]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    else
        echo "[WARN] venv/bin/activate not found. Assuming environment is set up."
    fi
fi

echo "Starting Pirate Fine-Tuning..."
echo "Model: mlx-community/Llama-3.2-3B-Instruct-4bit"
echo "Data: MultiLoRa/data/data_pirate"
echo "Output: MultiLoRa/adapters/pirate"

# 4. Run Training
# Updated syntax to avoid deprecation warning (python -m mlx_lm.lora -> mlx_lm.lora)
# We pipe to 'tee' to see output in real-time AND save it to a log file for the plotter.
LOG_FILE="MultiLoRa/training_log.txt"
echo "[INFO] Logging training progress to $LOG_FILE"

python3 -m mlx_lm.lora \
    --config MultiLoRa/lora_config.yaml \
    --model mlx-community/Llama-3.2-3B-Instruct-4bit \
    --data MultiLoRa/data/data_pirate \
    --adapter-path MultiLoRa/adapters/pirate \
    --train | tee "$LOG_FILE"

echo "Training Complete. Adapters saved to 'MultiLoRa/adapters/pirate'."