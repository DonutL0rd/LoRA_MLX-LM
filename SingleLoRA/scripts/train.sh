#!/bin/bash
# Train the Sherlock Model

echo "Starting Fine-Tuning..."
echo "Model: mlx-community/Llama-3.2-3B-Instruct-4bit"
echo "Config: lora_config.yaml"
echo "Data: data/"

# Ensure we are in the root of the demo folder
# (Assuming script is run from root like ./scripts/train.sh)

mlx_lm.lora \
    --config lora_config.yaml \
    --train \

echo "Training Complete. Adapters saved to 'adapters/'."

