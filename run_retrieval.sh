#!/bin/bash

HUGGINGFACE_TOKEN=""
PROJECT_NAME="test"
SWEEP_CONFIG="./config/retrieval_sweep.yaml"
COUNT=2

use_sweep=false

# 1. Check huggingface-cli login status
login_status=$(huggingface-cli whoami)

if [[ $login_status == *"Not logged in"* ]]; then
    echo "Not logged in to huggingface-cli"
    huggingface-cli login --token ${HUGGINGFACE_TOKEN}
elif [[ $login_status == *"Invalid user token"* ]]; then
    echo "Invalid user token. Logging in again."
    huggingface-cli login --token ${HUGGINGFACE_TOKEN}
else
    echo "already logged in to Hugging Face. User: ${login_status}"
fi

# 2. install requirements
# pip3 install -r requirements.txt

# 3. train model (by wandb sweep or wandb run)
if [[ "${use_sweep}" = true ]]; then
    SWEEP_OUTPUT=$(wandb sweep -p ${PROJECT_NAME} ${SWEEP_CONFIG} 2>&1)
    SWEEP_ID=$(echo ${SWEEP_OUTPUT} | grep -o "wandb agent .*" | cut -d' ' -f3-)
    wandb agent --count ${COUNT} ${SWEEP_ID}
else
    python3 train_retrieval.py
fi

# 4. run inference.py