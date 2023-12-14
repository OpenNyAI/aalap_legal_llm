#!/bin/bash

# Install PyTorch and related packages from the PyTorch nightly build for CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
wait

# Install ninja for faster flash attention build
pip3 install ninja
wait

# Install Flash-Attn
pip install -U flash-attn
wait

# Install the Peft library
pip install git+https://github.com/huggingface/peft
wait

# Install additional Python packages
pip install transformers accelerate openai trl huggingface-hub bitsandbytes evaluate deepspeed datasets wandb tiktoken scipy jupyter notebook tensorboard
wait

# Setup hugging face cli 
echo "Setting up hugging-face. Please provide a read token of hugging-face"
huggingface-cli login
wait

# Setup wandb
echo "Setting up Wandb. Please provide a token of Wandb"
wandb login
