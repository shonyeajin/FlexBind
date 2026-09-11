#!/bin/bash

# 1. Download ProtT5-XL-U50 Pretrained Encoder Weights from official Hugging Face repository
echo "Downloading ProtT5-XL-U50 Pretrained Encoder Weights..."
wget -nc https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/main/pytorch_model.bin -O prott5_weights.bin

# 2. Download FlexBind Pretrained Final Weights from Google Drive
echo "Downloading FlexBind Pretrained Final Weights from Google Drive..."
# Install gdown to download from Google Drive folder
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1_bvX7NDH632E4ju6NmUOi1DO_ZDu2odq?usp=drive_link -O ./

echo "Download completed. Please ensure all .pt files are located in the weights/ directory."
