```bash
#!/bin/bash

# 1. Download ProtT5-XL-U50 Pretrained Encoder Weights from official Hugging Face repository
echo "Downloading ProtT5-XL-U50 Pretrained Encoder Weights..."
wget -nc https://huggingface.co/Rostlab/prot_t5_xl_uniref50/resolve/main/pytorch_model.bin -O prott5_weights.bin

echo "Download completed. FlexBind trained models are already present in this directory."
