# Weights Directory

This directory contains the necessary model weights for running FlexBind.

1. **FlexBind Trained Models** (Included in this repository):
   - `flexbind_dp81.pt`: Model trained on the DP81 dataset.
   - `flexbind_dp93.pt`: Model trained on the DP93 dataset.

2. **ProtT5 Encoder Weights**:
   - Due to size limits, the base ProtT5 encoder weights are not included directly. 
   - Please run `bash download_weights.sh` to download `prott5_weights.bin` before running the evaluation or training scripts.
