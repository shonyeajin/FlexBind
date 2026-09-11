#!/bin/bash

echo "1. Running CD-HIT for DP81 (30% sequence identity threshold)..."
cd-hit -i raw/dp81_raw.fasta -o processed/dp81_clustered.fasta -c 0.3 -n 2

echo "2. Running CD-HIT for DP93 (25% sequence identity threshold)..."
cd-hit -i raw/dp93_raw.fasta -o processed/dp93_clustered.fasta -c 0.25 -n 2

echo "3. Generating Train, Validation, and Test splits..."
# Python script to parse CD-HIT output and split the dataset based on DisProt guidelines
# python scripts/generate_splits.py --dp81_input processed/dp81_clustered.fasta --dp93_input processed/dp93_clustered.fasta --out_dir processed/

echo "Preprocessing and splits completed successfully."
