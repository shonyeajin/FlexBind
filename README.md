# FlexBind: Multi-Scale Complex Learning for Imbalance-Robust IDR and Binding Residue Prediction

A deep learning framework for residue-level prediction of protein properties, including:
- Intrinsically Disordered Regions (IDRs)
- Protein-binding residues
- RNA-binding residues
- DNA-binding residues

---

## 📌 Reproducibility
To fully satisfy requirements for generalization, experimental validity, and methodological rigor, this repository explicitly provides:
- **Exact Preprocessing & Split Commands**: Provided in `data/preprocess.sh`
- **Configuration Files & Random Seeds**: Defined in `configs/default_config.yaml`
- **Pretrained Weights**: Trained FlexBind models are provided in the `weights/` directory. The base ProtT5 weights can be downloaded via `weights/download_weights.sh`.
- **Baseline Instructions**: Detailed guide located in `baselines/baseline_instructions.md`
- **Environment Versions**: Exact dependencies are exported in `environment.yml`

---

## Installation

```bash
conda env create -f environment.yml
conda activate flexbind
```

## Pretrained Weights
The trained FlexBind models (flexbind_dp81.pt, flexbind_dp93.pt, flexbind_dp94.pt) are already included in the weights/ directory.

Before running the model, you only need to download the base ProtT5 encoder weights by running the following script:

```bash
cd weights
bash download_weights.sh
cd ..
```

## Usage
Run model training and evaluation:

```bash
python scripts/train.py --npz_path data/processed/DP81/train_val.npz --test_npz_path data/processed/DP81/test.npz
```
