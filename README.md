# FlexBind

A deep learning framework for residue-level prediction of protein properties, including:

- Intrinsically Disordered Regions (IDRs)
- Protein-binding residues
- RNA-binding residues
- DNA-binding residues

---

## Overview

FlexBind is a sequence-based model designed to handle:

- Multi-scale interaction patterns along protein sequences
- Structural organization of functional residues
- Severe class imbalance in residue-level prediction tasks

The framework leverages pretrained protein embeddings and multi-scale representations to model diverse interaction behaviors.

---

## Repository Structure

  FlexBind/  
  │  
  ├── model.py                
  ├── environment.yaml         
  ├── README.md  
  │  
  ├── DP81/  
  ├── DP93/  
  ├── DP94/  

---

## Installation

conda env create -f environment.yaml  
conda activate flexbind

---

## Usage

Run model (example):  

python model.py  


---

## Dataset

This repository uses DisProt-derived datasets:  

- DP81  
- DP93  
- DP94  

