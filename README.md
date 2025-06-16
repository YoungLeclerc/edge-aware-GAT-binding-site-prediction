# Edge-aware GAT for Protein Binding Site Prediction

A deep learning framework for predicting protein binding sites across five molecule types, integrating secondary structure (via DSSP) and relative surface accessibility into an edge-aware graph attention network.

## Features

- Residue-level binding site prediction
- Incorporates DSSP-derived secondary structure and RSA features
- Supports multi-molecule binding prediction (e.g., protein, DNA/RNA, ion, ligand, lipid)
- PyTorch-based, efficient GPU inference
- PyMOL-compatible structural output visualization

## Requirements

- Python ≥ 3.7
- PyTorch ≥ 1.10
- Biopython
- DSSP (`mkdssp` CLI tool, must be installed separately)
- NumPy, h5py, matplotlib, tqdm

## Quick Start

1. Install dependencies:

pip install -r requirements.txt


2. Preprocess data and build a dataset:
python preparation.py

3. Train and evaluate the model:
python main.py

4. Visualization results (loading pdb files in PyMOL)
load example_protein.pdb
spectrum b, green_white_red, minimum=0, maximum=1



