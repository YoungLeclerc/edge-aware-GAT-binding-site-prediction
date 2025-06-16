from Bio.PDB import PDBParser, DSSP
import numpy as np
import pandas as pd

SS_MAP = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}
MAX_ACC = {'A': 121, 'R': 265, 'N': 187, 'D': 187, 'C': 148, 'Q': 214, 'E': 214, 'G': 97,
           'H': 216, 'I': 195, 'L': 191, 'K': 230, 'M': 203, 'F': 228, 'P': 154, 'S': 143,
           'T': 163, 'W': 264, 'Y': 255, 'V': 165}

def one_hot_ss(ss):
    vec = [0] * 8
    vec[SS_MAP.get(ss, 7)] = 1
    return vec

def compute_dssp_features(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('X', pdb_file)
    model = structure[0]

    dssp = DSSP(model, pdb_file)
    features = []
    for key in dssp.keys():
        aa = dssp[key][1]
        ss = dssp[key][2]
        asa = dssp[key][3]
        max_acc = MAX_ACC.get(aa.upper(), 150)
        rsa = min(asa / max_acc, 1.0)
        one_hot = one_hot_ss(ss)
        features.append(one_hot + [rsa])
    return np.array(features)  # shape: [n_residues, 9]
