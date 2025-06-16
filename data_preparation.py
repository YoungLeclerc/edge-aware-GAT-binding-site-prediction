import re
import os
import h5py
import numpy as np
import torch as pt
from glob import glob
from tqdm import tqdm

from utilis.structure import clean_structure, tag_hetatm_chains, split_by_chain, filter_non_atomic_subunits, remove_duplicate_tagged_subunits
from utilis.data_encoding import config_encoding, encode_structure, extract_topology, extract_all_contacts
from utilis.dataset import StructuresDataset, save_data
from utilis.build_dssp  import parse_dssp_and_rsa, encode_features_dssp
from utilis.pdb_writer import save_structure_to_pdb

pt.multiprocessing.set_sharing_strategy('file_system')

config_dataset = {
    "r_thr": 5.0,
    "max_num_atoms": 1024*8,
    "max_num_nn": 64,
    "molecule_ids": np.array([...]),  # 省略长列表
    "pdb_filepaths": glob("data/all_biounits/*/*.pdb[0-9]*.gz"),
    "dataset_filepath": "data/datasets/contacts_mini.h5",
}
def contacts_types(s0, M0, s1, M1, ids, molecule_ids, device=pt.device("cpu")):
    c0 = pt.from_numpy(s0['resname'].reshape(-1,1) == molecule_ids.reshape(1,-1)).to(device)
    c1 = pt.from_numpy(s1['resname'].reshape(-1,1) == molecule_ids.reshape(1,-1)).to(device)
    H = (c1[ids[:,1]].unsqueeze(1) & c0[ids[:,0]].unsqueeze(2))
    rids0 = pt.where(M0[ids[:,0]])[1]
    rids1 = pt.where(M1[ids[:,1]])[1]
    Y = pt.zeros((M0.shape[1], M1.shape[1], H.shape[1], H.shape[2]), device=device, dtype=pt.bool)
    Y[rids0, rids1] = H
    T = pt.any(pt.any(Y, dim=1), dim=0)
    return Y, T

def pack_structure_data(X, qe, qr, qn, M, ids_topk):
    return {
        'X': X.cpu().numpy().astype(np.float32),
        'ids_topk': ids_topk.cpu().numpy().astype(np.uint16),
        'qe':pt.stack(pt.where(qe > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'qr':pt.stack(pt.where(qr > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'qn':pt.stack(pt.where(qn > 0.5), dim=1).cpu().numpy().astype(np.uint16),
        'M':pt.stack(pt.where(M), dim=1).cpu().numpy().astype(np.uint16),
    }, {
        'qe_shape': qe.shape, 'qr_shape': qr.shape, 'qn_shape': qn.shape, 'M_shape': M.shape,
    }

def pack_contacts_data(Y, T):
    return {
        'Y':pt.stack(pt.where(Y), dim=1).cpu().numpy().astype(np.uint16),
    }, {
        'Y_shape': Y.shape, 'ctype': T.cpu().numpy(),
    }

def pack_dataset_items(subunits, contacts, molecule_ids, max_num_nn, device=pt.device("cpu")):
    structures_data = {}
    contacts_data = {}

    for cid0 in contacts:
        s0 = subunits[cid0]

        # 写入临时PDB文件
        temp_pdb_path = f"/tmp/{cid0}.pdb"
        save_structure_to_pdb(s0, temp_pdb_path)

        # DSSP特征提取
        ss_codes, rsa_vals = parse_dssp_and_rsa(temp_pdb_path)
        qe0, qr0, qn0 = encode_features_dssp(s0, ss_codes, rsa_vals,
                                             std_elements=config_encoding['std_elements'],
                                             std_resnames=config_encoding['std_resnames'],
                                             std_names=config_encoding['std_names'],
                                             device=device)

        X0, M0 = encode_structure(s0, device=device)
        ids0_topk = extract_topology(X0, max_num_nn)[0]
        structures_data[cid0] = pack_structure_data(X0, qe0, qr0, qn0, M0, ids0_topk)

        if cid0 not in contacts_data:
            contacts_data[cid0] = {}

        for cid1 in contacts[cid0]:
            if cid1 not in contacts_data:
                contacts_data[cid1] = {}

            if cid1 not in contacts_data[cid0]:
                s1 = subunits[cid1]
                X1, M1 = encode_structure(s1, device=device)
                ctc_ids = contacts[cid0][cid1]['ids'].to(device)
                Y, T = contacts_types(s0, M0.to(device), s1, M1.to(device), ctc_ids, molecule_ids, device=device)
                if pt.any(Y):
                    contacts_data[cid0][cid1] = pack_contacts_data(Y, T)
                    contacts_data[cid1][cid0] = pack_contacts_data(Y.permute(1, 0, 3, 2), T.transpose(0, 1))
                pt.cuda.empty_cache()
    return structures_data, contacts_data

def store_dataset_items(hf, pdbid, bid, structures_data, contacts_data):
    metadata_l = []
    for cid0 in contacts_data:
        key = f"{pdbid.upper()[1:3]}/{pdbid.upper()}/{bid}/{cid0}"
        hgrp = hf.create_group(f"data/structures/{key}")
        save_data(hgrp, attrs=structures_data[cid0][1], **structures_data[cid0][0])
        for cid1 in contacts_data[cid0]:
            ckey = f"{key}/{cid1}"
            hgrp = hf.create_group(f"data/contacts/{ckey}")
            save_data(hgrp, attrs=contacts_data[cid0][cid1][1], **contacts_data[cid0][cid1][0])
            metadata_l.append({
                'key': key,
                'size': (np.max(structures_data[cid0][0]["M"], axis=0)+1).astype(int),
                'ckey': ckey,
                'ctype': contacts_data[cid0][cid1][1]["ctype"],
            })
    return metadata_l

if __name__ == "__main__":
    dataset = StructuresDataset(config_dataset['pdb_filepaths'], with_preprocessing=False)
    dataloader = pt.utils.data.DataLoader(dataset, batch_size=None, shuffle=True, num_workers=16, pin_memory=False, prefetch_factor=4)
    device = pt.device("cuda")
    with h5py.File(config_dataset['dataset_filepath'], 'w', libver='latest') as hf:
        for key in config_encoding:
            hf[f"metadata/{key}"] = config_encoding[key].astype(np.string_)
        hf["metadata/mids"] = config_dataset['molecule_ids'].astype(np.string_)

        metadata_l = []
        pbar = tqdm(dataloader)
        for structure, pdb_filepath in pbar:
            if structure is None:
                continue
            pdb_filepath = os.path.normpath(pdb_filepath)
            m = re.match(r'.*[/\\]([a-zA-Z0-9]+)\.pdb(\d+)\.gz', pdb_filepath)
            if m:
                pdbid, bid = m.group(1), m.group(2)
            else:
                continue
            if structure['xyz'].shape[0] >= config_dataset['max_num_atoms']:
                continue
            structure = clean_structure(structure)
            structure = tag_hetatm_chains(structure)
            subunits = split_by_chain(structure)
            subunits = filter_non_atomic_subunits(subunits)
            if len(subunits) < 2:
                continue
            subunits = remove_duplicate_tagged_subunits(subunits)
            contacts = extract_all_contacts(subunits, config_dataset['r_thr'], device=device)
            if len(contacts) == 0:
                continue
            structures_data, contacts_data = pack_dataset_items(
                subunits, contacts,
                config_dataset['molecule_ids'],
                config_dataset['max_num_nn'], device=device
            )
            metadata = store_dataset_items(hf, pdbid, bid, structures_data, contacts_data)
            metadata_l.extend(metadata)
            pbar.set_description(f"{metadata_l[-1]['key']}: {metadata_l[-1]['size']}")

        hf['metadata/keys'] = np.array([m['key'] for m in metadata_l]).astype(np.string_)
        hf['metadata/sizes'] = np.array([m['size'] for m in metadata_l])
        hf['metadata/ckeys'] = np.array([m['ckey'] for m in metadata_l]).astype(np.string_)
        hf['metadata/ctypes'] = np.stack(np.where(np.array([m['ctype'] for m in metadata_l])), axis=1).astype(np.uint32)
