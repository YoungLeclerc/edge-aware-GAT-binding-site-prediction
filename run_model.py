import os
import sys
import h5py
import json
import numpy as np
import torch as pt
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob

# 自定义模块导入
from utilis.dataset import StructuresDataset, collate_batch_features, select_by_sid, select_by_interface_types
from utilis.data_encoding import encode_structure, encode_features, extract_topology, categ_to_resnames, resname_to_categ
from utilis.structure import data_to_structure, encode_bfactor, concatenate_chains, split_by_chain
from utilis.structure_io import save_pdb, read_pdb
from utilis.scoring import bc_scoring, bc_score_names

# 设置数据路径
DATA_PATH = "examples/My_test_pdb"

# 选择模型路径
SAVE_PATH = "model/save/i_v4_1_2021-09-07_11-21"
MODEL_FILEPATH = os.path.join(SAVE_PATH, 'model_ckpt.pt')

# 添加模块路径
if SAVE_PATH not in sys.path:
    sys.path.insert(0, SAVE_PATH)

# 导入配置和模型
from config import config_model, config_data
from data_handler import Dataset
from model import Model


def load_model():
    """加载 PyTorch 预训练模型"""
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 初始化模型
    model = Model(config_model)

    # 加载权重
    model.load_state_dict(pt.load(MODEL_FILEPATH, map_location=device))

    # 设置模型为推理模式
    return model.eval().to(device)


def predict_and_save_results(model, dataset):
    """运行模型并保存带预测结果的 PDB 结构"""
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    with pt.no_grad():
        for subunits, filepath in tqdm(dataset, desc="Processing PDB files"):
            # 组合所有链
            structure = concatenate_chains(subunits)

            # 编码结构和特征
            X, M = encode_structure(structure)
            q = encode_features(structure)[0]

            # 提取拓扑结构
            ids_topk, _, _, _, _ = extract_topology(X, 64)

            # 组织数据格式
            X, ids_topk, q, M = collate_batch_features([[X, ids_topk, q, M]])

            # 运行模型
            z = model(X.to(device), ids_topk.to(device), q.to(device), M.float().to(device))

            # 处理所有预测结果
            for i in range(z.shape[1]):
                # 应用 Sigmoid 归一化
                p = pt.sigmoid(z[:, i])

                # 记录预测值
                structure = encode_bfactor(structure, p.cpu().numpy())

                # 保存 PDB 结构
                output_filepath = filepath[:-4] + f'_i{i}.pdb'
                save_pdb(split_by_chain(structure), output_filepath)


def main():
    """主函数"""

    pdb_filepaths = glob(os.path.join(DATA_PATH, "*.pdb"), recursive=True)
    pdb_filepaths = [fp for fp in pdb_filepaths if "_i" not in fp]

    # 创建数据集
    dataset = StructuresDataset(pdb_filepaths, with_preprocessing=True)

    print(f"Found {len(dataset)} PDB files to process.")

    # 加载模型
    model = load_model()

    # 运行预测并保存
    predict_and_save_results(model, dataset)


if __name__ == "__main__":
    main()
