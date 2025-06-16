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
SAVE_PATH = "model/save"
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


def find_highest_binding_sites(structure, bfactor, threshold=0.5):
    """ 选择结合概率大于 `threshold` 的原子，并聚合到残基级别 """
    print(f" bfactor min: {np.min(bfactor):.3f}, max: {np.max(bfactor):.3f}, mean: {np.mean(bfactor):.3f}")

    print(f" 结合概率 > {threshold:.2f} 的原子数量: {np.sum(bfactor > threshold)}")

    # 过滤出高于 `threshold` 的原子
    binding_indices = np.where(bfactor > threshold)[0]

    if len(binding_indices) == 0:
        print(" No binding sites found above the threshold.")
        return pd.DataFrame(columns=["chain", "residue_id", "residue_name", "mean_probability"])

    # 获取结合位点信息
    chain_ids = structure['chain_name'][binding_indices]
    res_ids = structure['resid'][binding_indices]
    res_names = structure['resname'][binding_indices]
    binding_probs = bfactor[binding_indices]

    df = pd.DataFrame({
        "chain": chain_ids,
        "residue_id": res_ids,
        "residue_name": res_names,
        "binding_probability": binding_probs
    })

    # 按残基级别计算平均结合概率
    grouped_df = df.groupby(["chain", "residue_id", "residue_name"]).agg(
        mean_probability=("binding_probability", "mean")
    ).reset_index()
    return grouped_df.sort_values(by="mean_probability", ascending=False)




def predict_and_save_results(model, dataset, threshold=0.5):
    """运行模型，保存带预测结果的 PDB 结构，并记录结合位点（残基级别）"""
    device = pt.device("cuda" if pt.cuda.is_available() else "cpu")

    with pt.no_grad():
        for subunits, filepath in tqdm(dataset, desc="Processing PDB files"):
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

            for i in range(z.shape[1]):  # 处理 5 个类别
                p = pt.sigmoid(z[:, i])  # 结合概率
                structure = encode_bfactor(structure, p.cpu().numpy())

                # 保存 PDB 结构
                output_filepath = filepath[:-4] + f'_i{i}.pdb'
                save_pdb(split_by_chain(structure), output_filepath)

                # 查找结合位点（残基级别）
                binding_df = find_highest_binding_sites(structure, p.cpu().numpy(), threshold)


                if not binding_df.empty:
                    output_csv_filepath = filepath[:-4] + f"_binding_sites_i{i}.csv"
                    binding_df.to_csv(output_csv_filepath, index=False)
                    print(f"✅ Saved binding site information to {output_csv_filepath}")
                else:
                    print(f"⚠️ No binding sites found for {output_filepath}")



def main():
    """主函数"""

    pdb_filepaths = glob(os.path.join(DATA_PATH, "*.pdb"), recursive=True)
    pdb_filepaths = [fp for fp in pdb_filepaths if "_i" not in fp]

    # 创建数据集
    dataset = StructuresDataset(pdb_filepaths, with_preprocessing=True)

    print(f"Found {len(dataset)} PDB files to process.")

    # 加载模型
    model = load_model()

    num_classes = 5

    # 运行预测并保存
    predict_and_save_results(model, dataset, threshold=0.5)  # 指定 threshold=0.5


if __name__ == "__main__":
    main()
