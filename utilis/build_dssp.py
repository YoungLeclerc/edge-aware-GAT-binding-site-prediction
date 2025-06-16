import torch as pt
import numpy as np
from Bio.PDB import PDBParser, DSSP
import numpy as np


def parse_dssp_and_rsa(pdb_filepath, dssp_exe='mkdssp'):
    """
    读取PDB结构文件，运行DSSP计算二级结构和RSA，并返回对应的结果映射到每个原子

    参数：
    - pdb_filepath: str, PDB文件路径
    - dssp_exe: str, DSSP可执行文件路径，默认mkdssp

    返回：
    - ss_list: list[str], 每个原子的二级结构码（H, E, etc.）
    - rsa_list: list[float], 每个原子的相对表面可及性(0~1)
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_filepath)
    model = structure[0]  # 只用第一个model

    # 运行DSSP
    dssp = DSSP(model, pdb_filepath, dssp=dssp_exe)

    # 准备结果
    ss_list = []
    rsa_list = []

    # 建立残基索引映射: (chain_id, res_id) -> DSSP tuple
    dssp_dict = {(chain_id, res_id): dssp[key] for key in dssp.keys()}

    # 遍历每个原子，将它所属残基的二级结构和RSA赋值给原子
    for chain in model:
        for residue in chain:
            chain_id = chain.id
            res_id = residue.id

            # 获取对应的DSSP信息，没找到用'-'和0代替
            dssp_key = (chain_id, res_id)
            if dssp_key in dssp_dict:
                ss = dssp_dict[dssp_key][2]  # 二级结构码
                rsa = dssp_dict[dssp_key][3]  # 相对表面可及性
            else:
                ss = '-'
                rsa = 0.0

            # 赋值给该残基所有原子
            for atom in residue:
                ss_list.append(ss)
                rsa_list.append(rsa)

    return ss_list, rsa_list


# 测试用法
pdb_file = "example.pdb"
ss_codes, rsa_vals = parse_dssp_and_rsa(pdb_file)

print(f"Length of SS: {len(ss_codes)}")
print(f"Example SS: {ss_codes[:10]}")
print(f"Example RSA: {rsa_vals[:10]}")




# DSSP 二级结构编码标准（示例）
std_ss = np.array(['H', 'B', 'E', 'G', 'I', 'T', 'S', '-'])  # 8种二级结构类型


# 你可以根据你的DSSP输出类型调整

def onehot(x, v):
    m = (x.reshape(-1, 1) == np.array(v).reshape(1, -1))
    return np.concatenate([m, ~np.any(m, axis=1).reshape(-1, 1)], axis=1)


def encode_features_dssp(structure, ss_codes, rsa_values, device=pt.device("cpu")):
    """
    生成包含DSSP二级结构和RSA特征的原子级特征

    Parameters:
    - structure: dict, 包含 'element', 'resname', 'name' 等字段的原子信息
    - ss_codes: array-like, 每个原子对应的DSSP二级结构字符数组（长度与原子数相同）
    - rsa_values: array-like, 每个原子的归一化相对表面可及性（0~1），长度与原子数相同

    Returns:
    - features: pt.Tensor, 形状 [N_atoms, 88], 包含原子元素、残基、原子名、DSSP one-hot和RSA数值
    """

    # 原子元素、残基名、原子名 one-hot
    qe = pt.from_numpy(onehot(structure['element'], std_elements).astype(np.float32)).to(device)  # shape: [N, 33]
    qr = pt.from_numpy(onehot(structure['resname'], std_resnames).astype(np.float32)).to(device)  # shape: [N, 30]
    qn = pt.from_numpy(onehot(structure['name'], std_names).astype(np.float32)).to(device)  # shape: [N, 16]

    # DSSP 二级结构 one-hot
    ss_onehot = pt.from_numpy(onehot(np.array(ss_codes), std_ss).astype(np.float32)).to(device)  # shape: [N, 8]

    # RSA 数值特征，保证是 [N, 1]
    rsa_tensor = pt.tensor(rsa_values, dtype=pt.float32).unsqueeze(1).to(device)  # shape: [N, 1]

    # 拼接所有特征
    features = pt.cat([qe, qr, qn, ss_onehot, rsa_tensor], dim=1)  # shape: [N, 33+30+16+8+1=88]

    return features
