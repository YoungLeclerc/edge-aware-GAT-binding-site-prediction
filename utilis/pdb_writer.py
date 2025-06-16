import os


def save_structure_to_pdb(structure, output_path):
    """
    将结构信息以 PDB 格式写入文件，供 DSSP 使用。

    参数:
    - structure: dict，包含以下字段
        - 'name': (N,) 原子名称，如 ['CA', 'CB', ...]
        - 'resname': (N,) 残基名称，如 ['ALA', 'GLY', ...]
        - 'chain': (N,) 链 ID，如 ['A', 'A', 'B', ...]
        - 'resid': (N,) 残基编号（int）
        - 'xyz': (N, 3) 原子坐标
        - 'element': (N,) 元素类型，如 ['C', 'N', 'O', ...]
    - output_path: str，输出的 PDB 文件路径
    """

    with open(output_path, 'w') as f:
        for i in range(len(structure['name'])):
            atom_name = structure['name'][i]
            res_name = structure['resname'][i]
            chain_id = structure['chain'][i]
            res_id = structure['resid'][i]
            x, y, z = structure['xyz'][i]
            element = structure['element'][i]

            # PDB 规范原子行格式
            pdb_line = (
                "ATOM  {atom_id:5d} {atom_name:^4s} {res_name:>3s} {chain_id:1s}"
                "{res_id:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {element:>2s}\n"
            ).format(
                atom_id=i + 1,
                atom_name=atom_name,
                res_name=res_name,
                chain_id=chain_id,
                res_id=res_id,
                x=x,
                y=y,
                z=z,
                element=element.strip().upper()[:2]
            )

            f.write(pdb_line)

        f.write("END\n")
