import torch as pt
from utilis.model_operations import StateUpdateLayer, StatePoolLayer, unpack_state_features

class ModularGATEncoder(pt.nn.Module):
    def __init__(self, layer_configs):
        super().__init__()
        self.layers = pt.nn.ModuleList([StateUpdateLayer(cfg) for cfg in layer_configs])

    def forward(self, q, p, ids_topk, edge_index, edge_attr):
        for layer in self.layers:
            q, p, _, _, _ = layer((q, p, ids_topk, edge_index, edge_attr))
        return q, p


class Model(pt.nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        # Feature Embedding: 输入维度改为 88（加入 DSSP 特征）
        self.em = pt.nn.Sequential(
            pt.nn.Linear(88, config['em']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['em']['N1'], config['em']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['em']['N1'], config['em']['N1']),
        )

        # GATConv 编码器
        self.encoder = ModularGATEncoder(config['sum'])

        # 残基池化（State Attention + Z direction modeling）
        self.pool = StatePoolLayer(config['spl']['N0'], config['spl']['N1'], config['spl']['Nh'])

        #  解码器（MLP + 输出维度为5类小分子结合概率）
        self.decoder = pt.nn.Sequential(
            pt.nn.Linear(config['dm']['N0'], config['dm']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['dm']['N1'], config['dm']['N1']),
            pt.nn.ELU(),
            pt.nn.Linear(config['dm']['N1'], config['dm']['N2']),
        )

    def forward(self, X, ids_topk, q0, M):
        # 特征嵌入（输入为 88 维）
        q = self.em(q0)

        # 初始化矢量状态张量（带 sink node）
        p = pt.zeros((q.shape[0] + 1, X.shape[1], q.shape[1]), device=X.device)

        # 图结构解包：构建 edge_index 与 edge_attr（距离 + 方向）
        q, edge_index, edge_attr = unpack_state_features(X, ids_topk, q)

        # 多层 GATConv 编码器
        q, p = self.encoder(q, p, ids_topk, edge_index, edge_attr)

        # 残基池化
        qr, pr = self.pool(q[1:], p[1:], M)

        # 拼接残基状态和方向模长作为 decoder 输入
        z_input = pt.cat([qr, pt.norm(pr, dim=1)], dim=1)  # shape: [N_res, 33]
        z = self.decoder(z_input)  # 输出为 [N_res, 5]

        return z
