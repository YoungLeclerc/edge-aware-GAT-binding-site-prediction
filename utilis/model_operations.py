import torch as pt
from torch_geometric.nn import GATConv


def unpack_state_features(X, ids_topk, q, sigma=3.0):
    R_nn = X[ids_topk - 1] - X.unsqueeze(1)
    D_nn = pt.norm(R_nn, dim=2)
    D_nn = D_nn + pt.max(D_nn) * (D_nn < 1e-2).float()
    R_nn = R_nn / D_nn.unsqueeze(2)

    edge_index_list = []
    edge_attr_list = []
    for i in range(ids_topk.shape[0]):
        for j in range(ids_topk.shape[1]):
            neighbor = ids_topk[i, j].item() - 1
            if neighbor >= 0:
                edge_index_list.append([neighbor, i])  # neighbor → i
                gaussian_weight = pt.exp(-D_nn[i, j].pow(2) / (2 * sigma ** 2)).unsqueeze(0)
                direction = R_nn[i, j]
                edge_attr_list.append(pt.cat([gaussian_weight, direction]))

    edge_index = pt.tensor(edge_index_list, dtype=pt.long).T.contiguous().to(q.device)
    edge_attr = pt.stack(edge_attr_list).to(q.device)
    return q, edge_index, edge_attr



class GATUpdateLayer(pt.nn.Module):
    def __init__(self, in_dim, out_dim, edge_dim=4, heads=4, concat=True):
        super().__init__()
        self.gat = GATConv(
            in_channels=in_dim,
            out_channels=out_dim // heads if concat else out_dim,
            heads=heads,
            concat=concat,
            edge_dim=edge_dim,
            add_self_loops=False
        )

    def forward(self, q, edge_index, edge_attr):
        return self.gat(q, edge_index, edge_attr)


class StateUpdateLayer(pt.nn.Module):
    def __init__(self, layer_params):
        super(StateUpdateLayer, self).__init__()
        self.m_nn = pt.nn.Parameter(pt.arange(layer_params['nn'], dtype=pt.int64), requires_grad=False)
        self.gat = GATUpdateLayer(
            in_dim=layer_params['Ns'],
            out_dim=layer_params['Ns'],
            edge_dim=4,
            heads=layer_params['Nh'],
            concat=True
        )

    def forward(self, Z):
        q, p, ids_topk, edge_index, edge_attr = Z
        q_updated = self.gat(q[1:], edge_index, edge_attr)
        q = pt.cat([q[:1], q_updated], dim=0)
        return q, p, ids_topk, edge_index, edge_attr


def state_max_pool(q, p, M):
    s = pt.norm(p, dim=2)
    q_max, _ = pt.max(M.unsqueeze(2) * q.unsqueeze(1), dim=0)
    _, s_ids = pt.max(M.unsqueeze(2) * s.unsqueeze(1), dim=0)
    p_max = pt.gather(p, 0, s_ids.unsqueeze(2).repeat((1, 1, p.shape[2])))
    return q_max, p_max


class StatePoolLayer(pt.nn.Module):
    def __init__(self, N0, N1, Nh):
        super(StatePoolLayer, self).__init__()
        self.sam = pt.nn.Sequential(
            pt.nn.Linear(2*N0, N0), pt.nn.ELU(),
            pt.nn.Linear(N0, N0), pt.nn.ELU(),
            pt.nn.Linear(N0, 2*Nh)
        )
        self.zdm = pt.nn.Sequential(
            pt.nn.Linear(Nh * N0, N0), pt.nn.ELU(),
            pt.nn.Linear(N0, N0), pt.nn.ELU(),
            pt.nn.Linear(N0, N1)
        )
        self.zdm_vec = pt.nn.Sequential(
            pt.nn.Linear(Nh * N0, N1, bias=False)
        )

    def forward(self, q, p, M):
        F = (1.0 - M + 1e-6) / (M - 1e-6)
        z = pt.cat([q, pt.norm(p, dim=1)], dim=1)
        Ms = pt.nn.functional.softmax(self.sam(z).unsqueeze(1) + F.unsqueeze(2), dim=0).view(M.shape[0], M.shape[1], -1, 2)
        qh = pt.matmul(pt.transpose(q, 0, 1), pt.transpose(Ms[:, :, :, 0], 0, 1))
        ph = pt.matmul(pt.transpose(pt.transpose(p, 0, 2), 0, 1), pt.transpose(Ms[:, :, :, 1], 0, 1).unsqueeze(1))
        qr = self.zdm(qh.view(Ms.shape[1], -1))
        pr = self.zdm_vec(ph.view(Ms.shape[1], p.shape[1], -1))
        return qr, pr


class CrossStateUpdateLayer(pt.nn.Module):
    def __init__(self, layer_params):
        super(CrossStateUpdateLayer, self).__init__()
        Ns = layer_params['Ns']
        self.Nh = layer_params['cNh']
        self.Nk = layer_params['cNk']
        self.sul = StateUpdateLayer(layer_params)
        self.cqm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns), pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns), pt.nn.ELU(),
            pt.nn.Linear(Ns, self.Nk*self.Nh)
        )
        self.ckm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns), pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns), pt.nn.ELU(),
            pt.nn.Linear(Ns, self.Nk)
        )
        self.cvm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns), pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns), pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns)
        )
        self.cpm = pt.nn.Sequential(
            pt.nn.Linear((self.Nh + 1)*Ns, Ns), pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns), pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns)
        )
        self.sdk = pt.nn.Parameter(pt.sqrt(pt.tensor(self.Nk).float()), requires_grad=False)

    def forward(self, Z):
        q0, p0, ids0_topk, D0_nn, R0_nn = Z[0]
        q1, p1, ids1_topk, D1_nn, R1_nn = Z[1]
        qa0, pz0, *_ = self.sul((q0, p0, ids0_topk, D0_nn, R0_nn))
        qa1, pz1, *_ = self.sul((q1, p1, ids1_topk, D1_nn, R1_nn))
        s0 = pt.cat([qa0, pt.norm(pz0, dim=1)], dim=1)
        s1 = pt.cat([qa1, pt.norm(pz1, dim=1)], dim=1)
        Q0 = self.cqm(s0).reshape(s0.shape[0], self.Nh, self.Nk)
        Q1 = self.cqm(s1).reshape(s1.shape[0], self.Nh, self.Nk)
        K0 = self.ckm(s0).transpose(0, 1)
        K1 = self.ckm(s1).transpose(0, 1)
        V0 = self.cvm(s0)
        V1 = self.cvm(s1)
        M10 = pt.nn.functional.softmax(pt.matmul(Q0, K1 / self.sdk), dim=2)
        qh0 = pt.matmul(M10, V1).view(Q0.shape[0], -1)
        M01 = pt.nn.functional.softmax(pt.matmul(Q1, K0 / self.sdk), dim=2)
        qh1 = pt.matmul(M01, V0).view(Q1.shape[0], -1)
        qz0 = self.cpm(pt.cat([qa0, qh0], dim=1))
        qz1 = self.cpm(pt.cat([qa1, qh1], dim=1))
        return (qz0, pz0, ids0_topk, D0_nn, R0_nn), (qz1, pz1, ids1_topk, D1_nn, R1_nn)
