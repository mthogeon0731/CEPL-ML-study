"""models.py : 그래프 -> 분자 벡터(fingerprint) -> 물성

연구실 BGNN_unary (9.ml_gnn/code/model/architecture.py) 와 같은 구조다.

    원자 특징 x (147)
      -> GIN 5층 (hidden 128, dropout 0.3)      : 이웃 정보를 5칸까지 모은다 (클래스 기본 0.5, 실제 학습 0.3)
      -> 원자별 벡터 (fingerprint 64)
      -> global_mean_pool                       : 원자 벡터 평균 = 분자 벡터
      -> Linear(64 -> 1)                        : 물성 하나

bayes=True 이면 마지막 Linear 만 확률 가중치(평균 mu, 표준편차 softplus(rho))로 바꾼다 (BGNN, Bayes by Backprop).
같은 분자를 n_rand 번 예측해 평균 = 예측값, 표준편차 = 불확실성.

주의: PyG 의 GIN 은 결합 특징(edge_attr)을 쓰지 않는다 (GIN.supports_edge_attr == False).
연구실 코드도 edge_attr 를 넘기지만 GIN 안에서 버려진다. 결합 특징을 쓰려면 conv='GINE'.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.nn.models import GIN, GCN, GAT
from torch_geometric.nn import GINEConv


class GINE(nn.Module):
    """결합 특징을 쓰는 GIN (Hu et al. 2020). 실습 비교용으로 직접 쌓는다."""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, edge_dim):
        super().__init__()
        self.convs = nn.ModuleList()
        for k in range(num_layers):
            d_in = in_channels if k == 0 else hidden_channels
            d_out = out_channels if k == num_layers - 1 else hidden_channels
            mlp = nn.Sequential(nn.Linear(d_in, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, d_out))
            self.convs.append(GINEConv(mlp, edge_dim=edge_dim))
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        for k, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr)
            if k < len(self.convs) - 1:
                x = F.dropout(F.relu(x), self.dropout, self.training)
        return x


CONV = {'GIN': GIN, 'GCN': GCN, 'GAT': GAT}


class GraphRegressor(nn.Module):
    def __init__(self, num_node_features, num_edge_features=16, conv='GIN', num_layers=5,
                 hidden_channels=128, fingerprint_channels=64, dropout=0.5,
                 pooling='mean', bayes=False, n_rand=100):
        super().__init__()
        if conv == 'GINE':
            self.gnn = GINE(num_node_features, hidden_channels, fingerprint_channels,
                            num_layers, dropout, num_edge_features)
        else:
            self.gnn = CONV[conv](in_channels=num_node_features, hidden_channels=hidden_channels,
                                  out_channels=fingerprint_channels, num_layers=num_layers, dropout=dropout)
        self.conv, self.pooling, self.bayes, self.n_rand = conv, pooling, bayes, n_rand
        if bayes:
            self.W_mu = nn.Parameter(0.1 * torch.randn(fingerprint_channels, 1))
            self.W_rho = nn.Parameter(-3 * torch.ones(fingerprint_channels, 1))   # softplus(-3) = 0.05
            self.b_mu = nn.Parameter(torch.zeros(1))
            self.b_rho = nn.Parameter(-3 * torch.ones(1))
        else:
            self.head = nn.Linear(fingerprint_channels, 1)

    def fingerprint(self, data):
        """분자 하나당 벡터 하나 (batch x fingerprint_channels)."""
        if self.conv == 'GINE':
            z = self.gnn(data.x, data.edge_index, data.edge_attr)
        else:
            z = self.gnn(x=data.x, edge_index=data.edge_index)
        pool = global_mean_pool if self.pooling == 'mean' else global_add_pool
        return pool(z, data.batch)

    def forward(self, data):
        z = self.fingerprint(data)
        if not self.bayes:
            return self.head(z)                                  # (batch, 1)
        eps_W = torch.randn(self.n_rand, *self.W_mu.shape, device=z.device)
        eps_b = torch.randn(self.n_rand, 1, device=z.device)
        W = self.W_mu + F.softplus(self.W_rho) * eps_W           # (n_rand, F, 1)
        b = self.b_mu + F.softplus(self.b_rho) * eps_b           # (n_rand, 1)
        y = torch.einsum('bf,rfo->bro', z, W).squeeze(-1) + b.T  # (batch, n_rand)
        return y

    def kl(self):
        """사후분포 N(mu, sigma) 와 사전분포 N(0, 1) 사이 KL. Bayes 손실의 정규화 항."""
        kl = 0.
        for mu, rho in ((self.W_mu, self.W_rho), (self.b_mu, self.b_rho)):
            s = F.softplus(rho)
            kl = kl + (0.5 * (s ** 2 + mu ** 2 - 1) - torch.log(s)).sum()
        return kl

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- T2 GIN 슬롯 (mp_mae_push, 2026-11-21 추가)
class HybridGraphRegressor(nn.Module):
    """수업 GIN(mean readout) + RDKit 기술자 하이브리드 readout.

        원자 특징 x (147) -> GIN 5층 (hidden 128, dropout 0.3) -> 원자 벡터 64 -> global_mean_pool (z, 64)
        [z (64) | 기술자 217 (train 표준화, ±10 클리핑)] -> MLP head 281 -> 256 -> 128 -> 1

    n_desc=0 이면 그래프 전용 대조(같은 MLP head, 입력 64)가 된다.
    GIN 은 수업 그대로 PyG GIN (edge_attr 미사용).
    """

    def __init__(self, num_node_features, n_desc=217, num_layers=5, hidden_channels=128,
                 fingerprint_channels=64, dropout=0.3, head_hidden=(256, 128), head_dropout=0.3, pooling='mean'):
        super().__init__()
        self.gnn = GIN(in_channels=num_node_features, hidden_channels=hidden_channels,
                       out_channels=fingerprint_channels, num_layers=num_layers, dropout=dropout)
        self.pooling, self.n_desc = pooling, n_desc
        layers, d = [], fingerprint_channels + n_desc
        for h in head_hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(head_dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.head = nn.Sequential(*layers)

    def fingerprint(self, data):
        z = self.gnn(x=data.x, edge_index=data.edge_index)
        pool = global_mean_pool if self.pooling == 'mean' else global_add_pool
        return pool(z, data.batch)

    def forward(self, data):
        z = self.fingerprint(data)
        if self.n_desc:
            z = torch.cat([z, data.desc], dim=1)
        return self.head(z)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
