"""[추가] 과제 '설명이 틀린 분자 2개' 그림. 04 노트북 [추가] 절에서 저장한 GIN(sum) seed 42 · 7 가중치를 불러 다시 그린다."""
import numpy as np, torch, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'DejaVu Sans']
from torch_geometric.data import Data
from feature import smiles_to_graph
from models import GraphRegressor
from xai import attr_readout, attr_integrated_gradients, atom_map_png, png_to_array, rank_agreement, predict_gnn
from datamanager import load_data, gen_dataset, split_dataset, Scaler
import mp_xai_ext as mx

sets = split_dataset(gen_dataset(load_data('mp', max_n=5000, seed=42), 'mp'), how='random', seed=42)
scaler = Scaler('standard').fit(sets['train'])
M = {}
for s in (42, 7):
    m = GraphRegressor(147, 16, conv='GIN', dropout=0.3, pooling='sum'); m.load_state_dict(torch.load(f'results_xai_mp_ext/gin_sum_seed{s}.pt')); m.eval(); M[s] = m
cases = [('NCCCN/C=C/NCCCN', 271.15, '① 엔다이아민: GIN·Joback 둘 다 +172 K 과대'),
         ('C=CCN(C)C', 273.15, '② N,N-다이메틸알릴아민: 같은 모델, 두 설명이 반대'),
         ('FC(F)C(F)(F)C(F)Cl', 358.15, '참고: 값 오류 의심 (설명 이전의 문제)')]
cols = ['Joback ΔTm', 'readout s42', 'IG s42', 'readout s7', 'IG s7']
fig, axes = plt.subplots(len(cases), len(cols), figsize=(2.7 * len(cols), 2.75 * len(cases)))
for r, (smi, y, title) in enumerate(cases):
    x, ei, ea = smiles_to_graph(smi); d = Data(x=x, edge_index=ei, edge_attr=ea, smiles=smi, y=torch.tensor([[y]]))
    pred = float(predict_gnn(M[42], [d], scaler)[0])
    ref = mx.joback_atom_contribs(smi)[0]
    W = {'Joback ΔTm': ref, 'readout s42': attr_readout(M[42], d), 'IG s42': attr_integrated_gradients(M[42], d),
         'readout s7': attr_readout(M[7], d), 'IG s7': attr_integrated_gradients(M[7], d)}
    for c, k in enumerate(cols):
        ax = axes[r, c]; ax.imshow(png_to_array(atom_map_png(smi, W[k]))); ax.set_xticks([]); ax.set_yticks([])
        sub = '' if c == 0 else f'\nρ vs Joback {rank_agreement(W[k], ref):+.2f}'
        ax.set_title(k + sub, fontsize=9)
        if c > 0:
            print(smi, k, 'sum', round(float(np.sum(W[k])), 3), 'min', round(float(np.min(W[k])), 3), 'max', round(float(np.max(W[k])), 3))
    axes[r, 0].set_ylabel(f'{smi}\ny {y:.0f} K / GIN s42 {pred:.0f} K\nJoback {mx.joback_tm(smi):.0f} K', fontsize=8)

fig.text(0.01, 0.995, '주황 = 예측 mp 를 올리는 쪽, 보라 = 내리는 쪽 (분자마다 최댓값으로 정규화)', fontsize=9, va='top')
plt.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig('nbfigs/X6_wrong_explanation_2mol.png', dpi=150)
