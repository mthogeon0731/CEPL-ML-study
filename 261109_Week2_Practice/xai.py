"""xai.py : 4주차 설명 가능 AI (XAI) 도구 모음

기존 파일(feature.py, datamanager.py, models.py, train.py)은 고치지 않고 import 만 한다.

    from xai import *
    sets, scaler = prepare_graphs('logP', max_n=3000)          # 80:10:10, seed 42 (train.py 와 같은 분할)
    model, info = train_gnn(sets, scaler, pooling='sum')       # train.run() 과 같은 학습 설정, 모델 객체를 돌려준다
    a = attr_integrated_gradients(model, sets['test'][0])      # 원자 하나당 숫자 하나
    ref = crippen_atom_contribs(sets['test'][0].smiles)        # 정답 역할: Wildman-Crippen 원자 기여

원자 기여 방법 (GNN)
    attr_saliency       : |dy/dx| 를 특징 방향으로 더함 (부호 없음)
    attr_grad_input     : x * dy/dx 를 특징 방향으로 더함 (부호 있음)
    attr_integrated_gradients : 0 벡터에서 x 까지 직선 경로의 평균 기울기 * (x - 0)  (Sundararajan 2017)
    attr_gnnexplainer   : 예측을 유지하는 최소 원자 마스크 (Ying 2019, PyG Explainer)
    attr_readout        : sum readout + 선형 head 일 때 원자 벡터 하나가 예측에 더하는 몫 (모델 구조에서 바로 나옴)
    attr_random, attr_element_rule : 비교 기준선 (무작위, 'C·할로젠 +, N·O -' 규칙)
"""
import copy
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from scipy.stats import spearmanr
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from datamanager import load_data, gen_dataset, split_dataset, Scaler
from models import GraphRegressor

OKABE = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442', '#000000']


# ---------------------------------------------------------------- 1. 기술자 (Crippen 계열 제외)
def is_crippen_family(name):
    """MolLogP 자체와, Crippen 원자 logP / MR 값으로 만든 기술자. logP 를 예측할 때 넣으면 정답을 넣는 셈."""
    n = name.lower()
    return n in ('mollogp', 'molmr') or n.startswith(('slogp_vsa', 'smr_vsa')) or n.startswith(('bcut2d_logp', 'bcut2d_mr'))


def descriptor_matrix(smiles, drop_crippen=True):
    names = [n for n, _ in Descriptors.descList if not (drop_crippen and is_crippen_family(n))]
    funcs = dict(Descriptors.descList)
    X = np.array([[funcs[n](Chem.MolFromSmiles(s)) for n in names] for s in smiles], dtype=float)
    X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    return X, names


# ---------------------------------------------------------------- 2. Morgan 비트 -> 부분구조
_MFP = {}


def _gen(radius, n_bits):
    if (radius, n_bits) not in _MFP:
        _MFP[(radius, n_bits)] = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    return _MFP[(radius, n_bits)]


def morgan_matrix(smiles, radius=2, n_bits=2048):
    g = _gen(radius, n_bits)
    return np.array([g.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.uint8)


def morgan_bit_info(mol, radius=2, n_bits=2048):
    """{비트 번호: ((중심 원자, 반경), ...)}  Draw.DrawMorganBit 에 그대로 넣는다."""
    ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
    _gen(radius, n_bits).GetFingerprintAsNumPy(mol, additionalOutput=ao)
    return {k: tuple(v) for k, v in ao.GetBitInfoMap().items()}


def bit_fragments(smiles, bit, radius=2, n_bits=2048):
    """비트 하나를 켠 부분구조(SMILES)를 모두 모아 센다. 종류가 둘 이상이면 비트 충돌."""
    out = {}
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        for atom, r in morgan_bit_info(m, radius, n_bits).get(bit, ()):
            if r == 0:
                frag = m.GetAtomWithIdx(atom).GetSymbol() + ('(arom)' if m.GetAtomWithIdx(atom).GetIsAromatic() else '')
            else:
                env = Chem.FindAtomEnvironmentOfRadiusN(m, r, atom)
                amap = {}
                sub = Chem.PathToSubmol(m, env, atomMap=amap)
                frag = Chem.MolToSmiles(sub)
            out[frag] = out.get(frag, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ---------------------------------------------------------------- 3. 정답 역할: Crippen 원자 기여
def crippen_atom_contribs(smiles):
    """Wildman & Crippen (1999) 원자 logP 기여. 수소의 몫은 붙어 있는 무거운 원자에 더한다 -> 합 = MolLogP."""
    m = Chem.MolFromSmiles(smiles)
    h = Chem.AddHs(m)
    c = rdMolDescriptors._CalcCrippenContribs(h)
    a = np.zeros(m.GetNumAtoms())
    for at in h.GetAtoms():
        i = at.GetIdx()
        j = at.GetNeighbors()[0].GetIdx() if (at.GetAtomicNum() == 1 and i >= m.GetNumAtoms()) else i
        a[j] += c[i][0]
    return a


# ---------------------------------------------------------------- 4. 그래프 데이터와 GNN 학습 (train.run 과 같은 설정)
def prepare_graphs(prop='logP', max_n=3000, seed=42, split='random'):
    df = load_data(prop, max_n=max_n, seed=seed)
    sets = split_dataset(gen_dataset(df, prop), how=split, seed=seed)
    scaler = Scaler('standard').fit(sets['train'])
    for k in sets:
        for d in sets[k]:
            d.y_raw = d.y.clone(); d.y = scaler.transform(d.y_raw)
    return sets, scaler


def predict_gnn(model, data_list, scaler, batch_size=256):
    model.eval(); out = []
    with torch.no_grad():
        for b in DataLoader(data_list, batch_size=batch_size):
            out.append(model(b))
    return scaler.inverse_transform(torch.cat(out)).numpy().ravel()


def train_gnn(sets, scaler, pooling='sum', seed=42, epochs=150, patience_stop=20, lr=1e-3, dropout=0.3,
              batch_size=64, verbose=True):
    """train.run() 의 학습 루프를 줄인 것. 차이: 모델 객체를 돌려주고, 결과표(summary.csv)에 쓰지 않는다."""
    torch.manual_seed(seed); np.random.seed(seed)
    model = GraphRegressor(sets['train'][0].x.shape[1], sets['train'][0].edge_attr.shape[1], conv='GIN',
                           dropout=dropout, pooling=pooling)
    for p in model.parameters():
        p.register_hook(lambda g: g.clamp(-10, 10))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.8, min_lr=1e-6)
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(sets['train'], batch_size=batch_size, shuffle=True, generator=g)
    yv = torch.cat([d.y_raw for d in sets['validate']]).numpy().ravel()
    best, best_state, bad, t0 = np.inf, None, 0, time.time()
    for ep in range(epochs):
        model.train()
        for b in loader:
            opt.zero_grad(); loss = F.mse_loss(model(b), b.y); loss.backward(); opt.step()
        val = float(np.mean(np.abs(predict_gnn(model, sets['validate'], scaler) - yv)))
        sched.step(val)
        if val < best - 1e-4:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and ep % 20 == 0:
            print(f'  epoch {ep:3d}  val MAE {val:.3f}  ({time.time() - t0:.0f} s)')
        if bad >= patience_stop:
            break
    model.load_state_dict(best_state); model.eval()
    yt = torch.cat([d.y_raw for d in sets['test']]).numpy().ravel()
    pt = predict_gnn(model, sets['test'], scaler)
    info = dict(seed=seed, pooling=pooling, epochs=ep + 1, sec=round(time.time() - t0, 1),
                val_MAE=round(best, 4), test_MAE=round(float(np.mean(np.abs(pt - yt))), 4))
    if verbose:
        print(f'[train_gnn] seed {seed}, {pooling} pool: test MAE {info["test_MAE"]:.3f}, {info["epochs"]} epochs, {info["sec"]:.0f} s')
    return model, info


# ---------------------------------------------------------------- 5. 원자 기여 (GNN)
def _one(data, x=None):
    b = Batch.from_data_list([data])
    if x is not None:
        b.x = x
    return b


def _grad(model, data, x):
    x = x.clone().requires_grad_(True)
    y = model(_one(data, x)).sum()
    (g,) = torch.autograd.grad(y, x)
    return g


def attr_saliency(model, data):
    model.eval()
    return _grad(model, data, data.x).abs().sum(1).numpy()


def attr_grad_input(model, data):
    model.eval()
    return (_grad(model, data, data.x) * data.x).sum(1).numpy()


def attr_integrated_gradients(model, data, steps=50, baseline=None, return_check=False):
    """IG_v = sum_f (x - x') * mean_alpha dF/dx at x' + alpha (x - x').  완전성: sum_v IG_v = F(x) - F(x')."""
    model.eval()
    x = data.x
    x0 = torch.zeros_like(x) if baseline is None else baseline
    alphas = (torch.arange(steps, dtype=torch.float) + 0.5) / steps          # 중점 규칙
    g = sum(_grad(model, data, x0 + a * (x - x0)) for a in alphas) / steps
    ig = ((x - x0) * g).sum(1)
    if return_check:
        with torch.no_grad():
            gap = (model(_one(data, x)) - model(_one(data, x0))).item()
        return ig.detach().numpy(), dict(sum_IG=float(ig.sum()), F_x_minus_F_baseline=gap)
    return ig.detach().numpy()


class _PygWrap(nn.Module):
    """PyG Explainer 는 model(x, edge_index, **kw) 형태를 부른다. GraphRegressor 는 Data 를 받으므로 감싼다."""

    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x, edge_index, batch=None):
        from torch_geometric.data import Data
        d = Data(x=x, edge_index=edge_index)
        d.batch = torch.zeros(x.shape[0], dtype=torch.long) if batch is None else batch
        return self.m(d)


def attr_gnnexplainer(model, data, epochs=100, lr=0.01, seed=0):
    from torch_geometric.explain import Explainer, GNNExplainer
    torch.manual_seed(seed)
    model.eval()
    ex = Explainer(model=_PygWrap(model), algorithm=GNNExplainer(epochs=epochs, lr=lr), explanation_type='model',
                   node_mask_type='object', edge_mask_type=None,
                   model_config=dict(mode='regression', task_level='graph', return_type='raw'))
    e = ex(data.x, data.edge_index)
    return e.node_mask.detach().numpy().ravel()


def attr_readout(model, data):
    """sum pooling + 선형 head: y = sum_v (w . z_v) + b  ->  원자 v 의 몫 = w . z_v.  mean pooling 이면 1/N 배."""
    assert not model.bayes
    model.eval()
    with torch.no_grad():
        z = model.gnn(x=data.x, edge_index=data.edge_index)
        a = (z @ model.head.weight.T).ravel()
    if model.pooling == 'mean':
        a = a / data.x.shape[0]
    return a.numpy()


def attr_random(data, rng):
    return rng.standard_normal(data.x.shape[0])


def attr_element_rule(smiles):
    """화학 상식 수준 규칙: C, S, 할로젠 +1 / N, O -1 / 나머지 0. 모델 없이 얼마나 맞는지 보는 기준선."""
    m = Chem.MolFromSmiles(smiles)
    val = {6: 1, 16: 1, 9: 1, 17: 1, 35: 1, 53: 1, 7: -1, 8: -1}
    return np.array([val.get(a.GetAtomicNum(), 0) for a in m.GetAtoms()], dtype=float)


# ---------------------------------------------------------------- 6. 설명 검증
def rank_agreement(attr, ref):
    """분자 하나 안에서 원자 순위 일치도 (Spearman). 원자 수가 적거나 한쪽이 상수면 NaN."""
    if len(attr) < 4 or np.ptp(attr) == 0 or np.ptp(ref) == 0:
        return np.nan
    return float(spearmanr(attr, ref).correlation)


def evaluate_attributions(model, data_list, methods, n=200, seed=0, min_atoms=6):
    """methods: {이름: f(data) -> 원자 기여}. 분자마다 Spearman(기여, Crippen) 과 Spearman(|기여|, |Crippen|)."""
    rng = np.random.default_rng(seed)
    pool = [d for d in data_list if d.x.shape[0] >= min_atoms][:n]
    rows, store = [], {}
    for k, d in enumerate(pool):
        ref = crippen_atom_contribs(d.smiles)
        for name, f in methods.items():
            a = f(d) if name != 'random' else attr_random(d, rng)
            rows.append(dict(i=k, smiles=d.smiles, n_atoms=len(ref), method=name,
                             rho_signed=rank_agreement(a, ref), rho_abs=rank_agreement(np.abs(a), np.abs(ref))))
            store[(k, name)] = a
    return pd.DataFrame(rows), store


def summarize(df):
    g = df.groupby('method')
    out = pd.DataFrame({'signed_mean': g.rho_signed.mean(), 'signed_sd': g.rho_signed.std(),
                        'abs_mean': g.rho_abs.mean(), 'abs_sd': g.rho_abs.std(), 'n_mol': g.rho_signed.count()})
    return out.round(3)


# ---------------------------------------------------------------- 7. 그림
def atom_map_png(smiles, weights, size=(360, 300), legend=''):
    """RDKit SimilarityMaps 로 원자 기여 열지도 (PNG bytes). 주황 = +, 보라 = - (PuOr, 색각 이상 친화)."""
    from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
    import matplotlib
    m = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().legendFontSize = 18
    w = np.asarray(weights, dtype=float)
    w = w / (np.abs(w).max() + 1e-12)
    SimilarityMaps.GetSimilarityMapFromWeights(m, [float(v) for v in w], draw2d=d,
                                               colorMap=matplotlib.colormaps['PuOr_r'], contourLines=0)
    d.FinishDrawing()
    return d.GetDrawingText()


def png_to_array(png):
    import io
    from PIL import Image
    return np.array(Image.open(io.BytesIO(png)).convert('RGB'))
