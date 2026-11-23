"""train_xgb.py : XGBoost + RDKit 기술자 학습 한 번 = 결과표 한 줄  [추가 — mp 팀, 교수님 원본 아님]

    python train_xgb.py --prop mp                     # random 분할
    python train_xgb.py --prop mp --split scaffold
    python train_xgb.py --prop mp --max_n 0           # 전체 데이터
    python train_xgb.py --prop mp --feature morgan    # Morgan 지문 (radius 2, 2048 bit)
    python train_xgb.py --prop mp --feature morgan --fp_size 4096 --radius 3

train.py 와 같은 표본(load_data, seed)과 같은 80/10/10 분할(split_dataset 과 같은 로직)을 쓰므로
같은 옵션의 GIN 결과와 test 500개가 똑같다 → 바로 비교 가능.
XGBoost 설정은 1주차 노트북 get_models() 와 같다. 학습은 train 만, validate 는 점수 확인용.
결과: results/summary.csv 에 한 줄 추가 (train.py 와 같은 열), results/<이름>_test_pred.csv, _parity.png
"""
import argparse
import os
import random
import time

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PROPS = {'viscosity'}   # datamanager.py 와 같음


def load(prop, max_n, seed):
    """datamanager.load_data + gen_dataset 의 SMILES 거르기와 같은 결과."""
    df = pd.read_csv(os.path.join(HERE, 'data', f'{prop}.csv'))
    if max_n and len(df) > max_n:
        df = df.sample(max_n, random_state=seed).reset_index(drop=True)
    print(f'[load] {prop}: {len(df):,} rows, unit = {df["unit"].iloc[0]}')
    mols = [Chem.MolFromSmiles(s) for s in df['smiles']]
    keep = [i for i, m in enumerate(mols) if m is not None and m.GetNumAtoms() > 0]
    df = df.iloc[keep].reset_index(drop=True); mols = [mols[i] for i in keep]
    y = df['value'].values.astype(float)
    if prop in LOG_PROPS:
        y = np.log(y)
    return df, mols, y


def split(smiles, how, seed, portion=0.1):
    """datamanager.split_dataset 과 같은 로직 (인덱스만 돌려준다)."""
    rng = random.Random(seed); N = len(smiles); n = int(np.ceil(N * portion))
    if how == 'random':
        idx = list(range(N)); rng.shuffle(idx)
        return idx[2 * n:], idx[n:2 * n], idx[:n]
    groups = {}
    for i, s in enumerate(smiles):
        groups.setdefault(MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False), []).append(i)
    sets = list(groups.values()); rng.shuffle(sets); sets.sort(key=len)
    test, validate, train = [], [], []
    for g in sets:
        if len(test) + len(g) <= n: test += g
        elif len(validate) + len(g) <= n: validate += g
        else: train += g
    return train, validate, test


def featurize(mols, kind, fp_size=2048, radius=2):
    """1주차 노트북과 같은 표현."""
    if kind == 'desc':
        X = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=float)
        return np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)


def metrics(y, p):
    return dict(MAE=float(np.mean(np.abs(y - p))), RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
                R2=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)))


def run(prop='mp', split_how='random', feature='desc', max_n=5000, seed=42, fp_size=2048, radius=2):
    feat_name = 'Descriptor' if feature == 'desc' else ('MorganFP' if (fp_size, radius) == (2048, 2) else f'MorganFP-r{radius}-{fp_size}')
    name = f'{prop}_XGBoost-{feat_name}_{split_how}'
    df, mols, y = load(prop, max_n, seed)
    tr, va, te = split(list(df['smiles']), split_how, seed)
    print(f'[split:{split_how}] train {len(tr):,} / validate {len(va):,} / test {len(te):,}')

    t0 = time.time()
    X = featurize(mols, feature, fp_size, radius)
    print(f'[featurize] {feat_name}: {X.shape[1]} features ({time.time() - t0:.0f} s)')

    model = XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                         subsample=0.8, colsample_bytree=0.5, n_jobs=-1, random_state=seed)
    t0 = time.time(); model.fit(X[tr], y[tr]); sec = time.time() - t0
    val_mae = metrics(y[va], model.predict(X[va]))['MAE']
    p = model.predict(X[te])
    res = dict(prop=prop, model=f'XGBoost-{feat_name}', pooling='-', split=split_how, n=len(df),
               epochs=np.nan, sec=round(sec, 1), **metrics(y[te], p),
               baseline_MAE=float(np.mean(np.abs(y[te] - y[tr].mean()))), mean_sd=np.nan, coverage_2sd=np.nan)
    print(f'[validate] MAE {val_mae:.3f}')
    print(f'[test] {name}: MAE {res["MAE"]:.3f} | RMSE {res["RMSE"]:.3f} | R2 {res["R2"]:.3f} '
          f'| baseline MAE {res["baseline_MAE"]:.3f} | fit {sec:.0f} s')

    outdir = os.path.join(HERE, 'results'); os.makedirs(outdir, exist_ok=True)
    summary = os.path.join(outdir, 'summary.csv')
    pd.DataFrame([res]).to_csv(summary, mode='a', header=not os.path.exists(summary), index=False)
    pd.DataFrame({'smiles': df['smiles'].values[te], 'y': y[te], 'pred': p}).to_csv(
        os.path.join(outdir, f'{name}_test_pred.csv'), index=False)

    lo, hi = min(y[te].min(), p.min()), max(y[te].max(), p.max()); pad = 0.05 * (hi - lo); lim = (lo - pad, hi + pad)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    ax.scatter(y[te], p, s=8, alpha=0.5, color='#0072B2', label=f'XGBoost + {feat_name}\nMAE = {res["MAE"]:.3f}')
    ax.plot(lim, lim, 'k', lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f'measured {prop}'); ax.set_ylabel(f'predicted {prop}'); ax.legend(loc='upper left', frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, f'{name}_parity.png'), dpi=150); plt.close(fig)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prop', default='mp')
    ap.add_argument('--split', default='random', choices=['random', 'scaffold'])
    ap.add_argument('--feature', default='desc', choices=['desc', 'morgan'])
    ap.add_argument('--max_n', type=int, default=5000, help='0 = 전체')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--fp_size', type=int, default=2048, help='Morgan 비트 수')
    ap.add_argument('--radius', type=int, default=2, help='Morgan 반경 (결합 수)')
    a = ap.parse_args()
    run(a.prop, a.split, a.feature, a.max_n or None, a.seed, a.fp_size, a.radius)
