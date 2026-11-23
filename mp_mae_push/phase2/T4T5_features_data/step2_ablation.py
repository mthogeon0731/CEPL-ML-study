"""step2_ablation.py : F0 vs F0+F3, F3에서 열 묶음 하나씩 빼기 (validation, seed 3, 두 분할) + 선택 구성의 train 5-fold OOF.
python 커널에서 exec 로 실행(작업 폴더 = workspace 루트). 적합 하나마다 예측 npy 와 ledger 한 줄을 남겨 이어하기 가능.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('T4_features/scripts'))
import fitlib as fl  # noqa: E402
import phys_feats as pf  # noqa: E402
mc = fl.mc

RAW = 'T4_features/preds/raw'
os.makedirs(RAW, exist_ok=True)
LEDGER = 'T4_features/ledger.csv'

y, X0, S = fl.load_base()
XP = np.load('mp_mae_push/cache/X_phys.npy').astype(np.float32)
cols = list(pd.read_csv('mp_mae_push/cache/X_phys_cols.csv')['col'])
assert cols == pf.COLS


def mat(drop=()):
    keep = [i for i, c in enumerate(cols) if c not in set(drop)]
    return np.hstack([X0, XP[:, keep]]), len(keep)


CONFIGS = {'F0': None, 'F0+F3': ()}
for g, cs in pf.GROUPS.items():
    CONFIGS[f'F0+F3-{g}'] = tuple(cs)


def run_val(name, drop):
    X = X0 if drop is None else mat(drop)[0]
    nf = X.shape[1]
    for sp in fl.SPLITS:
        idx = S[sp]['idx']
        for seed in fl.SEEDS:
            f = f'{RAW}/val_{name}_{sp}_s{seed}.npy'
            if os.path.exists(f):
                continue
            with mc.CostMeter('특징') as c:
                p = fl.fit_predict(X[idx['train']], y[idx['train']], X[S[sp]['va']], seed)
            np.save(f, p)
            mae = float(np.mean(np.abs(y[S[sp]['va']] - p)))
            mc.append_ledger(LEDGER, dict(run_id=f'T4_val_{name}_{sp}_s{seed}', track='T4 특징', model='XGBoost(기준선 설정)',
                                          features=f'{name} ({nf}열)', data_variant='mp train', objective='reg:squarederror',
                                          split=sp, seed=seed, params='BASE_XGB, n_jobs=8', val_MAE=round(mae, 4),
                                          val_RMSE=round(mc.metrics(y[S[sp]['va']], p)['RMSE'], 4), n_train=len(idx['train']),
                                          notes='validation 예측만(테스트 예측 없음)', **c.as_row()))
            print(name, sp, seed, round(mae, 3), round(c.wall_s, 1), flush=True)


def run_oof(name, drop):
    X = X0 if drop is None else mat(drop)[0]
    for sp in fl.SPLITS:
        oof = S[sp]['oof']
        for seed in fl.SEEDS:
            f = f'{RAW}/oof_{name}_{sp}_s{seed}.npy'
            if os.path.exists(f):
                continue
            pred = np.full(len(oof), np.nan)
            with mc.CostMeter('특징') as c:
                for k in sorted(oof['fold'].unique()):
                    tr = oof['fold'].to_numpy() != k
                    pred[~tr] = fl.fit_predict(X[oof['row'].to_numpy()[tr]], y[oof['row'].to_numpy()[tr]],
                                               X[oof['row'].to_numpy()[~tr]], seed)
            np.save(f, pred)
            mae = float(np.mean(np.abs(oof['y'].to_numpy() - pred)))
            mc.append_ledger(LEDGER, dict(run_id=f'T4_oof_{name}_{sp}_s{seed}', track='T4 특징', model='XGBoost(기준선 설정)',
                                          features=name, data_variant='mp train 5-fold OOF (1단계 fold)', objective='reg:squarederror',
                                          split=sp, seed=seed, params='BASE_XGB, n_jobs=8, 5 fits',
                                          n_train=int(len(oof) * 0.8), notes=f'train OOF MAE {mae:.4f} (val_MAE 열 비움)', **c.as_row()))
            print('OOF', name, sp, seed, round(mae, 3), round(c.wall_s, 1), flush=True)
