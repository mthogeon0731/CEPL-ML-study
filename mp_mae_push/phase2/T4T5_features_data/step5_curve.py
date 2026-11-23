"""step5_curve.py : H07 외부 데이터 추가량 곡선. 기준선 XGBoost 설정, mp train + 외부(특허 정제본) 무작위 k 배수.
배수 m ∈ {0.5, 1, 2, 4} 는 n_add = round(m × n_train_mp), 'full' = 정제본 전체. seed s 마다 표본도 seed s 로 새로 뽑는다.
대체 고품질 변형(보고 범위 폭 ≤ 2 K 기록만)과, 같은 수의 무작위 표본 대조를 함께 본다.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('T4_features/scripts'))
import fitlib as fl  # noqa: E402
mc = fl.mc
RAW = 'T5_external/preds/raw'
os.makedirs(RAW, exist_ok=True)
LEDGER = 'T5_external/ledger.csv'


def run(y, X0, S, XE, yE, name, pick, seeds=fl.SEEDS, note=''):
    """pick(seed, n_train) -> 외부 행 인덱스."""
    for sp in fl.SPLITS:
        tr = S[sp]['idx']['train']
        va = S[sp]['va']
        for seed in seeds:
            f = f'{RAW}/val_{name}_{sp}_s{seed}.npy'
            if os.path.exists(f):
                continue
            ei = np.sort(pick(seed, len(tr)))
            with mc.CostMeter('탐색') as c:
                Xtr = np.vstack([X0[tr], XE[ei]])
                ytr = np.concatenate([y[tr], yE[ei]])
                p = fl.fit_predict(Xtr, ytr, X0[va], seed)
                del Xtr
            np.save(f, p)
            yv = y[va]
            mae = float(np.mean(np.abs(yv - p)))
            mc.append_ledger(LEDGER, dict(run_id=f'T5_val_{name}_{sp}_s{seed}', track='T5 외부', model='XGBoost(기준선 설정)',
                                          features='desc217', data_variant=f'mp train + 특허 {name} ({len(ei)}분자)',
                                          objective='reg:squarederror', split=sp, seed=seed, params='BASE_XGB, n_jobs=8',
                                          val_MAE=round(mae, 4), val_RMSE=round(mc.metrics(yv, p)['RMSE'], 4),
                                          n_train=len(tr) + len(ei), notes=note, **c.as_row()))
            print(name, sp, seed, len(ei), round(mae, 3), round(c.wall_s), round(c.cpu_core_h, 3), flush=True)
