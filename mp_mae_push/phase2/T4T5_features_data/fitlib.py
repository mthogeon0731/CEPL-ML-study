"""fitlib.py : 특징·데이터 트랙 공통 적합·판정 도구 (기준선 XGBoost 설정 고정, n_jobs=8).

- 기준 = 기준선 n_jobs=8 validation 예측 파일(phase1/preds/baseline_xgb_desc217_<split>_val.csv; seed 42 단일 적합).
- 후보는 seed 42, 43, 44 로 적합해 seed 평균 예측을 만든다. 같은 seed 의 F0(기술자 217) 적합과도 짝지어 비교한다
  (seed 평균 자체가 3개 배깅 효과를 내므로, 특징 효과는 'F0 seed 3 평균' 대비가 공정한 비교).
- test 행은 예측하지 않는다.
"""
import os
import sys

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.dont_write_bytecode = True
ROOTW = os.getcwd()
sys.path.insert(0, os.path.join(ROOTW, 'mp_mae_push', 'common'))
import mpcommon as mc  # noqa: E402

SEEDS = (42, 43, 44)
SPLITS = ('random', 'scaffold')


def load_base():
    y = pd.read_csv(os.path.join(mc.WEEK2, 'data', 'mp.csv'))['value'].to_numpy(float)
    X0 = np.load(os.path.join(ROOTW, 'mp_mae_push', 'cache', 'X_desc217.npy')).astype(np.float32)
    S = {}
    for sp in SPLITS:
        idx, s = mc.load_split(sp)
        dd = pd.read_csv(os.path.join(ROOTW, 'mp_mae_push', 'phase2', 'prereg', f'dedup_rows_{sp}.csv'))
        keep = set(dd.loc[(dd.set == 'validate') & (dd.overlap == 0), 'row'])
        va = idx['validate']
        base = pd.read_csv(os.path.join(ROOTW, 'mp_mae_push', 'phase1', 'preds', f'baseline_xgb_desc217_{sp}_val.csv'))
        bmap = dict(zip(base['row'], base['pred']))
        oof = pd.read_csv(os.path.join(ROOTW, 'mp_mae_push', 'phase1', 'preds', f'oof_baseline_{sp}_train.csv'))
        S[sp] = dict(idx=idx, va=va, dedup_mask=np.array([r in keep for r in va]),
                     base_val=np.array([bmap[r] for r in va]), oof=oof)
    return y, X0, S


def xgb(seed, **kw):
    return XGBRegressor(**{**mc.BASE_XGB, **kw}, n_jobs=8, random_state=seed)


def fit_predict(Xtr, ytr, Xev, seed, w=None):
    m = xgb(seed)
    m.fit(Xtr, ytr, sample_weight=w)
    return m.predict(Xev).astype(float)


def judge(y_va, p_new, p_ref, mask=None, n_boot=2000):
    if mask is not None:
        y_va, p_new, p_ref = y_va[mask], p_new[mask], p_ref[mask]
    d, lo, hi = mc.paired_bootstrap(y_va, p_new, p_ref, n_boot=n_boot)
    return d, lo, hi


def summarize(name, sp, y, S, preds_by_seed, f0_avg=None):
    """preds_by_seed: (n_seed, n_val). 반환: 행 dict (전체·중복 제거 두 기준)."""
    va = S[sp]['va']
    yv = y[va]
    m = S[sp]['dedup_mask']
    P = np.asarray(preds_by_seed)
    pavg = P.mean(axis=0)
    seed_mae = [float(np.mean(np.abs(yv - p))) for p in P]
    row = dict(candidate=name, split=sp, n_val=len(va), n_val_dedup=int(m.sum()), n_seed=len(P),
               MAE_seedavg=float(np.mean(np.abs(yv - pavg))), MAE_seed_mean=float(np.mean(seed_mae)),
               MAE_seed_sd=float(np.std(seed_mae, ddof=1)) if len(P) > 1 else np.nan,
               RMSE_seedavg=mc.metrics(yv, pavg)['RMSE'],
               MAE_dedup_seedavg=float(np.mean(np.abs(yv[m] - pavg[m]))))
    for tag, ref in [('vsBase', S[sp]['base_val'])] + ([('vsF0', f0_avg)] if f0_avg is not None else []):
        d, lo, hi = judge(yv, pavg, ref)
        row.update({f'd_{tag}': d, f'lo_{tag}': lo, f'hi_{tag}': hi})
        d, lo, hi = judge(yv, pavg, ref, m)
        row.update({f'd_{tag}_dedup': d, f'lo_{tag}_dedup': lo, f'hi_{tag}_dedup': hi})
    return row, pavg


def save_val_pred(path, va, y, pavg):
    pd.DataFrame({'row': va, 'y': y[va], 'pred': pavg}).to_csv(path, index=False)
