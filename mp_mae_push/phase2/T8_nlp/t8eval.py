"""t8eval.py : T8(자연어·LLM) 트랙 공통 평가 — 같은 head 에서 표현만 바꿔 validation 으로 비교.

head (a) Ridge: train 안 5-fold CV 로 alpha 선택(목표 MAE). 전처리(train 으로만 적합): inf→NaN, 열 중앙값 대치,
         train 0.5/99.5 백분위 자르기, 표준화, 상수 열 제거.
head (b) XGBoost: mc.BASE_XGB, n_jobs=8, seed (42, 0, 1) — 42 는 1단계 기준선과 같은 seed.
test 행은 학습·예측 어디에도 쓰지 않는다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('mp_mae_push/common'))
import mpcommon as mc  # noqa: E402

TRACK = 'T8_nlp'
LEDGER = os.path.join(TRACK, 'ledger.csv')
PRED_DIR = os.path.join(TRACK, 'preds')
XGB_SEEDS = (42, 0, 1)
SUB_SEEDS = (0, 1, 2)
N_JOBS = 8

_Y = None
_SPL = {}
_DEDUP = {}
_BASE = {}


def y_all():
    global _Y
    if _Y is None:
        _Y = pd.read_csv(os.path.join(mc.WEEK2, 'data', 'mp.csv'))['value'].to_numpy(float)
    return _Y


def split(how):
    if how not in _SPL:
        _SPL[how] = mc.load_split(how)[0]
        d = pd.read_csv(f'mp_mae_push/phase2/prereg/dedup_rows_{how}.csv')
        _DEDUP[how] = set(d.loc[(d['set'] == 'validate') & (d['overlap'] == 0), 'row'])
        b = pd.read_csv(f'mp_mae_push/phase1/preds/baseline_xgb_desc217_{how}_val.csv')
        _BASE[how] = b.set_index('row')['pred']
    return _SPL[how]


def dedup_mask(how, rows):
    split(how)
    return np.array([r in _DEDUP[how] for r in rows])


def base_pred(how, rows):
    split(how)
    return _BASE[how].loc[rows].to_numpy(float)


class RidgeHead:
    def __init__(self, alphas=None):
        self.alphas = np.logspace(-2, 5, 22) if alphas is None else alphas

    def fit(self, X, y):
        from sklearn.linear_model import RidgeCV
        from sklearn.model_selection import KFold
        X = np.asarray(X, np.float64).copy()
        X[~np.isfinite(X)] = np.nan
        self.med = np.nanmedian(X, axis=0)
        self.med = np.where(np.isfinite(self.med), self.med, 0.0)
        X = np.where(np.isnan(X), self.med, X)
        self.lo, self.hi = np.percentile(X, [0.5, 99.5], axis=0)
        X = np.clip(X, self.lo, self.hi)
        self.mu, self.sd = X.mean(0), X.std(0)
        self.keep = self.sd > 1e-12
        Z = (X[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]
        self.m = RidgeCV(alphas=self.alphas, cv=KFold(5, shuffle=True, random_state=0),
                         scoring='neg_mean_absolute_error').fit(Z, y)
        self.alpha_ = float(self.m.alpha_)
        return self

    def predict(self, X):
        X = np.asarray(X, np.float64).copy()
        X[~np.isfinite(X)] = np.nan
        X = np.where(np.isnan(X), self.med, X)
        X = np.clip(X, self.lo, self.hi)
        Z = (X[:, self.keep] - self.mu[self.keep]) / self.sd[self.keep]
        return self.m.predict(Z)


def make_head(head, seed):
    if head == 'xgb':
        from xgboost import XGBRegressor
        return XGBRegressor(**mc.BASE_XGB, n_jobs=N_JOBS, random_state=seed)
    if head == 'ridge':
        return RidgeHead()
    raise ValueError(head)


def evaluate(name, X, head, features, how_list=('random', 'scaffold'), n_train=None, notes='',
             api_tokens='', api_cost='', stage='탐색', save=True, extra_infer_s_per_1k=0.0):
    """표현 X(19,945행) + head 를 두 분할의 train(또는 무작위 n_train 부분)으로 적합 → validation 예측.

    n_train=None: 전체 train. xgb 는 seed 3개(XGB_SEEDS), ridge 는 결정적이라 1회.
    n_train=int : 부분표본 seed 3개(SUB_SEEDS), xgb seed 는 같은 순서의 XGB_SEEDS.
    반환: {how: dict(rows, y, pred(seed 평균), per_seed MAE list, ...)}
    """
    y = y_all()
    out = {}
    for how in how_list:
        idx = split(how)
        tr, va = idx['train'], idx['validate']
        if n_train is None:
            plan = [(s, tr) for s in (XGB_SEEDS if head == 'xgb' else (XGB_SEEDS[0],))]
        else:
            plan = []
            for k, ss in enumerate(SUB_SEEDS):
                sub = np.sort(np.random.default_rng(ss).choice(tr, n_train, replace=False))
                plan.append((XGB_SEEDS[k], sub))
        preds, maes, maes_dd, alphas, inf_s = [], [], [], [], []
        dd = dedup_mask(how, va)
        with mc.CostMeter(stage) as c:
            for seed, rows in plan:
                mdl = make_head(head, seed)
                mdl.fit(X[rows], y[rows])
                t0 = time.perf_counter()
                p = np.asarray(mdl.predict(X[va]), float)
                inf_s.append((time.perf_counter() - t0) / len(va) * 1000)
                preds.append(p)
                maes.append(float(np.mean(np.abs(y[va] - p))))
                maes_dd.append(float(np.mean(np.abs(y[va][dd] - p[dd]))))
                if head == 'ridge':
                    alphas.append(mdl.alpha_)
        P = np.mean(preds, axis=0)
        met = mc.metrics(y[va], P)
        met_dd = mc.metrics(y[va][dd], P[dd])
        bref = base_pred(how, va)
        d_all = mc.paired_bootstrap(y[va], P, bref)
        d_dd = mc.paired_bootstrap(y[va][dd], P[dd], bref[dd])
        ntr = len(plan[0][1])
        tag = f'{name}_{how}' + ('' if n_train is None else f'_n{n_train}')
        if save:
            os.makedirs(os.path.join(PRED_DIR, 'seeds'), exist_ok=True)
            if n_train is None:
                pd.DataFrame({'row': va, 'y': y[va], 'pred': P}).to_csv(
                    os.path.join(PRED_DIR, f'{name}_{how}_val.csv'), index=False)
            pd.DataFrame({'row': va, **{f'pred_s{s}': p for (s, _), p in zip(plan, preds)}}).to_csv(
                os.path.join(PRED_DIR, 'seeds', f'{tag}_val_seeds.csv'), index=False)
        seeds_txt = ','.join(str(s) for s, _ in plan)
        sub_txt = '' if n_train is None else f'; 부분표본 seed {",".join(map(str, SUB_SEEDS[:len(plan)]))}'
        params = ('BASE_XGB n_jobs=8' if head == 'xgb' else
                  f'RidgeCV 5-fold(train) alpha={np.round(alphas, 3).tolist()} 전처리 대치·0.5/99.5 자르기·표준화')
        row = dict(run_id=f'T8_{tag}_{head}', track=TRACK, model='XGBoost' if head == 'xgb' else 'Ridge',
                   features=features, data_variant=('train 전체' if n_train is None else f'train 무작위 {n_train}'),
                   objective='MAE(validation)', split=how, seed=seeds_txt, params=params,
                   val_MAE=round(met['MAE'], 4), val_RMSE=round(met['RMSE'], 4), n_train=ntr,
                   api_tokens=api_tokens, api_cost=api_cost,
                   infer_s_per_1k=round(float(np.mean(inf_s)) + extra_infer_s_per_1k, 4),
                   notes=(f'seed평균예측 MAE {met["MAE"]:.3f} K (n {len(va)}); 시드별 MAE {np.mean(maes):.3f}±'
                          f'{np.std(maes, ddof=1) if len(maes) > 1 else 0:.3f}; 중복제거 MAE {met_dd["MAE"]:.3f} K '
                          f'(n {int(dd.sum())}); 기준선대비 ΔMAE {d_all[0]:+.3f} [{d_all[1]:+.3f},{d_all[2]:+.3f}] / '
                          f'중복제거 {d_dd[0]:+.3f} [{d_dd[1]:+.3f},{d_dd[2]:+.3f}]{sub_txt}. {notes}'),
                   **c.as_row())
        mc.append_ledger(LEDGER, row)
        out[how] = dict(rows=va, y=y[va], pred=P, preds=preds, maes=maes, maes_dd=maes_dd, mae=met['MAE'],
                        mae_dd=met_dd['MAE'], d_all=d_all, d_dd=d_dd, cpu_h=c.cpu_core_h, wall=c.wall_s)
        print(f'{tag:45s} {head:5s} MAE {met["MAE"]:.3f} (seeds {np.mean(maes):.3f}±'
              f'{np.std(maes, ddof=1) if len(maes) > 1 else 0:.3f}) dd {met_dd["MAE"]:.3f} '
              f'Δbase {d_all[0]:+.2f} [{d_all[1]:+.2f},{d_all[2]:+.2f}] cpu {c.cpu_core_h:.3f} h wall {c.wall_s:.0f}s',
              flush=True)
    return out
