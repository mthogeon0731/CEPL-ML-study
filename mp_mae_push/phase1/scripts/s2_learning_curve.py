"""s2_learning_curve.py : 고정 test 학습 곡선 (기준선 XGB 설정 고정, n_jobs=8).

random / scaffold 각각 train 의 10/25/50/100 % (부분집합 seed 0,1,2 = XGB random_state; 100 % 는 모델 seed 만).
출력: phase1/tables/learning_curve.csv, learning_curve_summary.csv, learning_curve_fit.csv, learning_curve_extrap.csv
"""
import json
import os
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import mpcommon as mc

OUT = os.path.join(mc.ROOT, 'phase1', 'tables')
LEDGER = os.path.join(mc.ROOT, 'phase1', 'ledger_errdecomp.csv')
os.makedirs(OUT, exist_ok=True)
N_JOBS = 8
FRACS, SEEDS = [0.10, 0.25, 0.50, 1.00], [0, 1, 2]

df, mols, y = mc.load_data()
X = mc.desc217(mols)
assert X.shape == (19945, 217), X.shape

rows = []
for how in ('random', 'scaffold'):
    idx, _ = mc.load_split(how)
    tr, va, te = idx['train'], idx['validate'], idx['test']
    for frac in FRACS:
        for seed in SEEDS:
            n = int(round(frac * len(tr)))
            if frac < 1.0:
                pos = np.sort(np.random.default_rng(seed).choice(len(tr), n, replace=False))   # train 안 순서 유지
                sub = tr[pos]
            else:
                sub = tr
            with mc.CostMeter('탐색') as c:
                t0 = time.perf_counter()
                m = XGBRegressor(**mc.BASE_XGB, n_jobs=N_JOBS, random_state=seed)
                m.fit(X[sub], y[sub])
                fit_s = time.perf_counter() - t0
                pv, pt = m.predict(X[va]), m.predict(X[te])
            mv, mt = mc.metrics(y[va], pv), mc.metrics(y[te], pt)
            rows.append(dict(split=how, frac=frac, n_train=len(sub), seed=seed, val_MAE=mv['MAE'], test_MAE=mt['MAE'],
                             val_RMSE=mv['RMSE'], test_RMSE=mt['RMSE'], wall_s=round(fit_s, 3), n_val=len(va), n_test=len(te)))
            mc.append_ledger(LEDGER, dict(
                run_id=f'errdecomp_s2_lc_{how}_f{int(frac * 100):03d}_s{seed}', track='errdecomp', model='XGBoost(base)',
                features='RDKit desc 217', data_variant=f'train {int(frac * 100)}% subset', objective='reg:squarederror',
                split=how, seed=seed, params=json.dumps(dict(mc.BASE_XGB, n_jobs=N_JOBS, random_state=seed)),
                val_MAE=round(mv['MAE'], 4), val_RMSE=round(mv['RMSE'], 4), test_MAE=round(mt['MAE'], 4),
                test_RMSE=round(mt['RMSE'], 4), test_R2=round(mt['R2'], 4), n_train=len(sub),
                notes='학습 곡선(진단); test 는 고정 기준선 설정의 곡선 기록용', **c.as_row()))
            print(f'{how} frac={frac:.2f} n={len(sub)} seed={seed} val={mv["MAE"]:.3f} test={mt["MAE"]:.3f} fit={fit_s:.1f}s', flush=True)

L = pd.DataFrame(rows)
L.to_csv(os.path.join(OUT, 'learning_curve.csv'), index=False, encoding='utf-8-sig')

# ---- 요약: 평균 +- SD ------------------------------------------------------------------------
S = (L.groupby(['split', 'frac', 'n_train'])
       .agg(n_seed=('seed', 'size'), val_MAE_mean=('val_MAE', 'mean'), val_MAE_sd=('val_MAE', 'std'),
            test_MAE_mean=('test_MAE', 'mean'), test_MAE_sd=('test_MAE', 'std'),
            val_MAE_min=('val_MAE', 'min'), val_MAE_max=('val_MAE', 'max'),
            test_MAE_min=('test_MAE', 'min'), test_MAE_max=('test_MAE', 'max'), wall_s_mean=('wall_s', 'mean'))
       .reset_index())
S.to_csv(os.path.join(OUT, 'learning_curve_summary.csv'), index=False, encoding='utf-8-sig')


# ---- MAE = a + b log2(n) 적합 ---------------------------------------------------------------
def fit_line(x, yv):
    """보통최소제곱. -> a, b, se_b (잔차 자유도 n-2)"""
    A = np.c_[np.ones_like(x), x]
    coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
    res = yv - A @ coef
    s2 = res @ res / (len(x) - 2)
    cov = s2 * np.linalg.inv(A.T @ A)
    return coef[0], coef[1], float(np.sqrt(cov[1, 1]))


def boot_slope(g, col, n_boot=2000, seed=0):
    """frac 수준 안에서 seed 를 복원추출해 b 의 95 % 구간 (seed 변동만 반영; test 표본 변동은 아님)."""
    rng = np.random.default_rng(seed)
    groups = [d[col].values for _, d in g.groupby('frac')]
    xs = [np.log2(d['n_train'].values) for _, d in g.groupby('frac')]
    out = []
    for _ in range(n_boot):
        yy = np.concatenate([v[rng.integers(0, len(v), len(v))] for v in groups])
        out.append(fit_line(np.concatenate(xs), yy)[1])
    return np.percentile(out, [2.5, 97.5])


fits, extra = [], []
for how, g in L.groupby('split'):
    n_full = int(g['n_train'].max())
    for col in ('val_MAE', 'test_MAE'):
        for label, gg in (('all 4 fracs', g), ('upper 3 fracs (25-100 %)', g[g.frac >= 0.25]), ('last doubling (50->100 %)', g[g.frac >= 0.5])):
            x, yv = np.log2(gg['n_train'].values.astype(float)), gg[col].values
            a, b, se = fit_line(x, yv)
            lo, hi = boot_slope(gg, col)
            fits.append(dict(split=how, metric=col, points=label, n_obs=len(gg), a_K=a, b_K_per_doubling=b, b_se_K=se,
                             b_ci95_lo_ols=b - 1.96 * se, b_ci95_hi_ols=b + 1.96 * se, b_boot_lo=lo, b_boot_hi=hi,
                             mae_at_full_fit_K=a + b * np.log2(n_full), n_full=n_full))
            if label != 'last doubling (50->100 %)':
                full_obs = g.loc[g.frac == 1.0, col].mean()
                for mult, name in ((2, 'x2'), (4, 'x4'), (300000 / n_full, 'x%.1f (n=300,000)' % (300000 / n_full))):
                    d = b * np.log2(mult)
                    extra.append(dict(split=how, metric=col, points=label, train_multiple=name, n_train_extrap=int(round(n_full * mult)),
                                      mae_full_obs_K=full_obs, delta_K=d, delta_lo_K=(b - 1.96 * se) * np.log2(mult),
                                      delta_hi_K=(b + 1.96 * se) * np.log2(mult), mae_extrap_K=full_obs + d,
                                      mae_extrap_lo_K=full_obs + (b - 1.96 * se) * np.log2(mult),
                                      mae_extrap_hi_K=full_obs + (b + 1.96 * se) * np.log2(mult)))
F = pd.DataFrame(fits)
F.to_csv(os.path.join(OUT, 'learning_curve_fit.csv'), index=False, encoding='utf-8-sig')
E = pd.DataFrame(extra)
E.to_csv(os.path.join(OUT, 'learning_curve_extrap.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(S.round(3).to_string(index=False))
print(F[['split', 'metric', 'points', 'b_K_per_doubling', 'b_se_K', 'b_boot_lo', 'b_boot_hi']].round(3).to_string(index=False))
