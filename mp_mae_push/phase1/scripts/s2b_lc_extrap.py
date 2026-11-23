"""s2b_lc_extrap.py : 학습 곡선 외삽 (모델 적합 없음; learning_curve.csv 만 읽는다).

세 가지 외삽 모형을 나란히 둔다 — 로그선형은 구간별 기울기가 줄어드는 것을 무시하므로 낙관적.
  M1  MAE = a + b log2(n), 4개 수준 전부        (낙관)
  M2  MAE = a + b log2(n), 상위 3개 수준(25-100 %)
  M3  MAE = c + A n^(-beta)  (바닥 c 가 있는 거듭제곱; 수준 4개에 모수 3개라 자유도 1 — 불확도 큼)
  M4  마지막 구간(50->100 %) 기울기가 2배마다 r 배로 줄어든다고 가정 (r = 앞 두 구간 기울기 비의 평균)
출력: phase1/tables/learning_curve_extrap.csv (덮어씀), learning_curve_segments.csv
"""
import os

import numpy as np
import pandas as pd
from scipy import optimize

import mpcommon as mc

TAB = os.path.join(mc.ROOT, 'phase1', 'tables')
L = pd.read_csv(os.path.join(TAB, 'learning_curve.csv'))
N_EXT = 300_000


def powerlaw(n, c, A, beta):
    return c + A * n ** (-beta)


def fit_power(n, v):
    best = None
    for b0 in (0.2, 0.5, 1.0):
        try:
            p, _ = optimize.curve_fit(powerlaw, n, v, p0=[v.min() - 5, (v.max() - v.min()) * n.min() ** b0, b0],
                                      bounds=([0, 0, 0.01], [v.min(), np.inf, 3.0]), maxfev=20000)
            sse = np.sum((powerlaw(n, *p) - v) ** 2)
            if best is None or sse < best[1]:
                best = (p, sse)
        except Exception:
            pass
    return best[0]


def resample(g, col, rng):
    parts = []
    for _, d in g.groupby('frac'):
        parts.append(d.iloc[rng.integers(0, len(d), len(d))][['n_train', col]])
    return pd.concat(parts)


seg_rows, ext_rows = [], []
for how, g in L.groupby('split'):
    n_full = int(g['n_train'].max())
    targets = [(2 * n_full, 'x2'), (4 * n_full, 'x4'), (N_EXT, 'x%.1f (n=300,000)' % (N_EXT / n_full))]
    for col in ('val_MAE', 'test_MAE'):
        m = g.groupby('n_train')[col].mean().sort_index()
        ns, vs = m.index.values.astype(float), m.values
        slopes = np.diff(vs) / np.diff(np.log2(ns))               # 구간별 2배당 변화
        for i in range(len(slopes)):
            seg_rows.append(dict(split=how, metric=col, n_from=int(ns[i]), n_to=int(ns[i + 1]), mae_from_K=vs[i], mae_to_K=vs[i + 1],
                                 doublings=np.log2(ns[i + 1] / ns[i]), slope_K_per_doubling=slopes[i]))
        r_point = float(np.mean(slopes[1:] / slopes[:-1]))        # 기울기 감쇠비 (인접 구간의 2배당 기울기 비의 평균)
        full_obs = vs[-1]

        def predict_all(gg):
            x, yv = np.log2(gg['n_train'].values.astype(float)), gg[col].values
            out = {}
            b1 = np.polyfit(x, yv, 1)[0]
            u = gg[gg['n_train'] >= ns[1]]
            b2 = np.polyfit(np.log2(u['n_train'].values.astype(float)), u[col].values, 1)[0]
            p = fit_power(gg['n_train'].values.astype(float), yv)
            mm = gg.groupby('n_train')[col].mean().sort_index().values
            sl = np.diff(mm) / np.diff(np.log2(ns))
            r = float(np.mean(sl[1:] / sl[:-1]))
            s_last = sl[-1]
            for nt, name in targets:
                d = np.log2(nt / n_full)
                out[('M1 loglinear all', name)] = b1 * d
                out[('M2 loglinear upper3', name)] = b2 * d
                out[('M3 powerlaw+floor', name)] = powerlaw(nt, *p) - powerlaw(n_full, *p)
                k = np.arange(1, int(np.floor(d)) + 1)
                geo = np.sum(s_last * r ** k) + s_last * r ** (np.floor(d) + 1) * (d - np.floor(d))
                out[('M4 decaying slope', name)] = geo
            out[('M3 powerlaw+floor', 'asymptote c')] = p[0] - mm[-1]
            out[('M4 decaying slope', 'asymptote c')] = s_last * r / (1 - r) if 0 < r < 1 else np.nan
            return out

        point = predict_all(g)
        rng = np.random.default_rng(0)
        boots = [predict_all(resample(g, col, rng)) for _ in range(500)]
        for key, d in point.items():
            bv = np.array([b[key] for b in boots], float)
            lo, hi = np.nanpercentile(bv, [2.5, 97.5])
            ext_rows.append(dict(split=how, metric=col, model=key[0], target=key[1], n_full=n_full, mae_full_obs_K=full_obs,
                                 delta_K=d, delta_boot_lo_K=lo, delta_boot_hi_K=hi, mae_extrap_K=full_obs + d,
                                 mae_extrap_lo_K=full_obs + lo, mae_extrap_hi_K=full_obs + hi, slope_decay_ratio_r=r_point,
                                 note='boot = seed 복원추출 500회(수준 안); test 표본 변동·외부 데이터 분포 차이는 미반영'))
pd.DataFrame(seg_rows).to_csv(os.path.join(TAB, 'learning_curve_segments.csv'), index=False, encoding='utf-8-sig')
E = pd.DataFrame(ext_rows)
E.to_csv(os.path.join(TAB, 'learning_curve_extrap.csv'), index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(pd.DataFrame(seg_rows).round(3).to_string(index=False))
print(E[E.metric == 'val_MAE'][['split', 'model', 'target', 'delta_K', 'delta_boot_lo_K', 'delta_boot_hi_K', 'mae_extrap_K']].round(2).to_string(index=False))
