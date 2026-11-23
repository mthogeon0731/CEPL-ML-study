"""T3 단계 2 요약: work/*.npy (validation 예측) -> 표·예측 CSV.

- 짝지은 차이: 같은 head·같은 seed의 RDKit217 대비 분자별 d_i = mean_s(|e_new,s| - |e_ref,s|),
  d_i 부트스트랩 95% CI (2,000회, seed 0). 두 기준(전체 / 중복 제거 validation).
- seed 평균 예측의 MAE, seed별 MAE 평균 ± SD.
- 오차 상관: 잔차(y - pred, seed 평균)의 Pearson r — 같은 head R217, 기준선 파일(XGB seed 42).
- 고정 50:50 평균(검증 튜닝 없음): 탐색적 참고치.
"""
import glob
import os
import re
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('mp_mae_push/common'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import mpcommon as mc  # noqa: E402

T3 = 'T3_pretrained'
Y = np.load(os.path.join(T3, 'work_y.npy'))
IDX = {s: mc.load_split(s)[0] for s in ('random', 'scaffold')}
DED = {}
for s in IDX:
    d = pd.read_csv(f'mp_mae_push/phase2/prereg/dedup_rows_{s}.csv')
    d = d[d['set'] == 'validate']
    keep = set(d.loc[d['overlap'] == 0, 'row'])
    DED[s] = np.array([r in keep for r in IDX[s]['validate']])
BASE = {s: pd.read_csv(f'{T3}/inputs/baseline_xgb_desc217_{s}_val.csv') for s in IDX}

runs = {}
for f in glob.glob(f'{T3}/work/*.npy'):
    m = re.match(r'(xgb|ridge)_(.+)_(random|scaffold)_s(\d)\.npy', os.path.basename(f))
    head, rep, split, seed = m.groups()
    if int(seed) > 2:   # seed 3-5 (R217 배깅 대조)는 표 밖 — s2_blend5050_vs_R217_6seed.csv
        continue
    runs.setdefault((head, rep, split), {})[int(seed)] = np.load(f)


def boot(d, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), *np.percentile(b, [2.5, 97.5])


rows = []
os.makedirs(f'{T3}/preds', exist_ok=True)
for (head, rep, split), P in sorted(runs.items()):
    seeds = sorted(P)
    vi = IDX[split]['validate']; y = Y[vi]; dm = DED[split]
    pm = np.mean([P[s] for s in seeds], 0)
    pd.DataFrame(dict(row=vi, y=y, pred=pm)).to_csv(f'{T3}/preds/{head}_{rep}_{split}_val.csv', index=False)
    per_seed = [np.mean(np.abs(y - P[s])) for s in seeds]
    per_seed_d = [np.mean(np.abs(y - P[s])[dm]) for s in seeds]
    r = dict(head=head, rep=rep, split=split, n_seeds=len(seeds), seeds=','.join(map(str, seeds)),
             MAE_seedmean=np.mean(per_seed), MAE_seedsd=np.std(per_seed, ddof=1) if len(seeds) > 1 else np.nan,
             MAE_dedup_seedmean=np.mean(per_seed_d),
             MAE_avgpred=np.mean(np.abs(y - pm)), MAE_avgpred_dedup=np.mean(np.abs(y - pm)[dm]),
             RMSE_avgpred=float(np.sqrt(np.mean((y - pm) ** 2))))
    ref = runs.get((head, 'R217', split))
    if ref is not None and rep != 'R217':
        common = [s for s in seeds if s in ref]
        d = np.mean([np.abs(y - P[s]) - np.abs(y - ref[s]) for s in common], 0)
        r.update(paired_seeds=','.join(map(str, common)))
        for tag, mask in (('full', slice(None)), ('dedup', dm)):
            mu, lo, hi = boot(d[mask])
            r.update({f'dMAE_{tag}': mu, f'dMAE_{tag}_lo': lo, f'dMAE_{tag}_hi': hi})
        pr = np.mean([ref[s] for s in common], 0)
        r['err_corr_same_head_R217'] = float(np.corrcoef(y - pm, y - pr)[0, 1])
        r['MAE_R217_same_head_avgpred'] = float(np.mean(np.abs(y - pr)))
        r['MAE_5050_with_R217_same_head'] = float(np.mean(np.abs(y - 0.5 * (pm + pr))))
        d5 = np.abs(y - 0.5 * (pm + pr)) - np.abs(y - pr)   # 50:50 대 R217 seed 평균 예측 (탐색적)
        for tag, mask in (('full', slice(None)), ('dedup', dm)):
            mu, lo, hi = boot(d5[mask])
            r.update({f'd5050_{tag}': mu, f'd5050_{tag}_lo': lo, f'd5050_{tag}_hi': hi})
    b = BASE[split]
    r['err_corr_baseline_s42'] = float(np.corrcoef(y - pm, b['y'].values - b['pred'].values)[0, 1])
    rows.append(r)

T = pd.DataFrame(rows)
for h in ('xgb', 'ridge'):
    for s in ('random', 'scaffold'):
        k = (T['head'] == h) & (T['split'] == s)
        T.loc[k, 'rank_full'] = T.loc[k, 'MAE_seedmean'].rank(method='min')
        T.loc[k, 'rank_dedup'] = T.loc[k, 'MAE_dedup_seedmean'].rank(method='min')
T['rank_differs'] = T.rank_full != T.rank_dedup


def verdict(r):
    if r['rep'] == 'R217' or pd.isna(r.get('dMAE_full')):
        return 'reference'
    if not r['rep'].startswith('R217_'):  # 파일명에서 '+' -> '_'
        return 'embedding only (vs R217)' + (': better' if r['dMAE_full_hi'] < 0 else ': not better')
    sig_full = r['dMAE_full_hi'] < 0
    sig_dedup = r['dMAE_dedup_hi'] < 0
    if sig_full and sig_dedup:
        return 'combined gain (both bases)'
    if sig_full or sig_dedup:
        return 'combined gain (one base only)'
    return 'ensemble member only'


T['verdict'] = T.apply(verdict, axis=1)
T.to_csv(f'{T3}/tables/s2_val_summary.csv', index=False)
print(T[['head', 'rep', 'split', 'n_seeds', 'MAE_seedmean', 'MAE_seedsd', 'MAE_dedup_seedmean', 'dMAE_full', 'dMAE_full_lo',
         'dMAE_full_hi', 'dMAE_dedup', 'dMAE_dedup_lo', 'dMAE_dedup_hi', 'err_corr_same_head_R217', 'err_corr_baseline_s42',
         'MAE_R217_same_head_avgpred', 'MAE_5050_with_R217_same_head', 'd5050_full_lo', 'd5050_full_hi', 'd5050_dedup_lo', 'd5050_dedup_hi', 'rank_full', 'rank_dedup', 'verdict']].round(3).to_string())
