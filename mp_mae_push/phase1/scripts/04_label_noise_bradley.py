"""04_label_noise_bradley.py : Bradley 원본(figshare, 측정 단위 28,645행)의 화합물 내 측정 산포로 라벨 잡음 추정.

입력 : external/bradley_figshare/bradley_open_mp_raw.csv, bradley_double_plus_good_raw.csv (CC0)
출력 : phase1/tables/mp_replicate_stats.csv        (mp.csv 행별 측정 수·범위·SD·정제판 포함 여부)
       phase1/tables/label_noise_bradley_ranges.csv   (반복측정 범위 분포, README 수치와 대조)
       phase1/tables/label_noise_bradley_estimates.csv(단일 측정 오차와 test 라벨 잡음 하한 추정)
       phase1/tables/baseline_mae_by_label_quality.csv(라벨 품질 구간별 기준선 오차: validation·test)
외부 값은 진단에만 쓴다 (특징·학습 라벨로 쓰지 않음).
"""
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import mpcommon as mc

RDLogger.DisableLog('rdApp.*')
EXT = os.path.join(mc.ROOT, 'external', 'bradley_figshare')
P1 = os.path.join(mc.ROOT, 'phase1')
T = os.path.join(P1, 'tables')
rng = np.random.default_rng(0)


def canon(s):
    m = Chem.MolFromSmiles(str(s))
    return Chem.MolToSmiles(m) if m is not None and m.GetNumAtoms() > 0 else None


raw = pd.read_csv(os.path.join(EXT, 'bradley_open_mp_raw.csv'))
raw['K'] = raw['mpC'].astype(float) + 273.15
raw['can'] = [canon(s) for s in raw['smiles']]
u = raw[raw['donotuse'].isna() & raw['can'].notna()].copy()
print(f'원본 {len(raw):,}행 | donotuse {int(raw.donotuse.notna().sum())} | SMILES 파싱 실패 {int(raw.can.isna().sum())} | 사용 {len(u):,}행, {u.can.nunique():,}종')

g = u.groupby('can')
stat = g['K'].agg(n_meas='size', median_K='median', min_K='min', max_K='max')
stat['range_K'] = stat['max_K'] - stat['min_K']
stat['sd_K'] = g['K'].std(ddof=1)
stat['n_sources'] = g['source'].nunique()

dpg = pd.read_csv(os.path.join(EXT, 'bradley_double_plus_good_raw.csv'))
dpg_can = {canon(s) for s in dpg['smiles']} - {None}

mp = pd.read_csv(os.path.join(mc.WEEK2, 'data', 'mp.csv'))
m = mp[['smiles', 'value']].join(stat, on='smiles')
m.insert(0, 'row', np.arange(len(m)))
m['in_dpg'] = m['smiles'].isin(dpg_can)
found = m['n_meas'].notna()
agree = (np.abs(m.loc[found, 'value'] - m.loc[found, 'median_K']) < 0.02)
print(f'mp.csv {len(m):,}행 중 원본에서 찾은 행 {int(found.sum()):,} ({found.mean():.1%}) | 그중 값 = 원본 중앙값 {agree.mean():.1%} | 정제판(DPG) 포함 {int(m.in_dpg.sum()):,}')
m.to_csv(os.path.join(T, 'mp_replicate_stats.csv'), index=False)

# ---- 반복측정 범위 분포 (README: 4,679종 반복, 410종 >= 10 K, 86종 >= 50 K)
rep = m[found & (m['n_meas'] >= 2)]
rows = [dict(item='반복측정 있는 분자 수', value=len(rep), readme=4679),
        dict(item='범위 >= 10 K', value=int((rep.range_K >= 10).sum()), readme=410),
        dict(item='범위 >= 50 K', value=int((rep.range_K >= 50).sum()), readme=86),
        dict(item='범위 < 0.01 K (모두 같은 값)', value=int((rep.range_K < 0.01).sum()), readme=''),
        dict(item='범위 < 1 K', value=int((rep.range_K < 1).sum()), readme=''),
        dict(item='범위 < 5 K', value=int((rep.range_K < 5).sum()), readme=''),
        dict(item='범위 중앙값 K', value=float(rep.range_K.median()), readme=''),
        dict(item='범위 평균 K', value=float(rep.range_K.mean()), readme=''),
        dict(item='분자당 측정 수 평균(반복 있는 분자)', value=float(rep.n_meas.mean()), readme=''),
        dict(item='측정 1개뿐인 분자 수', value=int((m.n_meas == 1).sum()), readme='')]
pd.DataFrame(rows).to_csv(os.path.join(T, 'label_noise_bradley_ranges.csv'), index=False)
print(pd.DataFrame(rows).to_string(index=False))

# ---- 화합물 내 측정 쌍의 차이 d = e1 - e2  (mp.csv 에 들어간 분자만)
keep = set(rep['smiles'])
uu = u[u['can'].isin(keep)]
d_all, d_cross = [], []
for can, grp in uu.groupby('can'):
    k = grp['K'].to_numpy()
    i, j = np.triu_indices(len(k), 1)
    w = 1.0 / len(i)                               # 분자마다 같은 무게
    d_all += [(k[a] - k[b], w) for a, b in zip(i, j)]
    s = grp.groupby('source')['K'].first().to_numpy()   # 출처가 다른 측정끼리만 (출처당 1개)
    if len(s) >= 2:
        i, j = np.triu_indices(len(s), 1)
        w = 1.0 / len(i)
        d_cross += [(s[a] - s[b], w) for a, b in zip(i, j)]


def summarize(pairs, label):
    d = np.array([p[0] for p in pairs]); w = np.array([p[1] for p in pairs]); w = w / w.sum()
    Ead = float(np.sum(w * np.abs(d))); Ed2 = float(np.sum(w * d ** 2))
    nz = np.abs(d) >= 0.01
    wn = w[nz] / w[nz].sum()
    return d, w, dict(pairs=label, n_pairs=len(d), frac_identical=float(w[~nz].sum()),
                      E_abs_d_K=Ead, sigma_single_K=float(np.sqrt(Ed2 / 2)),
                      single_MAE_lower_K=Ead / 2, single_MAE_normal_K=Ead / np.sqrt(2), single_MAE_upper_K=Ead,
                      E_abs_d_nonidentical_K=float(np.sum(wn * np.abs(d[nz]))),
                      single_MAE_normal_nonidentical_K=float(np.sum(wn * np.abs(d[nz])) / np.sqrt(2)))


est = []
d1, w1, r1 = summarize(d_all, '같은 분자의 모든 측정 쌍'); est.append(r1)
d2, w2, r2 = summarize(d_cross, '출처가 다른 측정 쌍'); est.append(r2)

# ---- test 라벨 잡음 하한: 완벽한 모델이라도 남는 MAE = E|라벨 오차|
#      단일 측정 오차 e* = d/sqrt(2) (d 는 위 경험 분포에서 추출; 분산은 정확, 모양은 근사). n>=2 는 n개 추출의 중앙값.
def floor_for(rows_idx, d, w, n_sim=400):
    n = m.loc[rows_idx, 'n_meas'].fillna(1).astype(int).clip(upper=16).to_numpy()
    tot = np.zeros(n_sim)
    for nn in np.unique(n):
        cnt = int((n == nn).sum())
        e = rng.choice(d, size=(n_sim, cnt, nn), p=w) / np.sqrt(2)
        tot += np.abs(np.median(e, axis=2)).sum(axis=1)
    v = tot / len(n)
    return float(v.mean()), float(np.sqrt(np.mean((rng.choice(d, size=200000, p=w) / np.sqrt(2)) ** 2)))


for how in ('random', 'scaffold'):
    idx, _ = mc.load_split(how)
    for nm, d, w in (('모든 쌍', d1, w1), ('출처가 다른 쌍', d2, w2)):
        fl, _ = floor_for(idx['test'], d, w)
        te = m.loc[idx['test']]
        est.append(dict(pairs=f'[{how} test n={len(te)}] 라벨 잡음 MAE 하한 ({nm} 기준, e*=d/√2)', n_pairs=len(d),
                        E_abs_d_K=np.nan, single_MAE_normal_K=fl, single_MAE_lower_K=fl / np.sqrt(2), single_MAE_upper_K=fl * np.sqrt(2),
                        frac_identical=float((te.n_meas.fillna(1) >= 2).mean())))
pd.DataFrame(est).to_csv(os.path.join(T, 'label_noise_bradley_estimates.csv'), index=False)
print(pd.DataFrame(est).round(3).to_string(index=False))

# ---- 라벨 품질 구간별 기준선 오차 (validation 과 test; 구간 정의는 모델과 무관)
def qual(r):
    if r['in_dpg']:
        return '1 정제판(DPG: 측정>=2, 범위<=5 K)'
    if not r['n_meas'] >= 2:
        return '5 측정 1개'
    if r['range_K'] < 5:
        return '2 반복, 범위 <5 K (DPG 외)'
    if r['range_K'] < 10:
        return '3 반복, 범위 5–10 K'
    return '4 반복, 범위 >=10 K'


m['quality'] = m.apply(qual, axis=1)
heavy = pd.Series([Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in m['smiles']])
out = []
files = [('random', 'val', 'n_jobs=8 재적합', os.path.join(P1, 'preds', 'baseline_xgb_desc217_random_val.csv')),
         ('random', 'test', 'n_jobs=8 재적합', os.path.join(P1, 'preds', 'baseline_xgb_desc217_random_test.csv')),
         ('random', 'test', '원본 기준 예측', os.path.join(mc.ROOT, 'reference', 'orig_baseline_random_test_pred.csv')),
         ('scaffold', 'val', 'n_jobs=8 재적합', os.path.join(P1, 'preds', 'baseline_xgb_desc217_scaffold_val.csv')),
         ('scaffold', 'test', 'n_jobs=8 재적합', os.path.join(P1, 'preds', 'baseline_xgb_desc217_scaffold_test.csv'))]
for how, part, model, path in files:
    p = pd.read_csv(path)
    if 'row' not in p:
        idx, _ = mc.load_split(how)
        p.insert(0, 'row', idx['test'])
    p['abs_err'] = np.abs(p['y'] - p['pred'])
    p['quality'] = m.loc[p['row'], 'quality'].to_numpy()
    p['heavy'] = heavy.loc[p['row']].to_numpy()
    for q, grp in list(p.groupby('quality')) + [('전체', p)]:
        lo, hi = mc.boot_ci(grp['abs_err'].to_numpy())
        out.append(dict(split=how, part=part, model=model, quality=q, n=len(grp), MAE_K=float(grp.abs_err.mean()),
                        ci_lo=lo, ci_hi=hi, RMSE_K=float(np.sqrt(np.mean((grp.y - grp.pred) ** 2))),
                        mp_mean_K=float(grp.y.mean()), heavy_atoms_mean=float(grp.heavy.mean())))
tab = pd.DataFrame(out)
tab.to_csv(os.path.join(T, 'baseline_mae_by_label_quality.csv'), index=False)
print(tab[(tab.model != '원본 기준 예측')].round(2).to_string(index=False))
