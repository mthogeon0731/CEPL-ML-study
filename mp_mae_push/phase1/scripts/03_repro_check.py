"""03_repro_check.py : 기준선 27.50 K 와 재현값 27.63 K 차이의 원인 확인.

(1) 원본 예측 파일과 test 분자·라벨이 같은가  (2) 분자별 예측 차이  (3) 스레드 수·반복 실행에 따른 변동
"""
import os

import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBRegressor

import mpcommon as mc

P1 = os.path.join(mc.ROOT, 'phase1')
orig = pd.read_csv(os.path.join(mc.WEEK2, 'results', 'mp_XGBoost-Descriptor_random_test_pred.csv'))
mine = pd.read_csv(os.path.join(P1, 'preds', 'baseline_xgb_desc217_random_test.csv'))
print(f'원본 n={len(orig)} MAE={np.mean(np.abs(orig.y - orig.pred)):.4f} | 재현 n={len(mine)} MAE={np.mean(np.abs(mine.y - mine.pred)):.4f}')
same_order = len(orig) == len(mine) and (orig.smiles.values == mine.smiles.values).all()
print('test SMILES 같은 순서:', same_order, '| 집합 일치:', set(orig.smiles) == set(mine.smiles),
      '| 라벨 최대 차이 K:', float(np.max(np.abs(orig.y.values - mine.y.values))) if same_order else 'n/a')
if same_order:
    d = mine.pred.values - orig.pred.values
    print(f'예측 차이(재현-원본): 평균 {d.mean():+.3f}, |차이| 중앙값 {np.median(np.abs(d)):.3f}, 95% {np.percentile(np.abs(d), 95):.3f}, '
          f'최대 {np.abs(d).max():.3f} K | 정확히 같은 분자 {(np.abs(d) < 1e-3).mean():.1%} | r={np.corrcoef(mine.pred, orig.pred)[0, 1]:.5f}')
    diff, lo, hi = mc.paired_bootstrap(mine.y.values, mine.pred.values, orig.pred.values)
    print(f'짝지은 부트스트랩 MAE(재현)-MAE(원본) = {diff:+.3f} K [{lo:+.3f}, {hi:+.3f}]')

df, mols, y = mc.load_data()
X = mc.desc217(mols)
idx, _ = mc.load_split('random')
tr, te = idx['train'], idx['test']
print('xgboost', xgboost.__version__, '| os.cpu_count', os.cpu_count(), '| X nan', int(np.isnan(X).sum()),
      '| X dtype', X.dtype)
rows = []
for nj, rep in [(-1, 0), (-1, 1), (12, 0), (8, 0), (4, 0), (1, 0)]:
    m = XGBRegressor(**mc.BASE_XGB, n_jobs=nj, random_state=mc.SEED).fit(X[tr], y[tr])
    p = m.predict(X[te])
    r = dict(n_jobs=nj, rep=rep, test_MAE=mc.metrics(y[te], p)['MAE'], test_RMSE=mc.metrics(y[te], p)['RMSE'])
    if same_order:
        r['max_abs_diff_vs_orig'] = float(np.abs(p - orig.pred.values).max())
        r['max_abs_diff_vs_first'] = float(np.abs(p - rows[0]['_p']).max()) if rows else 0.0
    r['_p'] = p
    rows.append(r)
tab = pd.DataFrame([{k: v for k, v in r.items() if k != '_p'} for r in rows])
tab.to_csv(os.path.join(P1, 'tables', 'repro_check_threads.csv'), index=False)
print(tab.round(4).to_string(index=False))
