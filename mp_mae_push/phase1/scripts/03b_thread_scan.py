"""03b_thread_scan.py : n_jobs 1..12 전부 적합해 원본 예측(27.4995 K)과 정확히 일치하는 스레드 수가 있는지 확인."""
import os

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import mpcommon as mc

P1 = os.path.join(mc.ROOT, 'phase1')
orig = pd.read_csv(os.path.join(mc.WEEK2, 'results', 'mp_XGBoost-Descriptor_random_test_pred.csv'))
df, mols, y = mc.load_data()
X = mc.desc217(mols)
idx, _ = mc.load_split('random')
tr, va, te = idx['train'], idx['validate'], idx['test']
assert (df['smiles'].values[te] == orig.smiles.values).all()
rows, preds = [], {}
for nj in range(1, 13):
    m = XGBRegressor(**mc.BASE_XGB, n_jobs=nj, random_state=mc.SEED).fit(X[tr], y[tr])
    p = m.predict(X[te])
    preds[nj] = p
    same_as = [k for k in preds if k != nj and np.abs(preds[k] - p).max() < 1e-6]
    rows.append(dict(n_jobs=nj, val_MAE=mc.metrics(y[va], m.predict(X[va]))['MAE'], test_MAE=mc.metrics(y[te], p)['MAE'],
                     test_RMSE=mc.metrics(y[te], p)['RMSE'], max_abs_diff_vs_orig=float(np.abs(p - orig.pred.values).max()),
                     identical_to_n_jobs=same_as[0] if same_as else ''))
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(P1, 'tables', 'repro_check_thread_scan.csv'), index=False)
print(tab.round(4).to_string(index=False))
u = tab.drop_duplicates('test_MAE')
print(f'서로 다른 적합 {len(u)}개 | test MAE 평균 {u.test_MAE.mean():.3f} SD {u.test_MAE.std(ddof=1):.3f} '
      f'범위 {u.test_MAE.min():.3f}–{u.test_MAE.max():.3f} | 원본과 일치(최대 차이<0.01 K): '
      f'{list(tab.loc[tab.max_abs_diff_vs_orig < 0.01, "n_jobs"])}')
