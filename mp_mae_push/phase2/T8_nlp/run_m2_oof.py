"""run_m2_oof.py : train 안 5-fold OOF 로 RDKit217 vs RDKit217+F5 (XGBoost, seed 42) — 부분집합(>550 K, 고대칭) 보강용.

validation·test 행은 쓰지 않는다. 출력: T8_nlp/preds/oof_<name>_<split>_train.csv (row,y,pred)
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('T8_nlp'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

import t8eval as te  # noqa: E402

mc = te.mc
y = te.y_all()
X0 = np.load('mp_mae_push/cache/X_desc217.npy').astype(np.float32)
F5 = np.load('T8_nlp/features/F5_rules.npy').astype(np.float32)
reps = {'REF_R217_xgb': X0, 'M2_R217F5_xgb': np.hstack([X0, F5])}
for how in ('random', 'scaffold'):
    tr = te.split(how)['train']
    for nm, X in reps.items():
        out = os.path.join(te.PRED_DIR, f'oof_{nm}_{how}_train.csv')
        if os.path.exists(out):
            continue
        p = np.zeros(len(tr))
        with mc.CostMeter('탐색') as c:
            for k, (a, b) in enumerate(KFold(5, shuffle=True, random_state=0).split(tr)):
                m = te.make_head('xgb', 42).fit(X[tr[a]], y[tr[a]])
                p[b] = m.predict(X[tr[b]])
        pd.DataFrame({'row': tr, 'y': y[tr], 'pred': p}).to_csv(out, index=False)
        mae = float(np.mean(np.abs(y[tr] - p)))
        mc.append_ledger(te.LEDGER, dict(run_id=f'T8_oof_{nm}_{how}', track=te.TRACK, model='XGBoost',
                                         features=nm, data_variant='train 5-fold OOF(KFold shuffle seed 0)',
                                         objective='MAE(train OOF)', split=how, seed='42', params='BASE_XGB n_jobs=8',
                                         n_train=len(tr), notes=f'train OOF MAE {mae:.3f} K (n {len(tr)}); 부분집합 보강용, val_MAE 열 비움',
                                         **c.as_row()))
        print(nm, how, round(mae, 3), round(c.cpu_core_h, 3), flush=True)
print('DONE', flush=True)
