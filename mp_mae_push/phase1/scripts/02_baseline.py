"""02_baseline.py : 기준선 재현 — train_xgb.run('mp', <split>, 'desc', max_n=None) 과 같은 계산.

원본 run() 은 수업 폴더의 results/summary.csv 에 줄을 추가하므로 호출하지 않고,
원본의 load/split/featurize/metrics 와 같은 XGBoost 설정으로 계산한 뒤 결과만 작업 폴더에 저장한다.
"""
import json
import os
import time

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

import mpcommon as mc

try:
    import winreg
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
        print('CPU:', winreg.QueryValueEx(k, 'ProcessorNameString')[0].strip())
except Exception as e:  # noqa: BLE001
    print('CPU 이름 조회 실패:', type(e).__name__)

P1 = os.path.join(mc.ROOT, 'phase1')
LEDGER = os.path.join(P1, 'ledger.csv')
os.makedirs(os.path.join(P1, 'preds'), exist_ok=True)

df, mols, y = mc.load_data()
with mc.CostMeter('특징') as cf:
    X = mc.desc217(mols, cache=False)
os.makedirs(mc.CACHE_DIR, exist_ok=True)
np.save(os.path.join(mc.CACHE_DIR, 'X_desc217.npy'), X)
print(f'[featurize] {X.shape} | wall {cf.wall_s:.1f} s | cpu {cf.cpu_core_h * 3600:.1f} core-s | peak {cf.peak_ram_gb:.2f} GB')
mc.append_ledger(LEDGER, dict(run_id='p1_feat_desc217', track='P1', model='-', features='RDKit217', data_variant='orig',
                              objective='-', split='-', seed=mc.SEED, params='{}', n_train=len(df),
                              notes='19,945 분자 기술자 계산 (모든 분할·모델이 공유)', **cf.as_row()))

rows = []
for how in ('random', 'scaffold'):
    idx, _ = mc.load_split(how)
    tr, va, te = idx['train'], idx['validate'], idx['test']
    with mc.CostMeter('최종') as c:
        model = XGBRegressor(**mc.BASE_XGB, n_jobs=-1, random_state=mc.SEED)
        model.fit(X[tr], y[tr])
    pv = model.predict(X[va])
    t0 = time.perf_counter()
    pt = model.predict(X[te])
    infer = (time.perf_counter() - t0) / len(te) * 1000
    mv, mt = mc.metrics(y[va], pv), mc.metrics(y[te], pt)
    lo, hi = mc.boot_ci(np.abs(y[te] - pt))
    mean_mae = float(np.mean(np.abs(y[te] - y[tr].mean())))
    for name, ii, pp in (('val', va, pv), ('test', te, pt)):
        pd.DataFrame({'row': ii, 'smiles': df['smiles'].values[ii], 'y': y[ii], 'pred': pp}).to_csv(
            os.path.join(P1, 'preds', f'baseline_xgb_desc217_{how}_{name}.csv'), index=False)
    mc.append_ledger(LEDGER, dict(run_id=f'p1_baseline_{how}', track='P1', model='XGBoost', features='RDKit217',
                                  data_variant='orig', objective='MSE', split=how, seed=mc.SEED,
                                  params=json.dumps(mc.BASE_XGB), val_MAE=round(mv['MAE'], 4), val_RMSE=round(mv['RMSE'], 4),
                                  test_MAE=round(mt['MAE'], 4), test_RMSE=round(mt['RMSE'], 4), test_R2=round(mt['R2'], 4),
                                  n_train=len(tr), infer_s_per_1k=round(infer, 4),
                                  notes='기준선 재현 (train 만 학습, n_jobs=-1)', **c.as_row()))
    rows.append(dict(split=how, n_train=len(tr), n_val=len(va), n_test=len(te), val_MAE=mv['MAE'], test_MAE=mt['MAE'],
                     test_MAE_ci_lo=lo, test_MAE_ci_hi=hi, test_RMSE=mt['RMSE'], test_R2=mt['R2'],
                     mean_predictor_MAE=mean_mae, fit_wall_s=c.wall_s, fit_cpu_core_s=c.cpu_core_h * 3600,
                     peak_ram_gb=c.peak_ram_gb, infer_s_per_1k=infer))
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(P1, 'tables', 'baseline_summary.csv'), index=False)
print(tab.round(3).T.to_string())
