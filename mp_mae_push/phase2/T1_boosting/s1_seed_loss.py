"""1단계: O1 seed 배깅 + H02 손실 함수 비교 (특징 F0 = RDKit 217, 기준선 설정, 600그루 고정).
이어 하기 가능: 이미 저장된 work/<run_id>.npy 는 건너뛴다."""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t1lib as L

y = L.y_all()
X = L.X0()
for how in ['random', 'scaffold']:
    idx, _ = L.split(how)
    tr, va = idx['train'], idx['validate']
    med = float(np.median(y[tr]))
    # O1: seed 0-9, 기준선 설정 그대로
    for s in range(10):
        L.fit_record(f's1_sq_s{s}_{how}', lambda s=s: L.xgb(seed=s), X[tr], y[tr], X[va], how,
                     objective='reg:squarederror', seed=s, params=dict(L.mc.BASE_XGB, n_jobs=8), notes='O1/H02 기준 손실')
    # H02: pseudo-Huber slope 선택 (train 안 5-fold CV, seed 0)
    cvm = {}
    for d in [10, 30, 60]:
        oof = L.cv_record(f's1_cv_phuber{d}_{how}', lambda ytr, d=d: L.xgb(seed=0, objective='reg:pseudohubererror', huber_slope=d,
                                                                          base_score=float(np.median(ytr))),
                          X, how, objective=f'reg:pseudohubererror(slope={d})', seed=0,
                          params=dict(L.mc.BASE_XGB, n_jobs=8, huber_slope=d, base_score='fold train median'),
                          notes='H02 slope 선택')
        cvm[d] = float(np.mean(np.abs(y[tr] - oof)))
    best = min(cvm, key=cvm.get)
    print(how, 'pseudo-Huber CV MAE', cvm, '-> slope', best, flush=True)
    with open(os.path.join(L.WORK, f's1_phuber_best_{how}.txt'), 'w') as f:
        f.write(str(best))
    losses = {
        'l1': dict(objective='reg:absoluteerror', base_score=med),
        f'phuber{best}': dict(objective='reg:pseudohubererror', huber_slope=best, base_score=med),
        'q50': dict(objective='reg:quantileerror', quantile_alpha=0.5, base_score=med),
    }
    for name, kw in losses.items():
        for s in range(3):
            L.fit_record(f's1_{name}_s{s}_{how}', lambda s=s, kw=kw: L.xgb(seed=s, **kw), X[tr], y[tr], X[va], how,
                         objective=kw['objective'] + (f'(slope={best})' if 'huber' in name else ''), seed=s,
                         params=dict(L.mc.BASE_XGB, n_jobs=8, **kw), notes='H02')
print('track cost so far', round(L.track_cost(), 4), 'core-h')
