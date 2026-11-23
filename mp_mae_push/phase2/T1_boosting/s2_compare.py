"""2단계 비교: F0 / F0+O3 / F0+F1 / F0+F1+O3 (손실 reg:squarederror, 기준선 설정, 600그루).
colsample_bytree 0.5 vs 0.3 은 train 안 5-fold CV(seed 0)로 묶음·분할마다 선택 → seed 0-2 로 train 전체 적합."""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t1lib as L

C = os.path.join(L.WS, 'mp_mae_push', 'cache')
y = L.y_all()
F0 = L.X0()
O3 = np.load(os.path.join(C, 'X_stereo_o3.npy')).astype(np.float32)
F1 = np.load(os.path.join(C, 'X_morgan_count_r2_2048.npy')).astype(np.float32)
SETS = {'F0': [F0], 'F0+O3': [F0, O3], 'F0+F1': [F0, F1], 'F0+F1+O3': [F0, F1, O3]}
for how in ['random', 'scaffold']:
    idx, _ = L.split(how)
    tr, va = idx['train'], idx['validate']
    for fs, parts in SETS.items():
        X = np.hstack(parts)
        tag = fs.replace('+', '')
        cvm = {}
        for cs in [0.5, 0.3]:
            oof = L.cv_record(f's2_cv_{tag}_cs{int(cs*10)}_{how}', lambda ytr, cs=cs: L.xgb(seed=0, colsample_bytree=cs), X, how,
                              features=fs, seed=0, params=dict(L.mc.BASE_XGB, n_jobs=8, colsample_bytree=cs), notes='colsample 선택 CV')
            cvm[cs] = float(np.mean(np.abs(y[tr] - oof)))
        cs = min(cvm, key=cvm.get)
        print(how, fs, 'CV', cvm, '-> colsample', cs, flush=True)
        with open(os.path.join(L.WORK, f's2_cs_{tag}_{how}.txt'), 'w') as f:
            f.write(str(cs))
        for s in range(3):
            L.fit_record(f's2_{tag}_s{s}_{how}', lambda s=s, cs=cs: L.xgb(seed=s, colsample_bytree=cs), X[tr], y[tr], X[va], how,
                         features=fs, seed=s, params=dict(L.mc.BASE_XGB, n_jobs=8, colsample_bytree=cs), stage='탐색',
                         notes=f'H03/O3 비교, colsample {cs} (train CV 선택)')
        del X
print('track cost so far', round(L.track_cost(), 4), 'core-h')
