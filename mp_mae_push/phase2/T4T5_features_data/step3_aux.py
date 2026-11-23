"""step3_aux.py : H17 연쇄 특징(누설 차단). PREREG 3절 규칙.

보조 물성 P 마다:
  1) 보조 데이터에서 InChIKey 첫 블록이 두 분할의 mp validation·test(aux_exclusion_keys.csv)에 있는 행 삭제.
  2) 전체 보조 모델(XGBoost 기준선 설정, seed 42) + 첫 블록 그룹 5-fold 모델(GroupKFold, 같은 첫 블록은 같은 fold).
  3) mp 행 특징: 그 행의 첫 블록이 (삭제 후) 보조 데이터에 있으면 그 블록을 뺀 fold 모델의 예측, 없으면 전체 모델 예측.
     validation·test 행은 1)에서 보조 데이터에서 빠졌으므로 항상 전체 모델 예측. 측정값은 쓰지 않는다.
출력: T4_features/cache/X_aux5.npy (19,945 x 5, float32), aux_cols.csv, tables/aux_exclusion_recount.csv, tables/aux_model_cv.csv
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('T4_features/scripts'))
import fitlib as fl  # noqa: E402
mc = fl.mc
PROPS = ['logSolubility', 'logP', 'density', 'logPvap', 'bp']
LEDGER = 'T4_features/ledger.csv'


def build(y, S):
    keys = set(pd.read_csv('mp_mae_push/phase2/prereg/aux_exclusion_keys.csv')['block1'])
    sr = mc.load_split('random')[1].sort_values('row')
    ss = mc.load_split('scaffold')[1].sort_values('row')
    ik = sr['inchikey'].to_numpy()
    b1 = np.array([k[:14] for k in ik])
    recount, cvrows = [], []
    F = np.zeros((len(ik), len(PROPS)), np.float32)
    src = np.zeros((len(ik), len(PROPS)), np.int8)  # 1 = fold(out-of-fold) 예측, 0 = 전체 모델 예측
    X0 = np.load('mp_mae_push/cache/X_desc217.npy').astype(np.float32)
    for j, P in enumerate(PROPS):
        meta = pd.read_csv(f'T4_features/cache_aux/{P}_meta.csv')
        XA = np.load(f'T4_features/cache_aux/{P}_X.npy')
        ab1 = meta['inchikey'].fillna('').str[:14].to_numpy()
        aik = set(meta['inchikey'].fillna(''))
        ab1s = set(ab1)
        for spn, s in (('random', sr), ('scaffold', ss)):
            for st in ('validate', 'test'):
                m = s['set'].to_numpy() == st
                recount.append(dict(aux=P, n_aux=len(meta), split=spn, mp_set=st, n_mp=int(m.sum()),
                                    mp_in_aux_full=int(sum(k in aik for k in ik[m])),
                                    mp_in_aux_block1=int(sum(k in ab1s for k in b1[m]))))
        excl = np.isin(ab1, list(keys))
        recount.append(dict(aux=P, n_aux=len(meta), split='both', mp_set='validate+test 첫 블록 (보조 행 삭제 수)',
                            n_mp=len(keys), mp_in_aux_full=np.nan, mp_in_aux_block1=int(excl.sum())))
        keep = ~excl
        XA, ya, ga = XA[keep], meta['value'].to_numpy(float)[keep], ab1[keep]
        with mc.CostMeter('특징') as c:
            full = fl.xgb(42).fit(XA, ya)
            pred_full = full.predict(X0).astype(np.float32)
            F[:, j] = pred_full
            gkf = GroupKFold(n_splits=5)
            aux_oof = np.zeros(len(ya))
            blk2fold = {}
            fold_models = []
            for k, (tr, te) in enumerate(gkf.split(XA, ya, ga)):
                mdl = fl.xgb(42).fit(XA[tr], ya[tr])
                fold_models.append(mdl)
                aux_oof[te] = mdl.predict(XA[te])
                for g in set(ga[te]):
                    blk2fold[g] = k
            in_aux = np.array([g in blk2fold for g in b1])
            for k, mdl in enumerate(fold_models):
                rows = np.where(in_aux & np.array([blk2fold.get(g, -1) == k for g in b1]))[0]
                if len(rows):
                    F[rows, j] = mdl.predict(X0[rows])
                    src[rows, j] = 1
        r2 = 1 - np.sum((ya - aux_oof) ** 2) / np.sum((ya - ya.mean()) ** 2)
        # 안전 점검: 첫 블록이 validation·test 인 mp 행은 fold 예측을 쓰면 안 됨(보조 데이터에서 지워졌으므로)
        vt_rows = np.isin(b1, list(keys))
        assert src[vt_rows, j].sum() == 0
        cvrows.append(dict(aux=P, n_aux_after_excl=int(keep.sum()), n_excluded=int(excl.sum()),
                           aux_cv_MAE=float(np.mean(np.abs(ya - aux_oof))), aux_cv_R2=float(r2),
                           mp_rows_oof_feature=int(src[:, j].sum()), mp_rows_full_model=int((src[:, j] == 0).sum())))
        mc.append_ledger(LEDGER, dict(run_id=f'T4_aux_{P}', track='T4 특징', model='XGBoost 보조 모델(전체 + 첫 블록 그룹 5-fold)',
                                      features='desc217', data_variant=f'{P} − mp val·test 첫 블록 {int(excl.sum())}행',
                                      objective='reg:squarederror', split='both', seed=42, params='BASE_XGB, n_jobs=8, 6 fits',
                                      n_train=int(keep.sum()), notes=f'보조 CV MAE {cvrows[-1]["aux_cv_MAE"]:.4f}, R2 {r2:.3f}',
                                      **c.as_row()))
        print(P, cvrows[-1], flush=True)
    np.save('T4_features/cache/X_aux5.npy', F)
    np.save('T4_features/cache/X_aux5_src.npy', src)
    pd.DataFrame({'col': [f'aux_pred_{p}' for p in PROPS]}).to_csv('T4_features/cache/aux_cols.csv', index=False)
    pd.DataFrame(recount).to_csv('T4_features/tables/aux_exclusion_recount.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(cvrows).to_csv('T4_features/tables/aux_model_cv.csv', index=False, encoding='utf-8-sig')
    return F, src
