"""s3_recal.py : H11 재보정 확인. 대상 = 최선 단일(lgb_r1_F0F3F5) 과 최선 앙상블(정직한 교차 적합 예측).
재보정: (a) 선형 y ~ a + b·pred (OLS)  (b) isotonic (단조 증가, 범위 밖은 끝값).
평가: validation 5-fold × 10회 교차 적합(fold seed 0–9; 재보정기는 학습 fold 로만 적합) → 반복 평균 예측으로 MAE 와 짝지은 부트스트랩.
최선 단일은 train OOF 로 재보정기를 적합해 validation 에 적용한 결과도 낸다(앙상블은 train OOF 가 없는 멤버가 있어 해당 없음).
MP 구간(라벨 y 기준; 1단계 구간 <250, 250–350, 350–450, 450–550, >550 K) 별 MAE 득실.
사용: main(best_ens_file) — best_ens_file 은 preds/ens/ 의 정직한 예측 CSV 이름(분할별 dict).
"""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enslib as E
import numpy as np, pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

BINS = [-np.inf, 250, 350, 450, 550, np.inf]
BIN_LAB = ['<250', '250-350', '350-450', '450-550', '>550']


def fit_cal(kind, p, y):
    if kind == 'linear':
        b, a = np.polyfit(p, y, 1)
        return lambda q: a + b * q
    iso = IsotonicRegression(out_of_bounds='clip').fit(p, y)
    return iso.predict


def crossfit_cal(kind, p, y, n_rep=10):
    H = np.zeros((n_rep, len(y)))
    for r in range(n_rep):
        for i, j in KFold(5, shuffle=True, random_state=r).split(p):
            H[r, j] = fit_cal(kind, p[i], y[i])(p[j])
    return H


def main(best_ens_files):
    rows, brows = [], []
    reg = pd.read_csv(os.path.join(E.ENS, E.MEMBERS_FILE))
    for how in E.SPLITS:
        with E.mc.CostMeter('탐색') as cm:
            idx, _ = E.split(how)
            y = E.y_all()[idx['validate']]
            dm = E.dedup_mask(how)
            r0 = reg[(reg.split == how) & (reg.member == 'lgb_r1_F0F3F5')]
            pv, _ = E.load_members(how, 'val', {'best': r0[r0.kind == 'val'].path.iloc[0]})
            po, yo = E.load_members(how, 'OOF', {'best': r0[r0.kind == 'OOF'].path.iloc[0]})
            ens = pd.read_csv(os.path.join(E.ENS, 'preds', 'ens', best_ens_files[how]))
            assert np.array_equal(ens['row'].to_numpy(), idx['validate'])
            targets = {'best_single(lgb_r1_F0F3F5)': pv['best'], f'best_ens({best_ens_files[how].split("__")[1]}|{best_ens_files[how].split("__")[2].split("_val")[0]})': ens['pred'].to_numpy()}
            bins = pd.cut(y, BINS, labels=BIN_LAB)
            for tname, p in targets.items():
                variants = {}
                for kind in ('linear', 'isotonic'):
                    variants[f'{kind}_valCV'] = crossfit_cal(kind, p, y).mean(0)
                    if tname.startswith('best_single'):
                        variants[f'{kind}_trainOOF'] = fit_cal(kind, po['best'], yo)(p)
                for vname, q in variants.items():
                    t, td = E.pboot(y, q, p), E.pboot(y[dm], q[dm], p[dm])
                    rows.append(dict(split=how, target=tname, recal=vname, MAE_before=E.mae(y, p), MAE_after=E.mae(y, q),
                                     dMAE=E.fmt_ci(t), dMAE_num=t[0], sig_better=int(t[2] < 0), sig_worse=int(t[1] > 0),
                                     MAE_dedup_before=E.mae(y[dm], p[dm]), MAE_dedup_after=E.mae(y[dm], q[dm]), dMAE_dedup=E.fmt_ci(td),
                                     bias_before=float(np.mean(p - y)), bias_after=float(np.mean(q - y)), n=len(y), n_dedup=int(dm.sum())))
                    for b in BIN_LAB:
                        m = np.asarray(bins == b)
                        tb = E.pboot(y[m], q[m], p[m])
                        brows.append(dict(split=how, target=tname, recal=vname, mp_bin=b, n=int(m.sum()), MAE_before=E.mae(y[m], p[m]),
                                          MAE_after=E.mae(y[m], q[m]), dMAE=E.fmt_ci(tb), bias_before=float(np.mean(p[m] - y[m])),
                                          bias_after=float(np.mean(q[m] - y[m]))))
        E.ledger(dict(run_id=f'H11_recal_{how}', model='재보정(선형·isotonic)', features='후보 예측 1열', data_variant='validation 5-fold×10 교차 적합 + train OOF',
                      objective='MSE(OLS)/isotonic', split=how, seed='fold seed 0-9', params=dict(targets=list(targets)), n_train=len(y), **cm.as_row()))
    pd.DataFrame(rows).to_csv(os.path.join(E.ENS, 'tables', 'recal_results.csv'), index=False)
    pd.DataFrame(brows).to_csv(os.path.join(E.ENS, 'tables', 'recal_by_mpbin.csv'), index=False)
    return pd.DataFrame(rows), pd.DataFrame(brows)
