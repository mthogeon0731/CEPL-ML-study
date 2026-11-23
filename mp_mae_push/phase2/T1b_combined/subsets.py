"""부분집합 판정: validation(seed 평균 예측) + train 5-fold OOF(seed 0) 를 합쳐 (n ≈ 17,950)
고대칭(sym2d_high), y > 550 K, y < 250 K 와 MP 구간별 MAE·평균 잔차(pred − y), 기준 모델 대비 짝지은 부트스트랩 95 % CI."""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import t1blib as L

BANDS = [(-np.inf, 250), (250, 350), (350, 450), (450, 550), (550, np.inf)]


def pooled(how, val_pred, oof_pred):
    idx, _ = L.split(how)
    rows = np.concatenate([idx['validate'], idx['train']])
    src = np.array(['val'] * len(idx['validate']) + ['oof'] * len(idx['train']))
    return rows, src, np.concatenate([val_pred, oof_pred])


def subset_masks(rows):
    y = L.y_all()[rows]
    fl = pd.read_csv(os.path.join(L.CACHE, 'mol_flags.csv')).set_index('row')
    sym = fl.loc[rows, 'sym2d_high'].astype(bool).to_numpy()
    out = {'all': np.ones(len(rows), bool), 'sym2d_high': sym, 'y>550': y > 550, 'y<250': y < 250}
    for lo, hi in BANDS:
        out[f'band[{lo},{hi})'] = (y >= lo) & (y < hi)
    return out


def compare(how, name, val_new, oof_new, val_ref, oof_ref, ref_name, which=('pooled', 'val', 'oof')):
    rows, src, pn = pooled(how, val_new, oof_new)
    _, _, pr = pooled(how, val_ref, oof_ref)
    y = L.y_all()[rows]
    res = []
    for sname, m in subset_masks(rows).items():
        for w in which:
            mm = m & ((src == w) if w != 'pooled' else True)
            if mm.sum() == 0:
                continue
            d, lo, hi = L.mc.paired_bootstrap(y[mm], pn[mm], pr[mm])
            res.append(dict(split=how, model=name, ref=ref_name, subset=sname, source=w, n=int(mm.sum()),
                            MAE=float(np.mean(np.abs(pn[mm] - y[mm]))), MAE_ref=float(np.mean(np.abs(pr[mm] - y[mm]))),
                            bias=float(np.mean(pn[mm] - y[mm])), bias_ref=float(np.mean(pr[mm] - y[mm])),
                            dMAE=d, ci_lo=lo, ci_hi=hi,
                            dbias=float(np.mean(pn[mm] - pr[mm])),
                            dbias_lo=L.mc.boot_ci(pn[mm] - pr[mm])[0], dbias_hi=L.mc.boot_ci(pn[mm] - pr[mm])[1]))
    return pd.DataFrame(res)
