"""t8report.py : 저장된 validation 예측 CSV(row,y,pred)만 다시 읽어 짝지은 비교표를 만든다."""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('T8_nlp'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import t8eval as te  # noqa: E402

mc = te.mc
FLAGS = pd.read_csv('mp_mae_push/phase1/tables/mol_flags.csv').set_index('row')


def load(name, how):
    return pd.read_csv(os.path.join(te.PRED_DIR, f'{name}_{how}_val.csv')).set_index('row')


def compare(new, ref, how, subset=None):
    """ΔMAE = MAE(new) − MAE(ref), 짝지은 부트스트랩 2,000회. subset: None | 'dedup' | 'sym2d_high' | 'gt550'."""
    a, b = load(new, how), load(ref, how)
    rows = a.index.to_numpy()
    b = b.loc[rows]
    m = np.ones(len(rows), bool)
    if subset == 'dedup':
        m = te.dedup_mask(how, rows)
    elif subset == 'sym2d_high':
        m = FLAGS.loc[rows, 'sym2d_high'].to_numpy(bool)
    elif subset == 'gt550':
        m = a['y'].to_numpy() > 550
    y = a['y'].to_numpy()[m]
    pn, pr = a['pred'].to_numpy()[m], b['pred'].to_numpy()[m]
    d, lo, hi = mc.paired_bootstrap(y, pn, pr)
    return dict(new=new, ref=ref, split=how, subset=subset or 'all', n=int(m.sum()),
                MAE_new=float(np.mean(np.abs(y - pn))), MAE_ref=float(np.mean(np.abs(y - pr))),
                dMAE=d, lo=lo, hi=hi, sig=('개선' if hi < 0 else ('악화' if lo > 0 else '차이 없음(CI 0 포함)')))


def baseline_name(how):
    """1단계 기준선 예측(n_jobs=8, seed 42 단일)을 같은 형식으로 T8 preds 에 둔다."""
    nm = 'BASELINE_phase1'
    p = os.path.join(te.PRED_DIR, f'{nm}_{how}_val.csv')
    if not os.path.exists(p):
        b = pd.read_csv(f'mp_mae_push/phase1/preds/baseline_xgb_desc217_{how}_val.csv')
        b[['row', 'y', 'pred']].to_csv(p, index=False)
    return nm
