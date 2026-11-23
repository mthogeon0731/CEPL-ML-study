"""step3_models.py : H17 연쇄 특징 비교, O4 표준화, H09(R1–R3) 라벨 정제·가중치 — validation 비교(seed 3, 두 분할).
python 커널에서 import 해 run_h17(y, X0, S), run_o4(y, S), run_h09(y, X0, S) 를 차례로 호출. 적합마다 예측 npy + ledger 한 줄(이어하기 가능).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('T4_features/scripts'))
import fitlib as fl  # noqa: E402
mc = fl.mc
RAW = 'T4_features/preds/raw'
LEDGER = 'T4_features/ledger.csv'
PROPS = ['logSolubility', 'logP', 'density', 'logPvap', 'bp']


def _fit_all(name, X, y, S, desc, train_fn=None, notes=''):
    """train_fn(sp, tr_rows) -> (rows, y_tr, w) ; None 이면 원래 train."""
    for sp in fl.SPLITS:
        tr = S[sp]['idx']['train']
        rows, ytr, w = (tr, y[tr], None) if train_fn is None else train_fn(sp, tr)
        for seed in fl.SEEDS:
            f = f'{RAW}/val_{name}_{sp}_s{seed}.npy'
            if os.path.exists(f):
                continue
            with mc.CostMeter('특징') as c:
                p = fl.fit_predict(X[rows], ytr, X[S[sp]['va']], seed, w=w)
            np.save(f, p)
            yv = y[S[sp]['va']]
            mc.append_ledger(LEDGER, dict(run_id=f'T4_val_{name}_{sp}_s{seed}', track='T4 특징', model='XGBoost(기준선 설정)',
                                          features=desc, data_variant=name, objective='reg:squarederror', split=sp, seed=seed,
                                          params='BASE_XGB, n_jobs=8' + (', sample_weight' if w is not None else ''),
                                          val_MAE=round(float(np.mean(np.abs(yv - p))), 4), val_RMSE=round(mc.metrics(yv, p)['RMSE'], 4),
                                          n_train=len(rows), notes=notes, **c.as_row()))
            print(name, sp, seed, round(float(np.mean(np.abs(yv - p))), 3), flush=True)


def run_h17(y, X0, S):
    FA = np.load('T4_features/cache/X_aux5.npy')
    j = {p: i for i, p in enumerate(PROPS)}
    sets = {'F0+aux4(no logS)': ['logP', 'density', 'logPvap', 'bp'], 'F0+logS': ['logSolubility'], 'F0+aux5': PROPS}
    for name, ps in sets.items():
        X = np.hstack([X0, FA[:, [j[p] for p in ps]]])
        _fit_all(name, X, y, S, f'desc217 + 보조 예측 {len(ps)}열', notes='PREREG 3절: 보조 모델은 mp val·test 첫 블록 삭제 데이터, train 행은 out-of-fold')


def run_o4(y, S):
    Xs = np.load('T4_features/cache/X_desc217_std.npy').astype(np.float32)
    std = pd.read_csv('T4_features/cache/std_smiles.csv')
    _fit_all('O4 std', Xs, y, S, 'desc217 (표준화 SMILES)', notes=f'표준화로 바뀐 행 {int(std.changed.sum())}')
    ik = std['inchikey_std'].to_numpy()

    def merge(sp, tr):
        d = pd.DataFrame({'row': tr, 'ik': ik[tr], 'y': y[tr]})
        g = d.groupby('ik', sort=False).agg(row=('row', 'first'), y=('y', 'mean'))
        return g['row'].to_numpy(), g['y'].to_numpy(), None
    _fit_all('O4 std+merge', Xs, y, S, 'desc217 (표준화 SMILES)', train_fn=merge,
             notes='train 안 표준화 InChIKey 같은 행 병합(라벨 평균, 첫 행 특징)')


def run_h09(y, X0, S):
    flags = pd.read_csv('mp_mae_push/phase1/tables/mol_flags.csv').sort_values('row')
    rep = pd.read_csv('mp_mae_push/phase1/tables/mp_replicate_stats.csv').sort_values('row')
    rng = rep['range_K'].fillna(0).to_numpy()
    multi = flags['multi_fragment'].to_numpy(bool)
    r3 = (flags['acyclic'].to_numpy(bool) & (flags['hbd'].to_numpy() == 0) & ~flags['charged_any'].to_numpy(bool)
          & (flags['heavy'].to_numpy() <= 15) & (y >= 400))

    def excl(mask):
        def f(sp, tr):
            k = tr[~mask[tr]]
            return k, y[k], None
        return f

    def wfn(sp, tr):
        return tr, y[tr], 1.0 / (1.0 + rng[tr] / 10.0)
    _fit_all('R1 multi-frag 제외', X0, y, S, 'desc217', train_fn=excl(multi), notes=f'전체 {int(multi.sum())}행 해당')
    _fit_all('R2a 범위50K이상 제외', X0, y, S, 'desc217', train_fn=excl(rng >= 50), notes=f'전체 {int((rng >= 50).sum())}행 해당')
    _fit_all('R2b 범위 가중치', X0, y, S, 'desc217', train_fn=wfn, notes='w = 1/(1+range/10 K)')
    _fit_all('R3 의심 라벨 제외(탐색적)', X0, y, S, 'desc217', train_fn=excl(r3),
             notes=f'전체 {int(r3.sum())}행 해당; 1단계 OOF를 보고 만든 규칙 → 탐색적')
    return dict(multi=int(multi.sum()), r2=int((rng >= 50).sum()), r3=int(r3.sum()))
