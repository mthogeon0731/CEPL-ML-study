"""01_make_splits.py : 고정 분할 생성 (train_xgb.split, 80:10:10, seed 42) -> mp_mae_push/splits/{random,scaffold}.csv

열: row(mp.csv 행 번호, 0부터), smiles, inchikey, set(train/validate/test), pos(세트 안 순서 — train_xgb.split 이 돌려준 순서).
"""
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

import mpcommon as mc

df, mols, y = mc.load_data()
raw = pd.read_csv(os.path.join(mc.WEEK2, 'data', 'mp.csv'))
assert len(df) == len(raw) == 19945, (len(df), len(raw))
assert (df['smiles'].values == raw['smiles'].values).all(), 'load() 가 행을 걸렀다 — row 번호가 mp.csv 와 어긋남'
print(f'rows {len(df):,} | unit {set(raw["unit"])} | value {y.min():.1f}–{y.max():.1f} K | source {set(raw["source"])}')

ik = [Chem.MolToInchiKey(m) for m in mols]
print('InChIKey 빈 값', sum(1 for k in ik if not k), '| 서로 다른 InChIKey', len(set(ik)),
      '| 서로 다른 첫 블록', len({k[:14] for k in ik}))

os.makedirs(mc.SPLIT_DIR, exist_ok=True)
rows = []
for how in ('random', 'scaffold'):
    tr, va, te = mc.make_split(df['smiles'], how)
    assert len(tr) + len(va) + len(te) == len(df) == len(set(tr) | set(va) | set(te)), '세트가 겹치거나 빠진 행이 있음'
    lab = np.empty(len(df), dtype=object)
    pos = np.empty(len(df), dtype=int)
    for name, idx in (('train', tr), ('validate', va), ('test', te)):
        lab[idx] = name
        pos[idx] = np.arange(len(idx))
    out = pd.DataFrame({'row': np.arange(len(df)), 'smiles': df['smiles'].values, 'inchikey': ik, 'set': lab, 'pos': pos})
    out.to_csv(os.path.join(mc.SPLIT_DIR, f'{how}.csv'), index=False)
    if how == 'scaffold':   # 한 Murcko 뼈대가 두 세트에 걸치지 않는지
        sc = pd.Series([MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False) for s in df['smiles']])
        span = pd.DataFrame({'sc': sc, 'set': lab}).groupby('sc')['set'].nunique()
        print(f'[scaffold] 뼈대 {len(span):,}종, 두 세트 이상에 걸친 뼈대 {(span > 1).sum()}')
        acyc = (sc == '')
    for name, idx in (('train', tr), ('validate', va), ('test', te)):
        r = dict(split=how, set=name, n=len(idx), mp_median_K=float(np.median(y[idx])), mp_mean_K=float(y[idx].mean()),
                 n_gt550K=int((y[idx] > 550).sum()), n_lt250K=int((y[idx] < 250).sum()))
        if how == 'scaffold':
            r['acyclic_frac'] = float(acyc.values[idx].mean())
        rows.append(r)
    # 다시 읽어서 순서까지 같은지 확인
    idx2, _ = mc.load_split(how)
    assert list(idx2['train']) == list(tr) and list(idx2['validate']) == list(va) and list(idx2['test']) == list(te)
tab = pd.DataFrame(rows)
os.makedirs(os.path.join(mc.ROOT, 'phase1', 'tables'), exist_ok=True)
tab.to_csv(os.path.join(mc.ROOT, 'phase1', 'tables', 'split_summary.csv'), index=False)
print(tab.round(3).to_string(index=False))
