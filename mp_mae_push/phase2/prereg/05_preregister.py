"""05_preregister.py : 2단계 실행 전에 고정하는 행 목록 (모델 예측을 보지 않음).

출력 (mp_mae_push/phase2/prereg/):
  dedup_rows_<split>.csv     : validation·test 각 행의 겹침 표지
        val  : 첫 블록(InChIKey 14자)이 train 에 있으면 overlap=1   -> 중복 제거 validation
        test : 첫 블록이 train 또는 validation 에 있으면 overlap=1 -> 중복 제거 test (random n 1,896 이 되어야 함)
  aux_overlap_counts.csv     : 보조 물성 파일과 mp validation·test 의 겹침 수 (InChIKey 전체 / 첫 블록)
  aux_exclusion_keys.csv     : 보조 모델 학습에서 지울 InChIKey 첫 블록 목록 (두 분할의 mp validation·test 합집합)
"""
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import mpcommon as mc

RDLogger.DisableLog('rdApp.*')
OUT = os.path.join(mc.ROOT, 'phase2', 'prereg')
os.makedirs(OUT, exist_ok=True)
AUX = ['logSolubility', 'logP', 'density', 'logPvap', 'bp']

blocks = {}
for how in ('random', 'scaffold'):
    _, s = mc.load_split(how)
    s['block1'] = s['inchikey'].str[:14]
    tr = set(s.loc[s.set == 'train', 'block1'])
    va = set(s.loc[s.set == 'validate', 'block1'])
    rows = []
    for part, ref in (('validate', tr), ('test', tr | va)):
        sub = s[s.set == part]
        rows.append(pd.DataFrame({'row': sub.row, 'set': part, 'inchikey': sub.inchikey,
                                  'overlap': sub.block1.isin(ref).astype(int)}))
    d = pd.concat(rows)
    d.to_csv(os.path.join(OUT, f'dedup_rows_{how}.csv'), index=False)
    for part in ('validate', 'test'):
        x = d[d.set == part]
        print(f'[{how}] {part}: n {len(x)} | overlap {int(x.overlap.sum())} | dedup n {int((x.overlap == 0).sum())}')
    blocks[how] = s

# 보조 물성 겹침
ev = {}
for how, s in blocks.items():
    for part in ('validate', 'test'):
        sub = s[s.set == part]
        ev[(how, part)] = (set(sub.inchikey), set(sub.block1))
excl = set().union(*[b for _, b in ev.values()])
cnt = []
for prop in AUX:
    df = pd.read_csv(os.path.join(mc.WEEK2, 'data', f'{prop}.csv'))
    ik = []
    for smi in df['smiles']:
        m = Chem.MolFromSmiles(str(smi))
        ik.append(Chem.MolToInchiKey(m) if m is not None else '')
    ik = pd.Series(ik)
    b1 = ik.str[:14]
    for (how, part), (full, blk) in ev.items():
        cnt.append(dict(aux=prop, n_aux=len(df), split=how, mp_set=part, n_mp=len(full),
                        mp_in_aux_full=len(full & set(ik)), mp_in_aux_block1=len(blk & set(b1))))
    cnt.append(dict(aux=prop, n_aux=len(df), split='both', mp_set='validate+test (제외 대상)', n_mp=len(excl),
                    mp_in_aux_full=np.nan, mp_in_aux_block1=int(b1.isin(excl).sum())))
    print(f'{prop}: n {len(df):,} | 제외할 보조 행(첫 블록이 mp val/test 에 있음) {int(b1.isin(excl).sum()):,}')
c = pd.DataFrame(cnt)
c.to_csv(os.path.join(OUT, 'aux_overlap_counts.csv'), index=False)
pd.DataFrame({'block1': sorted(excl)}).to_csv(os.path.join(OUT, 'aux_exclusion_keys.csv'), index=False)
print(c[c.mp_set == 'test'].to_string(index=False))
