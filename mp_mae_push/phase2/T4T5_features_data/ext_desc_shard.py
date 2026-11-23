"""ext_desc_shard.py SHARD NSHARD : patent_clean.csv 행 i (i % NSHARD == SHARD) 의 RDKit 기술자 217개(train_xgb.featurize 'desc').
5,000행마다 부분 파일(npy float32)을 저장해 이어하기 가능. 출력: T5_external/cache/ext_desc_s{SHARD}_{k}.npy (+ _rows.npy)
"""
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog('rdApp.*')
WEEK2 = r'C:\Users\mthog\CEPL ML\261109_Week2_Practice'
sys.path.insert(0, WEEK2)
import train_xgb  # noqa: E402

shard, nshard = int(sys.argv[1]), int(sys.argv[2])
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(BASE, 'cache')
os.makedirs(OUT, exist_ok=True)
smi = pd.read_csv(os.path.join(BASE, 'tables', 'patent_clean.csv'))['smiles'].tolist()
rows = np.array([i for i in range(len(smi)) if i % nshard == shard])
CH = 5000
t0 = time.time()
for k in range(0, len(rows), CH):
    f = os.path.join(OUT, f'ext_desc_s{shard}_{k // CH:03d}.npy')
    if os.path.exists(f):
        continue
    rr = rows[k:k + CH]
    mols = [Chem.MolFromSmiles(smi[i]) for i in rr]
    X = train_xgb.featurize(mols, 'desc').astype(np.float32)
    np.save(f.replace('.npy', '_rows.npy'), rr)
    np.save(f, X)
    print(k + len(rr), '/', len(rows), f'{time.time() - t0:.0f}s', flush=True)
print('DONE', flush=True)
