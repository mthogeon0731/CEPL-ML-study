"""run_phys_shard.py SHARD NSHARD NCONF : mp.csv 행 i (i % NSHARD == SHARD) 의 F3 특징 계산, 500행마다 중간 저장(이어하기 가능)."""
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import RDLogger  # noqa: E402
import phys_feats as pf  # noqa: E402

RDLogger.DisableLog('rdApp.*')
shard, nshard, nconf = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
WEEK2 = r'C:\Users\mthog\CEPL ML\261109_Week2_Practice'
out_dir = os.path.join(HERE, '..', 'cache_shards')
os.makedirs(out_dir, exist_ok=True)
out = os.path.join(out_dir, f'phys_shard{shard}of{nshard}.csv')
smiles = pd.read_csv(os.path.join(WEEK2, 'data', 'mp.csv'))['smiles'].tolist()
rows = [i for i in range(len(smiles)) if i % nshard == shard]
done = set()
if os.path.exists(out):
    done = set(pd.read_csv(out, usecols=['row'])['row'])
todo = [i for i in rows if i not in done]
print(f'shard {shard}/{nshard}: {len(rows)} rows, {len(todo)} to do', flush=True)
buf, t0 = [], time.time()
for k, i in enumerate(todo):
    try:
        d = pf.featurize_smiles(smiles[i], n_conf=nconf)
    except Exception as e:  # 실패 행은 NaN
        d = {c: np.nan for c in pf.COLS}
        print('ERR', i, repr(e), flush=True)
    d['row'] = i
    buf.append(d)
    if len(buf) >= 500 or k == len(todo) - 1:
        pd.DataFrame(buf)[['row'] + pf.COLS].to_csv(out, mode='a', header=not os.path.exists(out), index=False)
        buf = []
        print(f'{k + 1}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
print('DONE', flush=True)
