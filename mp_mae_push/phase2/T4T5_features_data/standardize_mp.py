"""standardize_mp.py : O4 표준화. mp.csv SMILES → MolStandardize (Cleanup=정규화, 가장 큰 조각, 전하 중화, 호변 이성질체 정규형).
표준화 뒤 canonical SMILES 가 원래 canonical SMILES 와 다른 행만 기술자 217개를 다시 계산하고, 같은 행은 X_desc217 캐시를 그대로 쓴다.
출력: T4_features/cache/X_desc217_std.npy (float32), std_smiles.csv (row, smiles, smiles_std, changed, inchikey_std)
"""
import os
import sys
import time

sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem.MolStandardize import rdMolStandardize as S  # noqa: E402

RDLogger.DisableLog('rdApp.*')
WEEK2 = r'C:\Users\mthog\CEPL ML\261109_Week2_Practice'
sys.path.insert(0, WEEK2)
import train_xgb  # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(BASE, 'cache')
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
smiles = pd.read_csv(os.path.join(WEEK2, 'data', 'mp.csv'))['smiles'].tolist()
X0 = np.load(os.path.join(BASE, '..', 'mp_mae_push', 'cache', 'X_desc217.npy')).astype(np.float32)
lfc, unch = S.LargestFragmentChooser(), S.Uncharger()
te = S.TautomerEnumerator()
te.SetMaxTautomers(200)
te.SetMaxTransforms(200)
rows, std_mols, changed_idx = [], [], []
for i, s in enumerate(smiles):
    m = Chem.MolFromSmiles(s)
    can0 = Chem.MolToSmiles(m)
    try:
        x = S.Cleanup(m)
        x = lfc.choose(x)
        x = unch.uncharge(x)
        x = te.Canonicalize(x)
        Chem.SanitizeMol(x)
        can1 = Chem.MolToSmiles(x)
    except Exception as e:  # 실패 시 원본 유지
        print('ERR', i, repr(e)[:80], flush=True)
        x, can1 = m, can0
    ch = can1 != can0
    rows.append((i, s, can1, int(ch), Chem.MolToInchiKey(x) or ''))
    if ch:
        changed_idx.append(i)
        std_mols.append(x)
    if (i + 1) % 2000 == 0:
        print(i + 1, len(changed_idx), f'{time.time() - t0:.0f}s', flush=True)
X = X0.copy()
if std_mols:
    X[changed_idx] = train_xgb.featurize(std_mols, 'desc').astype(np.float32)
np.save(os.path.join(OUT, 'X_desc217_std.npy'), X)
pd.DataFrame(rows, columns=['row', 'smiles', 'smiles_std', 'changed', 'inchikey_std']).to_csv(
    os.path.join(OUT, 'std_smiles.csv'), index=False)
print('DONE changed', len(changed_idx), f'{time.time() - t0:.0f}s', flush=True)
