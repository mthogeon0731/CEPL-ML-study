"""aux_desc.py : 보조 물성 5종의 RDKit 기술자 217개(train_xgb.featurize 'desc', mp-push RDKit) + InChIKey.
출력: T4_features/cache_aux/<prop>_X.npy (float32), <prop>_meta.csv (aux_row, smiles, inchikey, value)
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

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache_aux')
os.makedirs(OUT, exist_ok=True)
for prop in sys.argv[1:]:
    if os.path.exists(os.path.join(OUT, f'{prop}_X.npy')):
        continue
    t0 = time.time()
    df = pd.read_csv(os.path.join(WEEK2, 'data', f'{prop}.csv'))
    mols = [Chem.MolFromSmiles(s) for s in df['smiles']]
    keep = [i for i, m in enumerate(mols) if m is not None and m.GetNumAtoms() > 0]
    mols = [mols[i] for i in keep]
    X = train_xgb.featurize(mols, 'desc').astype(np.float32)
    meta = pd.DataFrame({'aux_row': keep, 'smiles': df['smiles'].values[keep],
                         'inchikey': [Chem.MolToInchiKey(m) or '' for m in mols], 'value': df['value'].values[keep]})
    np.save(os.path.join(OUT, f'{prop}_X.npy'), X)
    meta.to_csv(os.path.join(OUT, f'{prop}_meta.csv'), index=False)
    print(prop, len(df), len(keep), f'{time.time() - t0:.0f}s', flush=True)
print('DONE', flush=True)
