"""run_m2.py : M2(LLM 규칙 → RDKit 특징 F5) 비교. 이미 예측 파일이 있는 설정은 건너뛴다(이어 하기)."""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('T8_nlp'))
import numpy as np  # noqa: E402

import f5_rules  # noqa: E402
import t8eval as te  # noqa: E402

X0 = np.load('mp_mae_push/cache/X_desc217.npy').astype(np.float32)
F5 = np.load('T8_nlp/features/F5_rules.npy').astype(np.float32)
cols = f5_rules.COLS
cfg = [('REF_R217', X0, 'RDKit217', ('xgb', 'ridge')),
       ('M2_R217F5', np.hstack([X0, F5]), 'RDKit217+F5(101)', ('xgb', 'ridge')),
       ('M2_F5only', F5, 'F5(101)', ('xgb', 'ridge'))]
for g, gc in f5_rules.GROUPS.items():
    keep = [i for i, c in enumerate(cols) if c not in gc]
    cfg.append((f'M2_R217F5_minus_{g}', np.hstack([X0, F5[:, keep]]), f'RDKit217+F5-{g}({len(keep)})', ('xgb',)))
for g, gc in f5_rules.GROUPS.items():
    only = [i for i, c in enumerate(cols) if c in gc]
    cfg.append((f'M2_R217_plus_{g}', np.hstack([X0, F5[:, only]]), f'RDKit217+{g}({len(only)})', ('xgb',)))
for name, X, feat, heads in cfg:
    for head in heads:
        nm = f'{name}_{head}'
        if os.path.exists(os.path.join(te.PRED_DIR, f'{nm}_scaffold_val.csv')):
            print('skip', nm, flush=True)
            continue
        te.evaluate(nm, X, head, feat, notes='M2 H21; LLM 규칙 64개(문헌 39 + 데이터 25) 구현')
print('DONE', flush=True)
