"""run_reps.py <name> <features_label> <npy>[+<npy>...] [heads=xgb,ridge] [curve=1]

표현 하나를 두 head × 두 분할로 평가하고, 학습 표본 수 곡선(1,000 / 4,000 / 전체)을 만든다. 이미 있는 결과는 건너뛴다.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('T8_nlp'))
import numpy as np  # noqa: E402

import t8eval as te  # noqa: E402

name, feat, paths = sys.argv[1:4]
heads = sys.argv[4].split(',') if len(sys.argv) > 4 else ['xgb', 'ridge']
curve = (sys.argv[5] == '1') if len(sys.argv) > 5 else True
X = np.hstack([np.load(p).astype(np.float32) for p in paths.split('+')])
print(name, X.shape, flush=True)
for head in heads:
    nm = f'{name}_{head}'
    if not os.path.exists(os.path.join(te.PRED_DIR, f'{nm}_scaffold_val.csv')):
        te.evaluate(nm, X, head, feat, notes='T8 표현 비교')
    if curve:
        for n in (1000, 4000):
            if not os.path.exists(os.path.join(te.PRED_DIR, 'seeds', f'{nm}_scaffold_n{n}_val_seeds.csv')):
                te.evaluate(nm, X, head, feat, n_train=n, notes='T8 표본 수 곡선')
print('DONE', name, flush=True)
