"""rescan_width.py : 통과 기록(patent_records_kept.csv 의 rec_id)의 보고 범위 폭(°C = K)을 SDF에서 다시 읽는다.
단계 5의 '고품질' 대체 부분집합(보고 범위 폭이 좁은 기록) 정의용. 출력: tables/patent_kept_width.csv (rec_id, width_K; 단일 값은 NaN)
"""
import os
import re

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
kept = set(pd.read_csv(os.path.join(BASE, 'tables', 'patent_records_kept.csv'), usecols=['rec_id'])['rec_id'])
RE_RANGE = re.compile(r'\s*(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)\s*')
out = []
sup = Chem.ForwardSDMolSupplier(open(os.path.join(BASE, 'raw', 'melting_point.sdf'), 'rb'), sanitize=False, removeHs=False)
for rec_id, m in enumerate(sup):
    if rec_id not in kept or m is None:
        continue
    v = str(m.GetProp('Value'))
    mr = RE_RANGE.fullmatch(v)
    out.append((rec_id, abs(float(mr.group(2)) - float(mr.group(1))) if mr else np.nan))
pd.DataFrame(out, columns=['rec_id', 'width_K']).to_csv(os.path.join(BASE, 'tables', 'patent_kept_width.csv'), index=False)
print('DONE', len(out), flush=True)
