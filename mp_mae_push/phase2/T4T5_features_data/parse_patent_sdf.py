"""parse_patent_sdf.py : Tetko 2016 특허 melting_point.sdf 를 RULES_applied.md 2절(기록 단위) 규칙으로 정리.
출력: tables/patent_records_kept.csv (기록 단위 통과분), tables/patent_record_rule_counts.csv, tables/patent_field_counts.csv
"""
import os
import re
import sys
import time
from collections import Counter

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, '..')
SDF = os.path.join(BASE, 'raw', 'melting_point.sdf')

RE_DEC = re.compile(r'\bdec(\.|\b|omp)|zers|pyroly', re.I)
RE_INEQ = re.compile(r'[<>≥≤~]|ca\.', re.I)
RE_NUM = re.compile(r'\s*(-?\d+(?:\.\d+)?)\s*')
RE_RANGE = re.compile(r'\s*(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)\s*')
RE_CEL = re.compile(r'(°|º|˚|deg\.?|degrees?)\s*C\b|centigrade|celsius', re.I)

counts = Counter()
fields = Counter()
diag = Counter()
kept = []
t0 = time.time()
sup = Chem.ForwardSDMolSupplier(open(SDF, 'rb'), sanitize=False, removeHs=False)
for rec_id, m in enumerate(sup):
    counts['n_records'] += 1
    if m is None:
        counts['r0_unreadable_record'] += 1
        continue
    p = m.GetPropsAsDict()
    fields.update(p.keys())
    diag['SuspiciousValue=' + str(p.get('SuspiciousValue'))] += 1
    diag['FromLiterature=' + str(p.get('FromLiterature'))] += 1
    if p.get('QuantityType') != 'MeltingPoint' or 'Value' not in p or 'SMILES' not in p:
        counts['r0_field_assumption'] += 1
        continue
    val, txt = str(p['Value']), str(p.get('OriginalText', ''))
    if RE_DEC.search(txt) or RE_DEC.search(val):
        counts['r1_decomposition'] += 1
        continue
    if RE_INEQ.search(val) or RE_INEQ.search(txt):
        counts['r2_inequality_approx'] += 1
        continue
    mr, mn = RE_RANGE.fullmatch(val), RE_NUM.fullmatch(val)
    if mr:
        a, b = float(mr.group(1)), float(mr.group(2))
        if abs(b - a) > 5:
            counts['r4_range_gt5K'] += 1
            continue
        v = (a + b) / 2
    elif mn:
        v = float(mn.group(1))
    else:
        counts['r3_value_format'] += 1
        continue
    if not RE_CEL.search(txt):
        if re.search(r'(°|º|deg)\s*F\b', txt):
            counts['r5_unit_F'] += 1
        elif re.search(r'°|º|˚', txt):
            counts['r5_unit_bare_degree'] += 1
        else:
            counts['r5_unit_none'] += 1
        continue
    vk = v + 273.15
    if not (50 <= vk <= 800):
        counts['r6_outside_50_800K'] += 1
        continue
    smi = str(p['SMILES'])
    if '.' in smi:
        counts['r7_multi_fragment'] += 1
        continue
    mol = Chem.MolFromSmiles(smi)
    if mol is None or mol.GetNumAtoms() == 0:
        counts['r7_rdkit_unreadable'] += 1
        continue
    if mol.GetNumHeavyAtoms() > 100:
        counts['r7_heavy_gt100'] += 1
        continue
    can = Chem.MolToSmiles(mol)
    if '.' in can:
        counts['r7_multi_fragment'] += 1
        continue
    ik = Chem.MolToInchiKey(mol) or ''
    if not ik:
        counts['r7_no_inchikey'] += 1
        continue
    diag['inchikey_equals_StdInChIKey'] += int(ik == str(p.get('StdInChIKey', '')))
    kept.append((rec_id, can, ik, vk, str(p.get('Patent', '')), str(p.get('SuspiciousValue')), int(bool(mr))))
    counts['kept'] += 1
    if counts['n_records'] % 20000 == 0:
        print(counts['n_records'], len(kept), f'{time.time() - t0:.0f}s', flush=True)

os.makedirs(os.path.join(BASE, 'tables'), exist_ok=True)
pd.DataFrame(kept, columns=['rec_id', 'smiles', 'inchikey', 'value_K', 'patent', 'suspicious', 'was_range']).to_csv(
    os.path.join(BASE, 'tables', 'patent_records_kept.csv'), index=False)
pd.DataFrame(sorted(counts.items()), columns=['rule', 'n']).to_csv(
    os.path.join(BASE, 'tables', 'patent_record_rule_counts.csv'), index=False)
pd.DataFrame(sorted({**{'field:' + k: v for k, v in fields.items()}, **diag}.items()), columns=['item', 'n']).to_csv(
    os.path.join(BASE, 'tables', 'patent_field_counts.csv'), index=False)
print('DONE', dict(counts), f'{time.time() - t0:.0f}s', flush=True)
