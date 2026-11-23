"""mp_xai_ext.py : [추가] mp 팀용 XAI 확장 (교수님 xai.py 는 고치지 않고 import 만 한다)

Crippen 원자 logP 기여는 logP 전용 '정답'이라 mp 에는 쓸 수 없다. 대신 Joback & Reid (1987)
녹는점 기여법  Tm = 122.5 K + Σ_g n_g ΔTm,g  의 작용기 몫을 원자에 나눠 '참고 답'으로 쓴다.
(Joback 은 정확도가 낮은 추정식이다 → GIN 설명과 다르다고 GIN 이 틀렸다는 뜻은 아니다.)

    from mp_xai_ext import joback_atom_contribs, joback_tm
    a, ok = joback_atom_contribs('CC(=O)O')     # 원자별 ΔTm [K], 합 = Joback Tm − 122.5
"""
import numpy as np
from rdkit import Chem
from collections import Counter
from thermo.group_contribution.joback import JOBACK_GROUPS, JOBACK_GROUPS_FOR_FRAGMENTATION

TM0 = 122.5   # Joback Tm 상수항 [K]


_PATT = {k: Chem.MolFromSmarts(v) for k, v in JOBACK_GROUPS_FOR_FRAGMENTATION.items()}


def joback_fragment(mol):
    """thermo.smarts_fragment 와 같은 규칙(겹치면 큰 작용기 우선)으로 나누고, 작용기별 원자 매칭을 돌려준다.
    thermo 0.6 의 smarts_fragment 는 개수만 돌려주므로 같은 논리를 다시 쓴다."""
    allm = {}
    for k, patt in _PATT.items():
        hits = list(mol.GetSubstructMatches(patt))
        if hits:
            allm[k] = hits
    for _ in range(100):
        used = [i for v in allm.values() for t in v for i in t]
        dups = [i for i, c in Counter(used).items() if c > 1]
        if not dups:
            break
        dup = dups[0]
        cand = [(g, j, len(t)) for g, v in allm.items() for j, t in enumerate(v) if dup in t]
        big = max(c[2] for c in cand)
        if [c[2] for c in cand].count(big) > 1:
            continue
        for g, j, n in sorted(cand, key=lambda c: -c[1]):
            if n != big:
                del allm[g][j]
    used = [i for v in allm.values() for t in v for i in t]
    ok = sorted(used) == list(range(mol.GetNumAtoms()))     # 무거운 원자마다 정확히 한 번
    return {k: v for k, v in allm.items() if v}, ok


def joback_atom_contribs(smiles):
    """원자마다 ΔTm [K]. 여러 원자로 된 작용기(-COOH, -NO2 …)는 그 원자들에 똑같이 나눈다.
    모든 무거운 원자가 Joback 작용기로 정확히 한 번 덮이지 않으면 ok=False (그 분자는 비교에서 뺀다)."""
    m = Chem.MolFromSmiles(smiles)
    allm, ok = joback_fragment(m)
    a = np.zeros(m.GetNumAtoms())
    for gid, matches in allm.items():
        dT = JOBACK_GROUPS[gid].Tm
        if dT is None:                     # Tm 값이 없는 작용기 (Joback 표에 빈칸) → 이 분자는 Joback Tm 을 못 낸다
            ok = False; continue
        for t in matches:
            for i in t:
                a[i] += dT / len(t)
    return (a if ok else np.full(m.GetNumAtoms(), np.nan)), ok


def joback_tm(smiles):
    a, ok = joback_atom_contribs(smiles)
    return (TM0 + a.sum()) if ok else np.nan


def hbd_atoms(smiles):
    """수소결합 공여 원자 (N-H, O-H 를 가진 무거운 원자) 인덱스."""
    m = Chem.MolFromSmiles(smiles)
    return [a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum() in (7, 8) and a.GetTotalNumHs() > 0]
