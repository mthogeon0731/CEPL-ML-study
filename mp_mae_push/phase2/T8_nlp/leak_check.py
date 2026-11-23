"""leak_check.py : 기억 누설 점검(H26) — README 11절의 사전 고정 기준대로 저장된 validation 예측만 읽어 계산한다.

출력: table_leak_interactions.csv (층 상호작용), table_leak_split_gap.csv (random–scaffold 격차)
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('T8_nlp'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import t8eval as te  # noqa: E402

PRED = te.PRED_DIR
nmt = pd.read_csv('T8_nlp/n1_pubchem_names.csv').set_index('row')
zsm = pd.read_csv('T8_nlp/zs_recall_per_molecule.csv').set_index('row')


def load(nm, how):
    p = os.path.join(PRED, f'{nm}_{how}_val.csv')
    return pd.read_csv(p).set_index('row') if os.path.exists(p) else None


def fame_labels(how, rows):
    va = te.split(how)['validate']
    q = nmt.loc[va, 'n_synonyms'].dropna().astype(float).quantile([1 / 3, 2 / 3]).to_numpy()
    out = []
    for r in rows:
        if nmt.loc[r, 'in_pubchem'] == 0:
            out.append('미등재')
            continue
        s = float(nmt.loc[r, 'n_synonyms'])
        out.append('희귀' if s <= q[0] else ('중간' if s <= q[1] else '유명'))
    return np.array(out)


def inter(d, mA, mB, n=2000, seed=0):
    dA, dB = d[mA], d[mB]
    g = np.random.default_rng(seed)
    bs = [dA[g.integers(0, len(dA), len(dA))].mean() - dB[g.integers(0, len(dB), len(dB))].mean() for _ in range(n)]
    return float(dA.mean()), float(dB.mean()), float(dA.mean() - dB.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


PAIRS = [  # (대조 이름, 후보, 비교 기준, 계열)
    ('주① N3−N2 (MiniLM, XGBoost)', 'N3_minilm_xgb', 'N2_minilm_xgb', 'LLM 서술'),
    ('주① N3−N2 (bge-small, XGBoost)', 'N3_bge_xgb', 'N2_bge_xgb', 'LLM 서술'),
    ('주① N3−N2 (MiniLM, Ridge)', 'N3_minilm_ridge', 'N2_minilm_ridge', 'LLM 서술'),
    ('주① N3−N2 (bge-small, Ridge)', 'N3_bge_ridge', 'N2_bge_ridge', 'LLM 서술'),
    ('보조 N3(MiniLM)−RDKit217 XGBoost', 'N3_minilm_xgb', 'REF_R217_xgb', '텍스트'),
    ('보조 N3(bge)−RDKit217 XGBoost', 'N3_bge_xgb', 'REF_R217_xgb', '텍스트'),
    ('보조 N2(MiniLM)−RDKit217 XGBoost', 'N2_minilm_xgb', 'REF_R217_xgb', '텍스트'),
    ('보조 N2(bge)−RDKit217 XGBoost', 'N2_bge_xgb', 'REF_R217_xgb', '텍스트'),
    ('보조 N1(MiniLM)−RDKit217 XGBoost', 'N1_minilm_xgb', 'REF_R217_xgb', '텍스트'),
    ('보조 RDKit217+N3(bge)−RDKit217 XGBoost', 'C_R217N3bge_xgb', 'REF_R217_xgb', '결합'),
    ('기술자 M2(F5)−RDKit217 XGBoost', 'M2_R217F5_xgb', 'REF_R217_xgb', '기술자'),
]
rows = []
for how in ('random', 'scaffold'):
    for lab, a, b, fam in PAIRS:
        A, B = load(a, how), load(b, how)
        if A is None or B is None:
            continue
        r = A.index.to_numpy()
        yv = A['y'].to_numpy()
        d = np.abs(yv - A['pred'].to_numpy()) - np.abs(yv - B.loc[r, 'pred'].to_numpy())
        fm = fame_labels(how, r)
        lst = nmt.loc[r, 'in_pubchem'].to_numpy() == 1
        for sname, mA, mB, la, lb in [('S2 유명도', fm == '유명', fm == '희귀', '유명(상위1/3)', '희귀(하위1/3)'),
                                       ('S1 PubChem', lst, ~lst, '등재', '미등재')]:
            dA, dB, I, lo, hi = inter(d, mA, mB)
            rows.append(dict(split=how, 대조=lab, 계열=fam, 층화=sname, 층A=la, nA=int(mA.sum()), ΔA=dA, 층B=lb,
                             nB=int(mB.sum()), ΔB=dB, I=I, lo=lo, hi=hi,
                             판정=('A층에서만 유독 좋음' if hi < 0 else ('A층에서 유독 나쁨' if lo > 0 else 'CI 0 포함'))))
        if how == 'random':
            zr = zsm.index.to_numpy()
            pos = pd.Series(np.arange(len(r)), index=r).loc[zr].to_numpy()
            ok = zsm['recall_success_le5K'].to_numpy() == 1
            dA, dB, I, lo, hi = inter(d[pos], ok, ~ok)
            rows.append(dict(split=how, 대조=lab, 계열=fam, 층화='S3 zero-shot 회상(200분자)', 층A='회상 성공(≤5 K)',
                             nA=int(ok.sum()), ΔA=dA, 층B='회상 실패', nB=int((~ok).sum()), ΔB=dB, I=I, lo=lo, hi=hi,
                             판정=('A층에서만 유독 좋음' if hi < 0 else ('A층에서 유독 나쁨' if lo > 0 else 'CI 0 포함'))))
T = pd.DataFrame(rows)
T.to_csv('T8_nlp/table_leak_interactions.csv', index=False, encoding='utf-8-sig')

# random–scaffold 상대 격차: (MAE_s − MAE_r)/MAE_r, 각 분할 validation 을 따로 복원 추출
MODELS = ['REF_R217_xgb', 'M2_R217F5_xgb', 'N1_minilm_xgb', 'N2_minilm_xgb', 'N2_bge_xgb', 'N3_minilm_xgb', 'N3_bge_xgb',
          'C_R217N3bge_xgb', 'REF_R217_ridge', 'M2_R217F5_ridge', 'N2_bge_ridge', 'N3_bge_ridge', 'N1_minilm_ridge']
g = np.random.default_rng(0)
E = {}
for m in MODELS:
    R, S = load(m, 'random'), load(m, 'scaffold')
    if R is None or S is None:
        continue
    E[m] = (np.abs(R['y'] - R['pred']).to_numpy(), np.abs(S['y'] - S['pred']).to_numpy())
B = 2000
boot = {}
ir = [g.integers(0, len(E['REF_R217_xgb'][0]), len(E['REF_R217_xgb'][0])) for _ in range(B)]
isc = [g.integers(0, len(E['REF_R217_xgb'][1]), len(E['REF_R217_xgb'][1])) for _ in range(B)]
for m, (er, es) in E.items():
    boot[m] = np.array([(es[j].mean() - er[i].mean()) / er[i].mean() for i, j in zip(ir, isc)])
gap = []
for m, (er, es) in E.items():
    refm = 'REF_R217_xgb' if m.endswith('xgb') else 'REF_R217_ridge'
    rel = (es.mean() - er.mean()) / er.mean()
    dd = boot[m] - boot[refm]
    gap.append(dict(model=m, MAE_random=er.mean(), MAE_scaffold=es.mean(), 상대격차=rel,
                    기준=refm, 상대격차_차이=rel - (E[refm][1].mean() - E[refm][0].mean()) / E[refm][0].mean(),
                    lo=float(np.percentile(dd, 2.5)), hi=float(np.percentile(dd, 97.5))))
G = pd.DataFrame(gap)
G.to_csv('T8_nlp/table_leak_split_gap.csv', index=False, encoding='utf-8-sig')
pd.set_option('display.width', 250)
print(T[['split', '대조', '층화', 'nA', 'nB', 'ΔA', 'ΔB', 'I', 'lo', 'hi', '판정']].round(2).to_string())
print(G.round(3).to_string())
