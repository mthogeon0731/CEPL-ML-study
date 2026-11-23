"""run_n3rep_compare.py : run_n3rep_eval.py 예측으로 짝지은 부트스트랩 비교표와 기억 누설 층화(S2 유명도) 상호작용.

비교 기준 집합: validation 전체(N3 보유), 중복 제거, 기존 확보분 359(앞 잠정 결과와 같은 분자), 새로 생성한 1,636.
S2: PubChem 등재 validation 분자의 동의어 수 3분위 — 상위 1/3(n_syn > 66.7 백분위) 대 하위 1/3(n_syn <= 33.3 백분위),
    경계는 random validation 전체 1,995 중 등재 분자에서 계산(README 11절, 앞 실행 117/63 재현 확인). 미등재는 따로.
I = mean Δ(유명) − mean Δ(희귀), 층 안 복원 추출 부트스트랩 2,000회.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('mp_mae_push/common'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mpcommon as mc  # noqa: E402

D = 'T8_nlp_n3rep'
PRED = os.path.join(D, 'preds')
idx, _ = mc.load_split('random')
dd_tab = pd.read_csv('mp_mae_push/phase2/prereg/dedup_rows_random.csv')
DEDUP = set(dd_tab.loc[(dd_tab['set'] == 'validate') & (dd_tab['overlap'] == 0), 'row'])
old = set(pd.read_csv(os.path.join(D, 'in_n3_texts_covered.csv'))['row'])


def P(rep, head, ts):
    return pd.read_csv(os.path.join(PRED, f'{rep}_{head}_{ts}_val.csv')).set_index('row')


CONTRASTS = []
for ts in ('ALL', 'NEW'):
    for head in ('xgb', 'ridge'):
        for e in ('minilm', 'bge'):
            CONTRASTS += [(ts, head, f'N3_{e}', f'N2_{e}', 'N3 − N2'),
                          (ts, head, f'N3_{e}', 'R217', 'N3 − RDKit217'),
                          (ts, head, f'C_R217N3{e}', 'R217', 'RDKit217+N3 − RDKit217')]
        CONTRASTS += [(ts, head, 'C_R217N3bge', 'C_R217N2bge', 'RDKit217+N3 − RDKit217+N2 (bge)'),
                      (ts, head, 'N2_bge', 'R217', 'N2 − RDKit217 (bge, 참고)'),
                      (ts, head, 'M2_R217F5', 'R217', 'M2 − RDKit217 (기술자 대조)')]

rows = []
for ts, head, a, b, kind in CONTRASTS:
    A, B = P(a, head, ts), P(b, head, ts)
    va = A.index.to_numpy()
    yv, pa, pb = A['y'].to_numpy(), A['pred'].to_numpy(), B.loc[va, 'pred'].to_numpy()
    rec = dict(train_set=ts, head=head, contrast=kind, A=a, B=b)
    for name, mask in [('all', np.ones(len(va), bool)), ('dedup', np.array([r in DEDUP for r in va])),
                       ('old359', np.array([r in old for r in va])), ('new', np.array([r not in old for r in va]))]:
        d, lo, hi = mc.paired_bootstrap(yv[mask], pa[mask], pb[mask])
        rec.update({f'n_{name}': int(mask.sum()), f'MAE_A_{name}': np.mean(np.abs(yv[mask] - pa[mask])),
                    f'MAE_B_{name}': np.mean(np.abs(yv[mask] - pb[mask])),
                    f'd_{name}': d, f'lo_{name}': lo, f'hi_{name}': hi})
    rows.append(rec)
C = pd.DataFrame(rows)
C.to_csv(os.path.join(D, 'table_n3rep_comparisons.csv'), index=False)

# ---- S2 유명도 층화 ----
nm = pd.read_csv(os.path.join(D, 'in_n1_pubchem_names.csv')).set_index('row')
va_all = idx['validate']
reg = nm.loc[va_all]
reg = reg[reg['in_pubchem'] == 1]
q1, q2 = np.percentile(reg['n_synonyms'], [100 / 3, 200 / 3])


def stratum(r):
    if nm.loc[r, 'in_pubchem'] != 1:
        return 'unregistered'
    s = nm.loc[r, 'n_synonyms']
    return 'famous' if s > q2 else ('rare' if s <= q1 else 'mid')


rng_seed = 0
LEAK = []
for ts in ('ALL', 'NEW'):
    for head in ('xgb', 'ridge'):
        pairs = [(f'N3_{e}', f'N2_{e}', '주① N3 − N2') for e in ('minilm', 'bge')] + \
                [('N3_bge', 'R217', '보조 N3 − RDKit217'), ('C_R217N3bge', 'R217', '보조 RDKit217+N3 − RDKit217'),
                 ('M2_R217F5', 'R217', '기술자 대조 M2 − RDKit217')]
        for a, b, kind in pairs:
            A, B = P(a, head, ts), P(b, head, ts)
            va = A.index.to_numpy()
            st = np.array([stratum(r) for r in va])
            dlt = np.abs(A['y'] - A['pred']).to_numpy() - np.abs(B.loc[va, 'y'] - B.loc[va, 'pred']).to_numpy()
            dA, dB, dU = dlt[st == 'famous'], dlt[st == 'rare'], dlt[st == 'unregistered']
            rng = np.random.default_rng(rng_seed)
            Is = np.array([dA[rng.integers(0, len(dA), len(dA))].mean() - dB[rng.integers(0, len(dB), len(dB))].mean()
                           for _ in range(2000)])
            lo, hi = np.percentile(Is, [2.5, 97.5])
            LEAK.append(dict(train_set=ts, head=head, contrast=kind, A=a, B=b, n_famous=len(dA), n_rare=len(dB),
                             n_unreg=len(dU), dA=dA.mean(), dA_ci=mc.boot_ci(dA), dB=dB.mean(), dB_ci=mc.boot_ci(dB),
                             d_unreg=dU.mean() if len(dU) else np.nan, I=dA.mean() - dB.mean(), I_lo=lo, I_hi=hi))
Lk = pd.DataFrame(LEAK)
Lk['q_cuts'] = f'{q1:g}/{q2:g}'
Lk.to_csv(os.path.join(D, 'table_n3rep_leak_interactions.csv'), index=False)
pd.set_option('display.width', 250)
print('cuts', q1, q2)
print(C[['train_set', 'head', 'contrast', 'A', 'n_all', 'MAE_A_all', 'MAE_B_all', 'd_all', 'lo_all', 'hi_all',
         'd_dedup', 'lo_dedup', 'hi_dedup']].round(2).to_string())
print(Lk[['train_set', 'head', 'contrast', 'A', 'n_famous', 'n_rare', 'dA', 'dB', 'I', 'I_lo', 'I_hi']].round(2).to_string())
