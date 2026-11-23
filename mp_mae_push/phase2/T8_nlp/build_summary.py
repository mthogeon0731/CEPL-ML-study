"""build_summary.py : 저장된 validation 예측만 다시 읽어 T8 요약표(ΔMAE, 두 head·두 분할·두 기준)와 표본 수 곡선 표를 만든다."""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('T8_nlp'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import t8eval as te  # noqa: E402
import t8report as tr  # noqa: E402

REPS = [('REF_R217', 'RDKit 217 (비교 기준)'), ('M2_R217F5', 'M2: RDKit 217 + LLM 규칙 F5'), ('M2_F5only', 'M2: F5 단독'),
        ('N2_minilm', 'N2 템플릿 → MiniLM'), ('N2_bge', 'N2 템플릿 → bge-small'), ('N1_minilm', 'N1 IUPAC 이름 → MiniLM')]
rows = []
for rep, label in REPS:
    for head in ('xgb', 'ridge'):
        nm = f'{rep}_{head}'
        for how in ('random', 'scaffold'):
            if not os.path.exists(os.path.join(te.PRED_DIR, f'{nm}_{how}_val.csv')):
                continue
            ref = f'REF_R217_{head}'
            a = tr.compare(nm, ref, how, None)
            b = tr.compare(nm, ref, how, 'dedup')
            c = tr.compare(nm, tr.baseline_name(how), how, None)
            seeds = pd.read_csv(os.path.join(te.PRED_DIR, 'seeds', f'{nm}_{how}_val_seeds.csv')).set_index('row')
            yv = tr.load(nm, how)['y'].loc[seeds.index]
            per = [float(np.mean(np.abs(yv - seeds[c_]))) for c_ in seeds.columns]
            rows.append(dict(표현=label, run=nm, head=head, split=how, n_val=a['n'], MAE=a['MAE_new'],
                             MAE_seed_mean=np.mean(per), MAE_seed_sd=(np.std(per, ddof=1) if len(per) > 1 else 0.0),
                             n_seed=len(per), dMAE_vs_R217=a['dMAE'], lo=a['lo'], hi=a['hi'], n_dedup=b['n'],
                             MAE_dedup=b['MAE_new'], dMAE_dedup=b['dMAE'], lo_dedup=b['lo'], hi_dedup=b['hi'],
                             dMAE_vs_phase1=c['dMAE'], lo_p1=c['lo'], hi_p1=c['hi']))
S = pd.DataFrame(rows)
# 두 기준 순위 (분할·head 별)
for (how, head), g in S.groupby(['split', 'head']):
    S.loc[g.index, 'rank_all'] = g['MAE'].rank(method='min').astype(int)
    S.loc[g.index, 'rank_dedup'] = g['MAE_dedup'].rank(method='min').astype(int)
S['순위_불일치'] = np.where(S['rank_all'] != S['rank_dedup'], '※', '')
S.to_csv('T8_nlp/table_T8_summary.csv', index=False, encoding='utf-8-sig')

# 표본 수 곡선: 시드(=부분표본)별 MAE 평균 ± SD. 전체 n 은 XGBoost seed 3개, Ridge 는 1개.
cur = []
for rep, label in REPS:
    for head in ('xgb', 'ridge'):
        nm = f'{rep}_{head}'
        for how in ('random', 'scaffold'):
            for n in (1000, 4000, None):
                f = os.path.join(te.PRED_DIR, 'seeds', f'{nm}_{how}' + ('' if n is None else f'_n{n}') + '_val_seeds.csv')
                if not os.path.exists(f):
                    continue
                P = pd.read_csv(f).set_index('row')
                yv = te.y_all()[P.index.to_numpy()]
                dd = te.dedup_mask(how, P.index.to_numpy())
                per = [float(np.mean(np.abs(yv - P[c]))) for c in P.columns]
                per_dd = [float(np.mean(np.abs(yv[dd] - P[c].to_numpy()[dd]))) for c in P.columns]
                ntr = n if n is not None else len(te.split(how)['train'])
                cur.append(dict(표현=label, run=nm, head=head, split=how, n_train=ntr, n_seed=len(per),
                                MAE_mean=np.mean(per), MAE_sd=np.std(per, ddof=1) if len(per) > 1 else 0.0,
                                MAE_dedup_mean=np.mean(per_dd),
                                MAE_dedup_sd=np.std(per_dd, ddof=1) if len(per_dd) > 1 else 0.0))
C = pd.DataFrame(cur)
C.to_csv('T8_nlp/table_T8_learning_curves.csv', index=False, encoding='utf-8-sig')
print(S[['run', 'split', 'MAE', 'dMAE_vs_R217', 'lo', 'hi', 'MAE_dedup', 'dMAE_dedup', 'rank_all', 'rank_dedup']].round(3).to_string())
print(C.round(2).to_string())
