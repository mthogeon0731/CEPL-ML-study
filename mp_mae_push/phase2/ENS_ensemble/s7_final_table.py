"""s7_final_table.py : T3 전 멤버 풀(members.csv) 과 T3 포함 풀(members_T3.csv) 의 후보를 합쳐 최종 순위·비용·선정 표를 만든다.
T3 풀에서 나온 앙상블 후보는 이름 끝에 '@T3' 를 붙인다(단일 후보는 두 풀에서 같으므로 한 번만).
선정 규칙은 s6_select.py 머리말과 같다. 결과: tables/candidate_ranking_costs.csv, tables/candidate_per_split.csv
"""
import os, sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enslib as E
import numpy as np, pandas as pd

ACC = 'wavg_mse|noO2'
SEL = {'wavg_mse|noO2': '가장 정확', 'fixed_mean|best+GIN': '가장 경제적', 'fixed_mean|F0F5+GIN': '균형안'}


def run():
    import s6_select as S6
    E.MEMBERS_FILE = 'members.csv'
    T0, P0, ys, dms = S6.build('')
    E.MEMBERS_FILE = 'members_T3.csv'
    T1, P1, _, _ = S6.build('_T3')
    E.MEMBERS_FILE = 'members.csv'
    T1 = T1.copy()
    ens1 = T1.kind != '단일'
    T1.loc[ens1, 'candidate'] = T1.loc[ens1, 'candidate'] + '@T3'
    P1 = {((k[0] + '@T3') if (k[0] in set(T1.loc[ens1, 'candidate'].str[:-3])) else k[0], k[1]): v for k, v in P1.items()}
    T1 = T1[ens1 | T1.candidate.str.startswith('t3_')]
    T = pd.concat([T0, T1], ignore_index=True)
    preds = {**P0, **P1}
    for how in E.SPLITS:
        k = T.split == how
        T.loc[k, 'rank_full'] = T.loc[k, 'MAE'].rank(method='min').astype(int)
        T.loc[k, 'rank_dedup'] = T.loc[k, 'MAE_dedup'].rank(method='min').astype(int)
    T['rank_differs'] = (T['rank_full'] != T['rank_dedup']).astype(int)
    T['pool'] = np.where(T.candidate.str.endswith('@T3'), 'T3 포함', np.where(T.candidate.str.startswith('t3_'), 'T3 포함', 'T3 전'))
    Cdf, B, R = S6.costs_for(T)
    both = T.groupby('candidate').split.nunique()
    wide = T.pivot_table(index='candidate', columns='split', values=['MAE', 'MAE_dedup', 'RMSE', 'rank_full', 'rank_dedup'])
    wide.columns = [f'{a}_{b}' for a, b in wide.columns]
    info = T.groupby('candidate').agg(kind=('kind', 'first'), posthoc=('posthoc', 'first'), uses_O2=('uses_O2', 'max'), pool=('pool', 'first'))
    A = info.join(wide).join(Cdf.set_index('candidate')); A['both_splits'] = both
    A['MAE_mean'] = (A.MAE_random + A.MAE_scaffold) / 2
    A['MAE_dedup_mean'] = (A.MAE_dedup_random + A.MAE_dedup_scaffold) / 2
    cmp = []
    for c in A.index:
        r = {'candidate': c}
        for how in E.SPLITS:
            if (c, how) not in preds:
                continue
            y_, dm = ys[how], dms[how]
            for ref, lab in ((ACC, 'acc'), ('lgb_r1_F0F3F5', 'best'), ('baseline_xgb600_F0', 'base')):
                t = E.pboot(y_, preds[(c, how)], preds[(ref, how)]); td = E.pboot(y_[dm], preds[(c, how)][dm], preds[(ref, how)][dm])
                r[f'd_vs_{lab}_{how}'] = E.fmt_ci(t); r[f'd_vs_{lab}_dedup_{how}'] = E.fmt_ci(td)
                r[f'd_vs_{lab}_{how}_lo'] = t[1]; r[f'd_vs_{lab}_{how}_hi'] = t[2]
        cmp.append(r)
    A = A.join(pd.DataFrame(cmp).set_index('candidate'))
    A['indist_acc'] = ((A.both_splits == 2) & (A.d_vs_acc_random_lo <= 0) & (A.d_vs_acc_scaffold_lo <= 0)).astype(int)
    base_mae = A.loc['baseline_xgb600_F0', 'MAE_mean']
    B3 = A[(A.both_splits == 2) & (A.MAE_mean <= base_mae)].reset_index()
    A['on_pareto_total'] = A.index.isin(S6.pareto(B3, x='total_usd')).astype(int)
    A['on_pareto_infer'] = A.index.isin(S6.pareto(B3, x='infer_total_s_per_1k')).astype(int)
    kn_t, _ = S6.knee(B3[B3.candidate.isin(A.index[A.on_pareto_total == 1])], x='total_usd')
    kn_i, _ = S6.knee(B3[B3.candidate.isin(A.index[A.on_pareto_infer == 1])], x='infer_total_s_per_1k')
    A['knee'] = ''
    A.loc[kn_t, 'knee'] += '총비용 무릎;'; A.loc[kn_i, 'knee'] += '추론비용 무릎;'
    pre = A[(A.posthoc == 0) & (A.both_splits == 2)]
    acc_rule = pre.MAE_mean.idxmin()
    econ_rule = A[A.indist_acc == 1].sort_values(['total_usd', 'posthoc']).index[0]
    A['selected_role'] = A.index.map(lambda c: SEL.get(c, ''))
    for how in E.SPLITS:
        A[f'rank_differs_{how}'] = (A[f'rank_full_{how}'] != A[f'rank_dedup_{how}']).astype(float).where(A[f'rank_full_{how}'].notna())
    A.sort_values('MAE_mean').to_csv(os.path.join(E.ENS, 'tables', 'candidate_ranking_costs.csv'))
    T.assign(members=T.members.apply(';'.join)).to_csv(os.path.join(E.ENS, 'tables', 'candidate_per_split.csv'), index=False)
    return A, T, preds, dict(acc_rule=acc_rule, econ_rule=econ_rule, knee_total=kn_t, knee_infer=kn_i), B, R
