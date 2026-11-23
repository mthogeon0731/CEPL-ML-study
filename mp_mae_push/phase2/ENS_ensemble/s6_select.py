"""s6_select.py : 후보 전체 표(단일·앙상블), 비용, 두 기준 순위, 가장 정확 대비 짝지은 비교, 최종 후보 선정.
후보 = (a) 단일 멤버 validation 예측, (b) s2 사전 지정 방법(정직한 교차 적합 예측), (c) s2b 고정 부분집합(사후 지정, 표시).
train OOF 가중치 확인용 행은 순위에서 뺀다.
선정 규칙 (README 에 같은 문장으로 적음):
  가장 정확 = 사전 지정 방법(b)과 단일(a) 중, 두 분할 전체 validation MAE 평균이 가장 낮은 것.
  가장 경제적 = 두 분할 모두에서 정의되고, 두 분할 모두 '가장 정확' 대비 짝지은 부트스트랩 95 % CI 가 0 을 포함하는 후보 중
                총비용(USD; 탐색 + 최종 학습) 이 가장 싼 것. (PREREG 정의)
  균형안 = 두 분할 평균 MAE 대 log10 총비용의 파레토 전선에서 '무릎'(양 끝점을 잇는 직선에서 정규화 거리 최대) 인 후보.
"""
import os, sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enslib as E
import s2_ensemble as S
import s5_costs as C5
import numpy as np, pandas as pd


def members_of(row_w, names_all):
    w = json.loads(row_w['weights'])
    return [k for k, v in w.items() if abs(v) > 1e-4]


def build(tag=''):
    reg = pd.read_csv(os.path.join(E.ENS, E.MEMBERS_FILE))
    res = pd.read_csv(os.path.join(E.ENS, f'ensemble_results{tag}.csv'))
    W = pd.read_csv(os.path.join(E.ENS, 'tables', f'ensemble_final_weights{tag}.csv'))
    FX = pd.read_csv(os.path.join(E.ENS, 'tables', f'ensemble_fixed_subsets{tag}.csv'))
    FW = pd.read_csv(os.path.join(E.ENS, 'tables', f'ensemble_fixed_weights{tag}.csv'))
    preds, rows = {}, []
    for how in E.SPLITS:
        mv, y = E.load_members(how, 'val', S.members_for(how, 'val', reg))
        for n, p in mv.items():
            preds[(n, how)] = p
            rows.append(dict(candidate=n, kind='단일', posthoc=0, split=how, members=[n]))
        for r in res[(res.split == how) & (res.method != 'single') & (~res.method.str.endswith('trainOOF')) & (~res.member_set.str.contains('oofset'))].itertuples():
            name = f'{r.method}|{r.member_set}'
            f = os.path.join(E.ENS, 'preds', 'ens', f'{how}__{r.method}__{r.member_set}{tag}_val_honest.csv')
            preds[(name, how)] = pd.read_csv(f)['pred'].to_numpy()
            wr = W[(W.split == how) & (W.method == r.method) & (W.member_set == r.member_set)].iloc[0]
            rows.append(dict(candidate=name, kind='앙상블(사전 지정)', posthoc=0, split=how, members=members_of(wr, None),
                             weights=wr['weights'], intercept=wr['intercept'], alpha=wr['alpha']))
        for r in FX[FX.split == how].itertuples():
            name = f'{r.method}|{r.member_set}'
            f = os.path.join(E.ENS, 'preds', 'ens', f'{how}__{r.method}__{r.member_set}{tag}_val_honest.csv')
            preds[(name, how)] = pd.read_csv(f)['pred'].to_numpy()
            wr = FW[(FW.split == how) & (FW.method == r.method) & (FW.member_set == r.member_set)].iloc[0]
            rows.append(dict(candidate=name, kind='앙상블(고정 부분집합, 사후 지정)', posthoc=1, split=how,
                             members=[k for k, v in json.loads(wr['weights']).items() if abs(v) > 1e-4] if r.method != 'fixed_mean' else wr['selected'].split(';'),
                             weights=wr['weights'], intercept=0.0, alpha=''))
    T = pd.DataFrame(rows)
    ys = {how: E.y_all()[E.split(how)[0]['validate']] for how in E.SPLITS}
    dms = {how: E.dedup_mask(how) for how in E.SPLITS}
    T['MAE'] = [E.mae(ys[r.split], preds[(r.candidate, r.split)]) for r in T.itertuples()]
    T['MAE_dedup'] = [E.mae(ys[r.split][dms[r.split]], preds[(r.candidate, r.split)][dms[r.split]]) for r in T.itertuples()]
    T['RMSE'] = [float(np.sqrt(np.mean((ys[r.split] - preds[(r.candidate, r.split)]) ** 2))) for r in T.itertuples()]
    T['uses_O2'] = T['members'].apply(lambda m: int('O2_knn' in m))
    for how in E.SPLITS:
        k = T.split == how
        T.loc[k, 'rank_full'] = T.loc[k, 'MAE'].rank(method='min').astype(int)
        T.loc[k, 'rank_dedup'] = T.loc[k, 'MAE_dedup'].rank(method='min').astype(int)
    T['rank_differs'] = (T['rank_full'] != T['rank_dedup']).astype(int)
    T['primary_criterion'] = np.where(T['uses_O2'] == 1, '중복 제거(kNN 계열 규칙)', '전체')
    return T, preds, ys, dms


def costs_for(T):
    # 후보별 멤버 합집합(두 분할) 으로 비용
    mem = T.groupby('candidate')['members'].apply(lambda s: sorted({m for l in s for m in l}))
    feat = dict(C5.FEAT_S1K); feat['morgan_knn'] = 4.55; feat['aux'] = 0.0
    cands = {c: m for c, m in mem.items()}
    Cdf, B, R = C5.candidate_costs(cands, feat, extra_blocks=('ENS_ensemble',))
    return Cdf, B, R


def pareto(df, x='total_usd', y='MAE_mean'):
    d = df.sort_values([x, y]).reset_index(drop=True)
    best, keep = np.inf, []
    for i, r in d.iterrows():
        if r[y] < best - 1e-9:
            keep.append(r['candidate']); best = r[y]
    return keep


def knee(front_df, x='total_usd', y='MAE_mean'):
    d = front_df.sort_values(x)
    lx = np.log10(d[x].to_numpy()); yy = d[y].to_numpy()
    nx = (lx - lx.min()) / (lx.max() - lx.min() + 1e-12); ny = (yy - yy.min()) / (yy.max() - yy.min() + 1e-12)
    # 직선: (0,1) -> (1,0)  [가장 싼 점 (MAE 큼) → 가장 정확한 점]
    p0, p1 = np.array([nx[0], ny[0]]), np.array([nx[-1], ny[-1]])
    v = p1 - p0; v = v / np.linalg.norm(v)
    dist = [abs(np.cross(v, np.array([a, b]) - p0)) for a, b in zip(nx, ny)]
    return d['candidate'].iloc[int(np.argmax(dist))], dict(zip(d['candidate'], dist))
