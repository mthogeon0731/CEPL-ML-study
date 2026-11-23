"""s3b_family_adjusted.py : 계열 효과의 겹침 보정 (모델 적합 없음; OOF·validation 예측과 mol_flags 만 읽는다. test 미사용).

(1) 2D 대칭 높음 x 무거운 원자 구간 / MP 구간 — 대칭 계열의 큰 오차가 크기·MP 구간 때문인지 확인
(2) |오차| ~ 계열 표지(13) + 무거운 원자 구간  보통최소제곱 (계열이 겹치므로 '다른 표지가 같을 때의 추가 오차')
(3) 상위 50개 목록의 계열 농축도 (목록 안 비율 / 전체 비율)
(4) 큰 OOF 오차(꼬리)가 MAE 에 차지하는 몫과 '액체형 구조인데 고융점 라벨' 규칙의 해당 수
출력: phase1/tables/mae_sym_x_size.csv, family_adjusted_excess.csv, top50_family_enrichment.csv, oof_error_tail.csv
"""
import os

import numpy as np
import pandas as pd

import mpcommon as mc

TAB = os.path.join(mc.ROOT, 'phase1', 'tables')
PRED = os.path.join(mc.ROOT, 'phase1', 'preds')
LEDGER = os.path.join(mc.ROOT, 'phase1', 'ledger_errdecomp.csv')
FL = pd.read_csv(os.path.join(mc.CACHE_DIR, 'mol_flags.csv'))
FAMS = ['charged_excl_nitro', 'charged_any', 'stereo_annotated', 'stereo_possible_unspecified', 'dblbond_stereo_unspecified',
        'acyclic', 'polycyclic_ge3', 'halogen_heavy', 'carboxylic_acid', 'amide_urea', 'hbd_ge3', 'rigid_rotb0', 'sym2d_high']
MP_EDGES, MP_LAB = [-np.inf, 250, 350, 450, 550, np.inf], ['<250', '250-350', '350-450', '450-550', '>550']
HV_EDGES, HV_LAB = [-np.inf, 10, 20, 30, 45, np.inf], ['<=10', '11-20', '21-30', '31-45', '>45']

with mc.CostMeter('탐색') as cost:
    sets = {}
    for how in ('random', 'scaffold'):
        for src, f in (('oof_train', f'oof_baseline_{how}_train.csv'), ('validation', f'baseline_xgb_desc217_{how}_val.csv')):
            d = pd.read_csv(os.path.join(PRED, f))[['row', 'y', 'pred']]
            d = d.merge(FL.drop(columns=['smiles']), on='row', how='left')
            d['resid'] = d['pred'] - d['y']
            d['abs'] = d['resid'].abs()
            d['heavy_bin'] = pd.cut(d['heavy'], HV_EDGES, labels=HV_LAB, right=True).astype(str)
            d['mp_bin'] = pd.cut(d['y'], MP_EDGES, labels=MP_LAB, right=False).astype(str)
            sets[(how, src)] = d

    # (1) 대칭 x 크기 / MP
    rows = []
    for (how, src), d in sets.items():
        for by, labs in (('heavy_bin', HV_LAB), ('mp_bin', MP_LAB)):
            for lab in labs:
                g = d[d[by] == lab]
                a, b = g[g['sym2d_high']], g[~g['sym2d_high']]
                rows.append(dict(split=how, source=src, by=by, bin=lab, n_sym_high=len(a), MAE_sym_high_K=a['abs'].mean(),
                                 mean_resid_sym_high_K=a['resid'].mean(), n_other=len(b), MAE_other_K=b['abs'].mean(),
                                 mean_resid_other_K=b['resid'].mean(), diff_K=a['abs'].mean() - b['abs'].mean()))
    pd.DataFrame(rows).to_csv(os.path.join(TAB, 'mae_sym_x_size.csv'), index=False, encoding='utf-8-sig')

    # (2) 겹침 보정 OLS
    rows = []
    for (how, src), d in sets.items():
        use = [f for f in FAMS if d[f].sum() >= 20 and (~d[f]).sum() >= 20]
        dm = pd.get_dummies(d['heavy_bin'])[[h for h in HV_LAB if h != '11-20' and (d['heavy_bin'] == h).sum() >= 20]].astype(float)
        A = np.c_[np.ones(len(d)), d[use].astype(float).values, dm.values]
        names = ['(절편: 표지 없음, 무거운 원자 11-20)'] + use + [f'heavy {h}' for h in dm.columns]
        yv = d['abs'].values
        coef = np.linalg.lstsq(A, yv, rcond=None)[0]
        rng = np.random.default_rng(0)
        bs = np.array([np.linalg.lstsq(A[i], yv[i], rcond=None)[0] for i in (rng.integers(0, len(d), len(d)) for _ in range(500))])
        lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
        for k, nm in enumerate(names):
            n = int(A[:, k].sum())
            rows.append(dict(split=how, source=src, term=nm, n=n, adj_excess_abs_err_K=coef[k], ci_lo=lo[k], ci_hi=hi[k],
                             adj_contrib_to_MAE_K=(coef[k] * n / len(d)) if k > 0 else np.nan, set_n=len(d), set_MAE_K=yv.mean()))
    AD = pd.DataFrame(rows)
    AD.to_csv(os.path.join(TAB, 'family_adjusted_excess.csv'), index=False, encoding='utf-8-sig')

    # (3) 상위 50 농축도
    rows = []
    for how, fname in (('random', 'top50_oof_errors.csv'), ('scaffold', 'top50_oof_errors_scaffold.csv')):
        t = pd.read_csv(os.path.join(TAB, fname))
        d = sets[(how, 'oof_train')]
        tt = d[d['row'].isin(t['row'])]
        assert len(tt) == 50
        for f in FAMS:
            rows.append(dict(split=how, family=f, n_in_top50=int(tt[f].sum()), frac_top50=tt[f].mean(), frac_all_train=d[f].mean(),
                             enrichment=tt[f].mean() / d[f].mean() if d[f].mean() > 0 else np.nan))
        rows.append(dict(split=how, family='(참고) y > 550 K', n_in_top50=int((tt['y'] > 550).sum()), frac_top50=(tt['y'] > 550).mean(),
                         frac_all_train=(d['y'] > 550).mean(), enrichment=(tt['y'] > 550).mean() / (d['y'] > 550).mean()))
        rows.append(dict(split=how, family='(참고) 과소예측 (pred < y)', n_in_top50=int((tt['resid'] < 0).sum()), frac_top50=(tt['resid'] < 0).mean(),
                         frac_all_train=(d['resid'] < 0).mean(), enrichment=(tt['resid'] < 0).mean() / (d['resid'] < 0).mean()))
    EN = pd.DataFrame(rows)
    EN.to_csv(os.path.join(TAB, 'top50_family_enrichment.csv'), index=False, encoding='utf-8-sig')

    # (4) 오차 꼬리: 큰 OOF 오차가 MAE 에 차지하는 몫과 그 성격 (라벨 오류 후보)
    #     '액체형' 규칙(구조+라벨만, 모델 예측 미사용): 사슬형, 수소결합 주개 0, 전하 없음, 무거운 원자 <= 15 인데 라벨 >= 400 K
    rows = []
    for (how, src), d in sets.items():
        if src != 'oof_train':
            continue
        liquid_like = d['acyclic'] & (d['hbd'] == 0) & (~d['charged_any']) & (d['heavy'] <= 15)
        for thr in (50, 100, 150, 200):
            g = d[d['abs'] >= thr]
            rows.append(dict(split=how, source=src, subset=f'|OOF 오차| >= {thr} K', n=len(g), frac=len(g) / len(d),
                             sum_abs_over_N_K=g['abs'].sum() / len(d), MAE_without_K=d.loc[d['abs'] < thr, 'abs'].mean(),
                             frac_underpred=(g['resid'] < 0).mean(), frac_acyclic=g['acyclic'].mean(),
                             frac_liquid_like_struct=liquid_like[g.index].mean(), set_n=len(d), set_MAE_K=d['abs'].mean()))
        g = d[liquid_like & (d['y'] >= 400)]
        rows.append(dict(split=how, source=src, subset='규칙: 액체형 구조인데 라벨 >= 400 K', n=len(g), frac=len(g) / len(d),
                         sum_abs_over_N_K=g['abs'].sum() / len(d), MAE_without_K=d.loc[~(liquid_like & (d['y'] >= 400)), 'abs'].mean(),
                         frac_underpred=(g['resid'] < 0).mean(), frac_acyclic=g['acyclic'].mean(), frac_liquid_like_struct=1.0,
                         set_n=len(d), set_MAE_K=d['abs'].mean(), subset_MAE_K=g['abs'].mean()))
        g = d[liquid_like]
        rows.append(dict(split=how, source=src, subset='참고: 액체형 구조 전체', n=len(g), frac=len(g) / len(d),
                         sum_abs_over_N_K=g['abs'].sum() / len(d), frac_underpred=(g['resid'] < 0).mean(), frac_acyclic=1.0,
                         frac_liquid_like_struct=1.0, set_n=len(d), set_MAE_K=d['abs'].mean(), subset_MAE_K=g['abs'].mean()))
    TL = pd.DataFrame(rows)
    TL.to_csv(os.path.join(TAB, 'oof_error_tail.csv'), index=False, encoding='utf-8-sig')

mc.append_ledger(LEDGER, dict(run_id='errdecomp_s3b_family_adjusted', track='errdecomp', model='-(저장된 OOF·validation 예측 사용)',
                              features='-', data_variant='-', objective='-', split='random+scaffold', seed=0, params='{"n_boot": 500}',
                              n_train=0, notes='계열 겹침 보정 OLS·대칭x크기·상위50 농축도; 모델 적합 없음, test 미사용', **cost.as_row()))
pd.set_option('display.width', 250)
print(TL.round(3).to_string(index=False))
