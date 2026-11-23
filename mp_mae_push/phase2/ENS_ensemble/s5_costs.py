"""s5_costs.py : 후보별 비용 정리 (탐색 비용 + 최종 학습 비용, 추론 비용). 비용 가정: cost_assumptions.md
  core-h -> kWh = core-h × 25 W / 12 / 1000,  USD = core-h × 0.0504 (AWS m7i vCPU-h) + API USD.
탐색 비용 = 후보의 '계보'에 든 블록(아래 BLOCKS)의 ledger 합. 블록은 ledger 행을 run_id·features 규칙으로 고른다.
  트랙 안에서 후보와 무관한 실험(예: T8 N1–N3, T5 random 곡선)은 그 후보의 계보에 넣지 않는다. 같은 블록을 여러 멤버가
  쓰면 앙상블에서는 한 번만 센다(합집합).
최종 학습 비용 = train 전용 재적합(두 분할, 기록된 seed 수 전부)의 ledger 실측 core-h. 특징은 19,945행 캐시가 있어 넣지 않는다.
추론 비용 [s/1,000분자, 1 코어 기준] = 특징 계산(새 분자) + 모델 예측(seed 수만큼).
test 파일·test 열은 읽지 않는다 (ledger 의 test_* 열은 버린다).
"""
import os, sys, json, re
import numpy as np, pandas as pd

WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
ENS = os.path.join(WS, 'ENS')
USD_PER_COREH, P_W, NLOG = 0.0504, 25.0, 12


def load_ledgers():
    L = {}
    for f in os.listdir(os.path.join(ENS, 'inputs', 'ledgers')):
        t = f.replace('_ledger.csv', '')
        d = pd.read_csv(os.path.join(ENS, 'inputs', 'ledgers', f), encoding='utf-8-sig')
        d = d.drop(columns=[c for c in d.columns if c.startswith('test_')])
        for c in ('cpu_core_h', 'api_cost', 'infer_s_per_1k'):
            d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0 if c != 'infer_s_per_1k' else np.nan)
        L[t] = d
    d = pd.read_csv(os.path.join(ENS, 'ledger.csv'), encoding='utf-8-sig')
    d = d.drop(columns=[c for c in d.columns if c.startswith('test_')])
    for c in ('cpu_core_h', 'api_cost'):
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0)
    L['ENS'] = d
    return L


def s(d, m):
    x = d[m]
    return dict(core_h=float(x['cpu_core_h'].sum()), api=float(x['api_cost'].sum()), n_rows=int(len(x)))


def blocks(L):
    r = lambda t: L[t]['run_id'].astype(str)
    f = lambda t: L[t]['features'].astype(str)
    B = {
        'D0_desc217': s(L['P1'], (L['P1']['stage'] == '특징')),
        'P1_baseline_fit': s(L['P1'], (L['P1']['stage'] == '최종')),
        'T1_S1_seed_loss': s(L['T1'], r('T1').str.startswith('s1_')),
        'T1_S2_features_F1O3': s(L['T1'], r('T1').str.startswith('s2_')),
        'T1_S3_tuning': s(L['T1'], r('T1').str.match(r's3_(es|inner|lgb|xgb)')),
        'T1_fit_lgb_r1': s(L['T1'], r('T1').str.startswith('s3_fit_lgb_r1')),
        'T1_fit_xgb_r1': s(L['T1'], r('T1').str.startswith('s3_fit_xgb_r1')),
        'T1_oof_lgb_r1': s(L['T1'], r('T1').str.startswith('s3_oof_lgb')),
        'T1_oof_xgb_r1': s(L['T1'], r('T1').str.startswith('s3_oof_xgb')),
        'T1b_bundles': s(L['T1b'], r('T1b').str.startswith('s1_lgb')),
        'T1b_oof_lgb_F0F3F5': s(L['T1b'], r('T1b').str.startswith('oof_lgb')),
        'T1b_h06': s(L['T1b'], r('T1b').str.startswith('h06_')),
        'T4_F3_featcalc': s(L['T4'], f('T4').str.startswith('F3 39')),
        'T4_F3_fits': s(L['T4'], f('T4').str.startswith('F0+F3')),
        'T4_aux': s(L['T4'], r('T4').str.startswith('T4_aux') | f('T4').str.contains('보조') | (f('T4') == 'desc217 (보조 물성 5종)')),
        'T8_F5_rules_M2': s(L['T8'], f('T8').str.contains(r'F5|M2|규칙 제안', regex=True)),
        'T8rep_M2': s(L['T8rep'], f('T8rep').str.contains('M2')),
        'T5_patent': s(L['T5'], r('T5').str.startswith('T5_parse') | r('T5').str.startswith('T5_featcalc') | r('T5').str.startswith('T5_rescan')
                       | r('T5').str.contains('add1x')),
        'T5_patent_curve_other': s(L['T5'], r('T5').str.startswith('T5_val') & ~r('T5').str.contains('add1x')),
        'T2_GIN': s(L['T2'], L['T2']['stage'].isin(['main', 'pilot', 'pilot_probe', 'featurize'])),
        'T2_GIN_graphonly_control': s(L['T2'], L['T2']['stage'] == 'control'),
        'ENS_knn': s(L['ENS'], r('ENS').str.startswith('O2_knn')),
        'ENS_ensemble': s(L['ENS'], r('ENS').str.startswith('H10_')),
    }
    if 'T3' in L:
        t3 = L['T3']; r3 = t3['run_id'].astype(str)
        B.update({
            'T3_embed_cbmlm': s(t3, r3.isin(['embed_cb_mlm', 'pilot500_cb_mlm'])),
            'T3_embed_cbmtr': s(t3, r3.isin(['embed_cb_mtr', 'pilot500_cb_mtr'])),
            'T3_embed_molformer': s(t3, r3.isin(['embed_molformer', 'pilot500_molformer'])),
            'T3_fit_R217_cbmlm': s(t3, r3.str.match(r'xgb_R217_cb_mlm_(random|scaffold)_s\d$')),
            'T3_fit_R217_molformer': s(t3, r3.str.match(r'xgb_R217_molformer_(random|scaffold)_s\d$')),
            'T3_fit_R217_molformer_pca64': s(t3, r3.str.match(r'xgb_R217_molformer-pca64_(random|scaffold)_s\d$')),
            'T3_fit_molformer_only': s(t3, r3.str.match(r'xgb_molformer_(random|scaffold)_s\d$')),
            'T3_fit_cbmtr_only': s(t3, r3.str.match(r'xgb_cb_mtr_(random|scaffold)_s\d$')),
            'T3_fit_R217_s012': s(t3, r3.str.match(r'xgb_R217_(random|scaffold)_s[012]$')),
        })
    return B


LINEAGE = {
    'baseline_xgb600_F0': ['D0_desc217', 'P1_baseline_fit'],
    'xgb600_F0_bag10': ['D0_desc217', 'T1_S1_seed_loss'],
    'lgb_r1_F0': ['D0_desc217', 'T1_S1_seed_loss', 'T1_S3_tuning', 'T1_fit_lgb_r1', 'T1_oof_lgb_r1'],
    'xgb_r1_F0': ['D0_desc217', 'T1_S1_seed_loss', 'T1_S3_tuning', 'T1_fit_xgb_r1', 'T1_oof_xgb_r1'],
    'xgb600_F0F1O3': ['D0_desc217', 'T1_S2_features_F1O3'],
    'lgb_r1_F0F3F5': ['D0_desc217', 'T1_S1_seed_loss', 'T1_S3_tuning', 'T4_F3_featcalc', 'T8_F5_rules_M2', 'T1b_bundles', 'T1b_oof_lgb_F0F3F5'],
    'lgb_r1_F0F5': ['D0_desc217', 'T1_S1_seed_loss', 'T1_S3_tuning', 'T8_F5_rules_M2', 'T1b_bundles'],
    'h06_yalk_F0F3F5': ['D0_desc217', 'T1_S1_seed_loss', 'T1_S3_tuning', 'T4_F3_featcalc', 'T8_F5_rules_M2', 'T1b_bundles', 'T1b_oof_lgb_F0F3F5', 'T1b_h06'],
    'xgb600_F0F5_M2': ['D0_desc217', 'T8_F5_rules_M2', 'T8rep_M2'],
    'xgb600_F0F3': ['D0_desc217', 'T4_F3_featcalc', 'T4_F3_fits'],
    'xgb600_F0aux4_scaf': ['D0_desc217', 'T4_aux'],
    'xgb600_F0_patent1x_scaf': ['D0_desc217', 'T5_patent'],
    'gin_hybrid_seedavg': ['D0_desc217', 'T2_GIN'],
    'O2_knn': ['ENS_knn'],
    't3_xgb_R217_cbMLM': ['D0_desc217', 'T3_embed_cbmlm', 'T3_fit_R217_cbmlm'],
    't3_xgb_R217_molformer': ['D0_desc217', 'T3_embed_molformer', 'T3_fit_R217_molformer'],
    't3_xgb_R217_molformerPCA64': ['D0_desc217', 'T3_embed_molformer', 'T3_fit_R217_molformer_pca64'],
    't3_xgb_molformer_only': ['T3_embed_molformer', 'T3_fit_molformer_only'],
    't3_xgb_cbMTR_only': ['T3_embed_cbmtr', 'T3_fit_cbmtr_only'],
    't3_xgb_R217_seed012': ['D0_desc217', 'T3_fit_R217_s012'],
}

# 최종 학습(train 전용, 두 분할, 모든 seed) — ledger 실측 행
def refit_cost(L):
    r = lambda t: L[t]['run_id'].astype(str)
    f = lambda t: L[t]['features'].astype(str)
    c = lambda t, m: float(L[t].loc[m, 'cpu_core_h'].sum())
    T1b = L['T1b']
    lgbF0F3F5 = c('T1b', r('T1b').str.match(r's1_lgb_r1_F0F3F5_(random|scaffold)_s\d$'))
    lgbF0F5 = c('T1b', r('T1b').str.match(r's1_lgb_r1_F0F5_(random|scaffold)_s\d$'))
    h06 = c('T1b', r('T1b').str.match(r'h06_yalk_F0F3F5_huber_(random|scaffold)_s\d$'))
    R = {
        'baseline_xgb600_F0': c('P1', L['P1']['stage'] == '최종'),
        'xgb600_F0_bag10': c('T1', r('T1').str.match(r's1_sq_s\d_(random|scaffold)$')),
        'lgb_r1_F0': c('T1', r('T1').str.startswith('s3_fit_lgb_r1')),
        'xgb_r1_F0': c('T1', r('T1').str.startswith('s3_fit_xgb_r1')),
        'xgb600_F0F1O3': c('T1', r('T1').str.startswith('s2_F0F1O3_s')),
        'lgb_r1_F0F3F5': lgbF0F3F5, 'lgb_r1_F0F5': lgbF0F5, 'h06_yalk_F0F3F5': h06,
        'xgb600_F0F5_M2': c('T8', r('T8').str.match(r'T8_M2_R217F5_xgb_(random|scaffold)_xgb$')),
        'xgb600_F0F3': c('T4', r('T4').str.match(r'T4_val_F0\+F3_(random|scaffold)_s\d+$')),
        'xgb600_F0aux4_scaf': c('T4', r('T4').str.match(r'T4_val_F0\+aux4\(no logS\)_scaffold_s\d+$')),
        'xgb600_F0_patent1x_scaf': c('T5', r('T5').str.match(r'T5_val_add1x_scaffold_s\d+$')),
        'gin_hybrid_seedavg': c('T2', L['T2']['stage'] == 'main'),
        'O2_knn': c('ENS', r('ENS').str.startswith('O2_knn')),
    }
    if 'T3' in L:
        r3 = r('T3')
        R.update({'t3_xgb_R217_cbMLM': c('T3', r3.str.match(r'xgb_R217_cb_mlm_(random|scaffold)_s\d$')),
                  't3_xgb_R217_molformer': c('T3', r3.str.match(r'xgb_R217_molformer_(random|scaffold)_s\d$')),
                  't3_xgb_R217_molformerPCA64': c('T3', r3.str.match(r'xgb_R217_molformer-pca64_(random|scaffold)_s\d$')),
                  't3_xgb_molformer_only': c('T3', r3.str.match(r'xgb_molformer_(random|scaffold)_s\d$')),
                  't3_xgb_cbMTR_only': c('T3', r3.str.match(r'xgb_cb_mtr_(random|scaffold)_s\d$')),
                  't3_xgb_R217_seed012': c('T3', r3.str.match(r'xgb_R217_(random|scaffold)_s[012]$'))})
    return R


# 추론 [s/1k 분자]: 특징 + 모델 × seed 수
FEAT_S1K = {'cbmlm': 10.81, 'cbmtr': 10.84, 'molformer': 143.65, 'F0': 0.118342 * 3600 / 19945 * 1000, 'F3': 3.25701 * 3600 / 19945 * 1000, 'F5': 4.814, 'F1': 0.6508, 'O3': 1.608,
            'graph': 0.026367 * 3600 / 19945 * 1000, 'aux': np.nan, 'morgan_knn': np.nan}
MODEL = {  # (특징 목록, 모델 1회 s/1k, seed 수)
    'baseline_xgb600_F0': (['F0'], 0.0090, 1), 'xgb600_F0_bag10': (['F0'], 0.0069, 10), 'lgb_r1_F0': (['F0'], 0.368, 3),
    'xgb_r1_F0': (['F0'], 0.088, 3), 'xgb600_F0F1O3': (['F0', 'F1', 'O3'], 0.0215, 3), 'lgb_r1_F0F3F5': (['F0', 'F3', 'F5'], 0.3296, 3),
    'lgb_r1_F0F5': (['F0', 'F5'], 0.3348, 3), 'h06_yalk_F0F3F5': (['F0', 'F3', 'F5'], 0.4774, 3), 'xgb600_F0F5_M2': (['F0', 'F5'], 0.0102, 3),
    'xgb600_F0F3': (['F0', 'F3'], 0.0090, 3), 'xgb600_F0aux4_scaf': (['F0', 'aux'], 0.0090, 3), 'xgb600_F0_patent1x_scaf': (['F0'], 0.0090, 3),
    'gin_hybrid_seedavg': (['F0', 'graph'], 0.711, 3), 'O2_knn': (['morgan_knn'], 0.0, 1),
    't3_xgb_R217_cbMLM': (['F0', 'cbmlm'], 0.010, 3), 't3_xgb_R217_molformer': (['F0', 'molformer'], 0.010, 3),
    't3_xgb_R217_molformerPCA64': (['F0', 'molformer'], 0.010, 3), 't3_xgb_molformer_only': (['molformer'], 0.010, 3),
    't3_xgb_cbMTR_only': (['cbmtr'], 0.010, 3), 't3_xgb_R217_seed012': (['F0'], 0.009, 3),
}  # T3 XGB 예측 시간은 ledger 에 없어 기준선 XGB 값(약 0.01 s/1k)으로 가정


def infer_cost(members, feat_s1k):
    fs = sorted({x for m in members for x in MODEL[m][0]})
    feat = sum(feat_s1k[x] for x in fs)
    model = sum(MODEL[m][1] * MODEL[m][2] for m in members)
    return feat, model, '+'.join(fs)


def candidate_costs(cands, feat_s1k, extra_blocks=()):
    """cands: {name: [members]} -> DataFrame"""
    L = load_ledgers(); B = blocks(L); R = refit_cost(L)
    rows = []
    for name, mem in cands.items():
        bl = sorted({b for m in mem for b in LINEAGE[m]} | (set(extra_blocks) if len(mem) > 1 else set()))
        ch = sum(B[b]['core_h'] for b in bl); api = sum(B[b]['api'] for b in bl)
        rf = sum(R[m] for m in mem)
        fe, mo, fs = infer_cost(mem, feat_s1k)
        tot = ch + rf
        rows.append(dict(candidate=name, members=';'.join(mem), n_members=len(mem), blocks=';'.join(bl),
                         search_core_h=ch, final_refit_core_h=rf, total_core_h=tot, total_kwh=tot * P_W / NLOG / 1000,
                         api_usd=api, total_usd=tot * USD_PER_COREH + api, marginal_usd=rf * USD_PER_COREH,
                         infer_feat_s_per_1k=fe, infer_model_s_per_1k=mo, infer_total_s_per_1k=fe + mo, infer_features=fs,
                         needs_gin_retrain=int('gin_hybrid_seedavg' in mem), needs_conformers=int('F3' in fs)))
    return pd.DataFrame(rows), B, R
