"""s4_goalc.py : 목표 C 위험 점수 metaA_trainOOF_self 를 validation(재현 점검)과 test(라벨 없이)에 적용.

입력 정의는 ENS/s4_risk.py features() 와 같다:
  ens_sd  = OOF 멤버 예측의 SD(ddof 1) — random 6개(baseline, lgb_r1_F0, xgb_r1_F0, lgb_r1_F0F3F5, h06, M2), scaffold 5개(M2 제외)
  knn_gap = |lgb_r1_F0F3F5 − O2 kNN|;  nn1·n07·nn5mean = train 만 참조한 유사도 (s2_knn.py);  n_meas = 반복측정 수(NaN→1)
  pred    = lgb_r1_F0F3F5 예측;  heavy = 무거운 원자 수 (mol_flags)
메타모델은 ENS 저장 파일(LightGBM booster text)을 그대로 불러 쓴다(재적합하지 않음).
출력: P3/tables/repro_goalC_val.csv, P3/preds/val/goalC_score_<how>_val.csv, P3/preds/test_blind/goalC_score_<how>_test.csv
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as P  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import lightgbm as lgb  # noqa: E402

FEATS = ['ens_sd', 'knn_gap', 'nn1', 'n07', 'nn5mean', 'n_meas', 'pred', 'heavy']
OOF = {'random': ['baseline_xgb600_F0', 'lgb_r1_F0', 'xgb_r1_F0', 'lgb_r1_F0F3F5', 'h06_yalk_F0F3F5', 'xgb600_F0F5_M2'],
       'scaffold': ['baseline_xgb600_F0', 'lgb_r1_F0', 'xgb_r1_F0', 'lgb_r1_F0F3F5', 'h06_yalk_F0F3F5']}
BEST = 'lgb_r1_F0F3F5'


def load(member, how, which, rows):
    d = P.VAL_DIR if which == 'val' else P.TEST_DIR
    f = pd.read_csv(os.path.join(d, f'{member}_{how}_{which}.csv')).set_index('row').loc[rows]
    if which == 'test':
        assert 'y' not in f.columns
    return f['pred'].to_numpy(float)


def features(how, which):
    idx, _ = P.split(how)
    rows = idx['validate' if which == 'val' else 'test']
    M = np.column_stack([load(m, how, which, rows) for m in OOF[how]])
    best = load(BEST, how, which, rows)
    o2 = load('O2_knn', how, which, rows)
    sim = pd.read_csv(os.path.join(P.P3, 'work', f'sim_{how}.csv'))
    sim = sim[sim['set'] == ('validate' if which == 'val' else 'test')].set_index('row').loc[rows]
    rs = pd.read_csv(os.path.join(P.INP, 'ens', 'mp_replicate_stats.csv'), usecols=['row', 'n_meas']).set_index('row')
    fl = pd.read_csv(os.path.join(P.INP, 'mol_flags.csv'), usecols=['row', 'heavy']).set_index('row')
    X = pd.DataFrame({'ens_sd': M.std(1, ddof=1), 'knn_gap': np.abs(best - o2), 'nn1': sim['nn1'].to_numpy(),
                      'n07': sim['n07'].to_numpy(), 'nn5mean': sim['nn5mean'].to_numpy(),
                      'n_meas': rs.loc[rows, 'n_meas'].fillna(1).to_numpy(), 'pred': best, 'heavy': fl.loc[rows, 'heavy'].to_numpy()})
    return rows, X


if __name__ == '__main__':
    rep = []
    for how in P.SPLITS:
        bst = lgb.Booster(model_file=os.path.join(P.INP, 'ens', f'meta_{how}.txt'))
        # (a) 저장 입력 → 저장 점수 (booster 파일 확인)
        st = pd.read_csv(os.path.join(P.INP, 'ens', f'scores_A_{how}.csv'))
        s_from_stored = bst.predict(st[[f'feat_{c}' for c in FEATS]].rename(columns=lambda c: c[5:]))
        rows_v, Xv = features(how, 'val')
        st = st.set_index('row').loc[rows_v]
        sv = bst.predict(Xv[FEATS])
        r = dict(split=how, n_val=len(rows_v),
                 booster_check_max_abs_diff=float(np.abs(s_from_stored - pd.read_csv(os.path.join(P.INP, 'ens', f'scores_A_{how}.csv'))['score_metaA_trainOOF_self']).max()),
                 score_refit_vs_stored_max_abs_diff=float(np.abs(sv - st['score_metaA_trainOOF_self'].to_numpy()).max()),
                 score_refit_vs_stored_corr=float(np.corrcoef(sv, st['score_metaA_trainOOF_self'])[0, 1]),
                 score_spearman=float(pd.Series(sv).corr(pd.Series(st['score_metaA_trainOOF_self'].to_numpy()), method='spearman')))
        for c in FEATS:
            r[f'feat_{c}_max_abs_diff'] = float(np.abs(Xv[c].to_numpy() - st[f'feat_{c}'].to_numpy()).max())
        rep.append(r)
        os.makedirs(P.VAL_DIR, exist_ok=True)
        pd.DataFrame({'row': rows_v, 'score': sv, 'score_stored': st['score_metaA_trainOOF_self'].to_numpy(), **Xv}).to_csv(
            os.path.join(P.VAL_DIR, f'goalC_score_{how}_val.csv'), index=False)
        rows_t, Xt = features(how, 'test')
        stt = bst.predict(Xt[FEATS])
        pd.DataFrame({'row': rows_t, 'score': stt, **Xt}).to_csv(os.path.join(P.TEST_DIR, f'goalC_score_{how}_test.csv'), index=False)
        print(r, flush=True)
    pd.DataFrame(rep).to_csv(os.path.join(P.TAB, 'repro_goalC_val.csv'), index=False)
