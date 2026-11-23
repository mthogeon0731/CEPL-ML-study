"""s3_repro.py : (1) GIN seed 별 출력 → 멤버 파일(seed 평균) 조립, (2) 멤버 validation 재현 점검표.

비교 대상 = 2단계 저장 validation 예측(ENS members.csv 의 val 경로). 지표: seed 평균 예측의 최대 |차이|, MAE(재학습) − MAE(저장),
상관, seed 별 비교(저장본에 seed 열이 있을 때). GIN 기준: |ΔMAE| ≤ 0.1 K 이고 상관 ≥ 0.995 (명세).
출력: P3/tables/repro_val.csv, P3/preds/val/gin_hybrid_seedavg_<how>_val.csv, P3/preds/test_blind/gin_hybrid_seedavg_<how>_test.csv
"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as P  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

GIN_WORK = os.path.join(P.P3, 'work', 'gin')
MEMBERS = {'random': ['lgb_r1_F0F3F5', 'lgb_r1_F0F5', 'h06_yalk_F0F3F5', 'xgb600_F0F1O3', 'xgb600_F0F5_M2', 'baseline_xgb600_F0',
                      'lgb_r1_F0', 'xgb_r1_F0', 'gin_hybrid_seedavg'],
           'scaffold': ['lgb_r1_F0F3F5', 'h06_yalk_F0F3F5', 'xgb600_F0_patent1x_scaf', 'xgb600_F0F5_M2', 'baseline_xgb600_F0',
                        'lgb_r1_F0', 'xgb_r1_F0', 'gin_hybrid_seedavg']}


def assemble_gin(how, fit='train', seeds=(0, 1, 2)):
    tag = 'gin_hybrid_seedavg' if fit == 'train' else 'gin_hybrid_seedavg__trainval'
    idx, _ = P.split(how)
    va, te = idx['validate'], idx['test']
    V, T = [], []
    for s in seeds:
        v = pd.read_csv(os.path.join(GIN_WORK, f'P3_gin_{how}_{fit}_s{s}_val.csv')).set_index('row').loc[va]
        t = pd.read_csv(os.path.join(GIN_WORK, f'P3_gin_{how}_{fit}_s{s}_test.csv')).set_index('row').loc[te]
        assert 'y' not in t.columns
        V.append(v['pred'].to_numpy(float)); T.append(t['pred'].to_numpy(float))
    if fit == 'train':
        P.save_val(tag, how, va, V, seeds)
    P.save_test_blind(tag, how, te, T, seeds)


def main():
    ref = json.load(open(os.path.join(P.P3, 'notes', 'ref_val_paths.json')))
    rows = []
    for how in P.SPLITS:
        idx, _ = P.split(how)
        va = idx['validate']
        y = P.y_masked(how)[va]
        for m in MEMBERS[how]:
            mine = pd.read_csv(os.path.join(P.VAL_DIR, f'{m}_{how}_val.csv')).set_index('row').loc[va]
            rf = pd.read_csv(ref[f'{m}|{how}']).set_index('row').loc[va]
            assert np.abs(rf['y'].to_numpy(float) - y).max() < 1e-6
            a, b = mine['pred'].to_numpy(float), rf['pred'].to_numpy(float)
            r = dict(member=m, split=how, n=len(va), MAE_refit=float(np.mean(np.abs(y - a))), MAE_stored=float(np.mean(np.abs(y - b))),
                     max_abs_diff=float(np.abs(a - b).max()), mean_abs_diff=float(np.abs(a - b).mean()),
                     corr=float(np.corrcoef(a, b)[0, 1]))
            r['dMAE'] = r['MAE_refit'] - r['MAE_stored']
            sc = [c for c in mine.columns if c.startswith('pred_s')]
            r['seeds'] = ','.join(c[6:] for c in sc)
            r['seed_MAE_refit'] = ';'.join(f'{np.mean(np.abs(y - mine[c].to_numpy())):.4f}' for c in sc)
            if all(c in rf.columns for c in sc):
                r['seed_MAE_stored'] = ';'.join(f'{np.mean(np.abs(y - rf[c].to_numpy())):.4f}' for c in sc)
                r['seed_max_abs_diff'] = ';'.join(f'{np.abs(mine[c] - rf[c]).max():.4g}' for c in sc)
                r['seed_corr'] = ';'.join(f'{np.corrcoef(mine[c], rf[c])[0, 1]:.5f}' for c in sc)
            if m == 'gin_hybrid_seedavg':
                r['spec_pass(|dMAE|<=0.1 & corr>=0.995)'] = bool(abs(r['dMAE']) <= 0.1 and r['corr'] >= 0.995)
            else:
                r['identical(max_abs_diff<1e-3)'] = bool(r['max_abs_diff'] < 1e-3)
            rows.append(r)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(P.TAB, 'repro_val.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(d.drop(columns=[c for c in d.columns if c.startswith('seed_')]).round(4).to_string())
        print(d[['member', 'split'] + [c for c in d.columns if c.startswith('seed_')]].to_string())


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--assemble-gin':
        fit = sys.argv[2] if len(sys.argv) > 2 else 'train'
        for how in P.SPLITS:
            assemble_gin(how, fit)
        print('assembled', fit)
    else:
        main()
