"""s11_aux_eval.py : 보조 결과 — train+validation 재학습 멤버로 만든 후보의 test 지표 (주 결과 저장 뒤 실행).

규칙(명세): 나무 수 = best_iter × n_fit / n_inner (s1_gbdt.py --fit trainval), BASE_XGB 는 600그루 고정, 가중치는 validation 값 고정,
GIN 은 같은 내부 10 % 조기 종료(train+validation 안). 변형:
  'GBDT만 trainval' — GBDT 멤버만 train+validation, GIN 은 train 전용 그대로
  '전부 trainval'   — GIN 도 train+validation (파일이 있을 때만)
멤버 파일이 없으면 그 변형은 '미실시(예산)'으로 표시한다. 먼저 보조 test 예측 파일 sha256 을 기록한 뒤 라벨을 읽는다.
출력: P3/tables/aux_trainval_test.csv, P3/tables/test_pred_sha256_trainval.csv
"""
import datetime
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as P  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

mc = P.mc
SPEC = pd.read_csv(os.path.join(P.INP, 'final_candidates_spec.csv'))
GBDT = {'lgb_r1_F0F3F5', 'lgb_r1_F0F5', 'h06_yalk_F0F3F5', 'xgb600_F0F1O3', 'xgb600_F0_patent1x_scaf', 'xgb600_F0F5_M2'}
MAIN = {'A': 'A_wavg_mse_noO2', 'B': 'B_fixed_mean_best_GIN', 'C': 'C_xgb600_F0F5_M2'}


def member_file(m, how, tv):
    tag = f'{m}__trainval' if tv else m
    if m == 'gin_hybrid_seedavg' and tv:
        tag = 'gin_hybrid_seedavg__trainval'
    return os.path.join(P.TEST_DIR, f'{tag}_{how}_test.csv')


def main():
    aux_files = sorted(f for f in os.listdir(P.TEST_DIR) if '__trainval' in f)
    man = pd.DataFrame([dict(file=f, sha256=P.sha256(os.path.join(P.TEST_DIR, f)),
                             has_y=('y' in pd.read_csv(os.path.join(P.TEST_DIR, f), nrows=1).columns)) for f in aux_files])
    man['recorded_utc'] = datetime.datetime.utcnow().isoformat(timespec='seconds')
    man.to_csv(os.path.join(P.TAB, 'test_pred_sha256_trainval.csv'), index=False)
    assert not man['has_y'].any()
    y_all = pd.read_csv(os.path.join(mc.WEEK2, 'data', 'mp.csv'))['value'].to_numpy(float)   # 보조 평가에서 라벨을 읽는 곳
    rows = []
    for cid in ('C', 'B', 'A'):
        for how in P.SPLITS:
            te = P.split(how)[0]['test']
            y, keep = y_all[te], P.dedup_keep(how, 'test')
            w = SPEC[(SPEC.cand_id == cid) & (SPEC.split == how)].set_index('member')['weight'].astype(float)
            w = w / w.sum()
            main_p = pd.read_csv(os.path.join(P.TEST_DIR, f'{MAIN[cid]}_{how}_test.csv')).set_index('row').loc[te, 'pred'].to_numpy()
            variants = {'GBDT만 trainval': {m: (m in GBDT) for m in w.index}}
            if 'gin_hybrid_seedavg' in w.index:
                variants['전부 trainval'] = {m: True for m in w.index}
            for vn, tvmap in variants.items():
                files = {m: member_file(m, how, tvmap[m]) for m in w.index}
                missing = [os.path.basename(f) for f in files.values() if not os.path.exists(f)]
                if missing:
                    rows.append(dict(cand=cid, split=how, variant=vn, status='미실시(예산)', missing=';'.join(missing)))
                    continue
                p = sum(w[m] * pd.read_csv(f).set_index('row').loc[te, 'pred'].to_numpy(float) for m, f in files.items())
                r = dict(cand=cid, split=how, variant=vn, status='실시', members_trainval=';'.join(m for m in w.index if tvmap[m]))
                for basis, mk in (('전체', np.ones(len(y), bool)), ('중복제거', keep)):
                    met = mc.metrics(y[mk], p[mk])
                    d, lo, hi = mc.paired_bootstrap(y[mk], p[mk], main_p[mk])
                    r.update({f'MAE_{basis}': met['MAE'], f'RMSE_{basis}': met['RMSE'], f'R2_{basis}': met['R2'], f'n_{basis}': int(mk.sum()),
                              f'MAE_main_{basis}': float(np.mean(np.abs(main_p[mk] - y[mk]))),
                              f'd_vs_main_{basis}': d, f'd_vs_main_lo_{basis}': lo, f'd_vs_main_hi_{basis}': hi})
                    if how == 'random':
                        o = pd.read_csv(os.path.join(P.INP, 'base', 'orig_random_test_pred.csv'))['pred'].to_numpy(float)
                        d2, lo2, hi2 = mc.paired_bootstrap(y[mk], p[mk], o[mk])
                        r.update({f'd_vs_orig_{basis}': d2, f'd_vs_orig_lo_{basis}': lo2, f'd_vs_orig_hi_{basis}': hi2})
                rows.append(r)
                out = pd.DataFrame({'row': te, 'y': y, 'pred': p})
                vtag = 'gbdtTV' if vn.startswith('GBDT') else 'allTV'
                os.makedirs(os.path.join(P.P3, 'preds', 'test'), exist_ok=True)
                out.to_csv(os.path.join(P.P3, 'preds', 'test', f'{MAIN[cid]}__{vtag}_{how}_test.csv'), index=False)
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(P.TAB, 'aux_trainval_test.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        print(d.round(3).to_string())


if __name__ == '__main__':
    with mc.CostMeter('평가') as cm:
        main()
    P.ledger(dict(run_id='P3_s11_aux_eval', model='-', data_variant='보조 train+validation test 평가', split='both',
                  notes='보조 평가', **cm.as_row()))
