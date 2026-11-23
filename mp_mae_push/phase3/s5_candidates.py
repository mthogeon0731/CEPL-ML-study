"""s5_candidates.py : 후보 A·B·C (+ 대체안 B_alt = lgb_r1_F0F3F5 단일) test 예측을 멤버 test 예측과 고정 가중치로 조립,
validation 도 같은 가중치로 조립(재현 점검용, in-sample), test_blind 모든 파일의 sha256 기록. test 라벨은 읽지 않는다.

가중치는 final_candidates_spec.csv 그대로; 합이 1 에서 벗어나면 합으로 나눈다(명세 규칙; A random 합 1.0001).
사용: python P3/scripts/s5_candidates.py [--fit train|trainval]
"""
import datetime
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as P  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CAND_NAME = {'A': 'A_wavg_mse_noO2', 'B': 'B_fixed_mean_best_GIN', 'C': 'C_xgb600_F0F5_M2', 'B_alt': 'Balt_lgb_r1_F0F3F5'}


def weights():
    sp = pd.read_csv(os.path.join(P.INP, 'final_candidates_spec.csv'))
    W = {}
    for _, r in sp.iterrows():
        W.setdefault((r['cand_id'], r['split']), {})[r['member']] = float(r['weight'])
    for how in P.SPLITS:
        W[('B_alt', how)] = {'lgb_r1_F0F3F5': 1.0}
    return W


def build(fit='train'):
    W = weights()
    sfx = '' if fit == 'train' else '__trainval'
    rows = []
    for (cid, how), w in sorted(W.items()):
        tot = sum(w.values())
        wn = {m: v / tot for m, v in w.items()}
        idx, _ = P.split(how)
        te, va = idx['test'], idx['validate']
        T = []
        for m, v in wn.items():
            t = pd.read_csv(os.path.join(P.TEST_DIR, f'{m}{sfx}_{how}_test.csv')).set_index('row').loc[te]
            assert 'y' not in t.columns
            T.append(v * t['pred'].to_numpy(float))
        P.save_test_blind(CAND_NAME[cid] + sfx, how, te, [np.sum(T, 0)], ['w'])
        if fit == 'train':
            V = [v * pd.read_csv(os.path.join(P.VAL_DIR, f'{m}_{how}_val.csv')).set_index('row').loc[va, 'pred'].to_numpy(float)
                 for m, v in wn.items()]
            P.save_val(CAND_NAME[cid], how, va, [np.sum(V, 0)], ['w'])
        rows.append(dict(cand=cid, name=CAND_NAME[cid] + sfx, split=how, weight_sum_spec=tot,
                         weights_normalized=';'.join(f'{m}={v:.6f}' for m, v in wn.items())))
    pd.DataFrame(rows).to_csv(os.path.join(P.TAB, f'candidate_weights_used{sfx}.csv'), index=False)


def sha_manifest(tag):
    out = []
    for f in sorted(os.listdir(P.TEST_DIR)):
        p = os.path.join(P.TEST_DIR, f)
        d = pd.read_csv(p)
        out.append(dict(file=f, sha256=P.sha256(p), n_rows=len(d), cols=','.join(d.columns), has_y=('y' in d.columns),
                        bytes=os.path.getsize(p)))
    m = pd.DataFrame(out)
    m['recorded_utc'] = datetime.datetime.utcnow().isoformat(timespec='seconds')
    m.to_csv(os.path.join(P.TAB, f'test_pred_sha256_{tag}.csv'), index=False)
    assert not m['has_y'].any()
    return m


if __name__ == '__main__':
    fit = sys.argv[sys.argv.index('--fit') + 1] if '--fit' in sys.argv else 'train'
    build(fit)
    m = sha_manifest('main' if fit == 'train' else 'trainval')
    print(m[['file', 'sha256', 'n_rows']].to_string())
