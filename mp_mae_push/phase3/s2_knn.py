"""s2_knn.py : O2 Tanimoto kNN 의 validation(재현 점검)·test 예측과 유사도 특징 (nn1, n07, nn5mean).

ENS/s1_knn.py 의 fps_for / topk_sims / knn_pred 를 그대로 쓴다. 이웃은 **train 행만** 참조 (validation 행 미참조).
k·가중은 ENS 선택값 고정 (random k 10 sim⁴, scaffold k 20 sim⁴). 라벨은 y_masked(train 행만 사용).
출력: P3/preds/val/O2_knn_<how>_val.csv, P3/preds/test_blind/O2_knn_<how>_test.csv, P3/work/sim_<how>.csv (validate·test 행)
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as P  # noqa: E402
sys.path.insert(0, os.path.join(P.INP, 'ens'))
import s1_knn as K  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CHOSEN = {'random': (10, 4), 'scaffold': (20, 4)}

if __name__ == '__main__':
    _, sdf = P.split('random')
    smiles = sdf.sort_values('row')['smiles'].to_numpy()
    rep = []
    with P.mc.CostMeter('최종') as cm:
        for how in P.SPLITS:
            idx, _ = P.split(how)
            tr, va, te = idx['train'], idx['validate'], idx['test']
            y = P.y_masked(how)
            Xtr, Xva, Xte = K.fps_for(tr, smiles), K.fps_for(va, smiles), K.fps_for(te, smiles)
            k, pw = CHOSEN[how]
            Iv, Sv, n07v = K.topk_sims(Xva, Xtr)
            It, St, n07t = K.topk_sims(Xte, Xtr)
            pv, pt = K.knn_pred(Iv, Sv, y[tr], k, pw), K.knn_pred(It, St, y[tr], k, pw)
            P.save_val('O2_knn', how, va, [pv], [0])
            P.save_test_blind('O2_knn', how, te, [pt], [0])
            sim = pd.concat([pd.DataFrame({'row': va, 'set': 'validate', 'nn1': Sv[:, 0], 'n07': n07v, 'nn5mean': Sv[:, :5].mean(1)}),
                             pd.DataFrame({'row': te, 'set': 'test', 'nn1': St[:, 0], 'n07': n07t, 'nn5mean': St[:, :5].mean(1)})])
            sim.to_csv(os.path.join(P.P3, 'work', f'sim_{how}.csv'), index=False)
            # 재현 점검: ENS 저장 validation 예측·유사도
            o2 = pd.read_csv(os.path.join(P.INP, 'ens', f'O2_val_{how}.csv')).set_index('row').loc[va, 'pred'].to_numpy()
            sv = pd.read_csv(os.path.join(P.INP, 'ens', f'sim_{how}.csv'))
            sv = sv[sv['set'] == 'validate'].set_index('row').loc[va]
            rep.append(dict(member='O2_knn', split=how, n=len(va), max_abs_diff_pred=float(np.abs(pv - o2).max()),
                            max_abs_diff_nn1=float(np.abs(Sv[:, 0] - sv['nn1_self'].to_numpy()).max()),
                            max_abs_diff_n07=float(np.abs(n07v - sv['n07_self'].to_numpy()).max()),
                            max_abs_diff_nn5mean=float(np.abs(Sv[:, :5].mean(1) - sv['nn5mean_self'].to_numpy()).max()),
                            test_nn1_median=float(np.median(St[:, 0])), val_nn1_median=float(np.median(Sv[:, 0]))))
            print(rep[-1], flush=True)
    pd.DataFrame(rep).to_csv(os.path.join(P.TAB, 'repro_knn_sim.csv'), index=False)
    P.ledger(dict(run_id='P3_O2_knn_sim', model='Tanimoto kNN', features='Morgan r2 2048 bit', data_variant='train 참조',
                  split='both', params=dict(chosen=CHOSEN), n_train='15955 (random) / 15956 (scaffold)',
                  notes='validation 재현 + test 예측·유사도 특징 (train 만 참조)', **cm.as_row()))
