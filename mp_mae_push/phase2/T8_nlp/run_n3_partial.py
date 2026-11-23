"""run_n3_partial.py : N3 서술이 확보된 분자(4,044개, mp.csv 앞쪽 행 + 파일럿 500)만으로 N3 대 N2 대 RDKit217을 같은 행·같은 head로 비교한다.

host.llm 프레임당 토큰 한도(2.0M) 때문에 N3 전체가 생성되지 않아 만든 잠정 분석이다.
train = 확보분 ∩ train, validation = 확보분 ∩ validation (두 분할 각각). test 는 쓰지 않는다.
출력: T8_nlp/preds_n3sub/<name>_<split>_val.csv (row,y,pred), seeds/..., ledger 행.
"""
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('T8_nlp'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import t8eval as te  # noqa: E402

mc = te.mc
OUT = 'T8_nlp/preds_n3sub'
os.makedirs(os.path.join(OUT, 'seeds'), exist_ok=True)
y = te.y_all()
N = len(y)
cov = pd.read_csv('T8_nlp/n3_texts_covered.csv')['row'].to_numpy()


def full(path_or_arr, rows=None):
    A = np.load(path_or_arr).astype(np.float32) if isinstance(path_or_arr, str) else path_or_arr
    if rows is None:
        return A
    X = np.full((N, A.shape[1]), np.nan, np.float32)
    X[rows] = A
    return X


X0 = full('mp_mae_push/cache/X_desc217.npy')
F5 = full('T8_nlp/features/F5_rules.npy')
REP = {
    'R217': X0,
    'M2_R217F5': np.hstack([X0, F5]),
    'N2_minilm': full('T8_nlp/emb/n2_minilm.npy'),
    'N2_bge': full('T8_nlp/emb/n2_bge.npy'),
    'N3_minilm': full('T8_nlp/emb/n3cov_minilm.npy', cov),
    'N3_bge': full('T8_nlp/emb/n3cov_bge.npy', cov),
}
REP['C_R217N3bge'] = np.hstack([X0, REP['N3_bge']])
REP['C_R217N3minilm'] = np.hstack([X0, REP['N3_minilm']])
REP['C_R217N2bge'] = np.hstack([X0, REP['N2_bge']])
CFG = [(k, ('xgb', 'ridge')) for k in ('R217', 'N2_minilm', 'N2_bge', 'N3_minilm', 'N3_bge')] + \
      [(k, ('xgb',)) for k in ('M2_R217F5', 'C_R217N3bge', 'C_R217N3minilm', 'C_R217N2bge')]
covset = set(cov.tolist())
for how in ('random', 'scaffold'):
    idx = te.split(how)
    tr = np.array([r for r in idx['train'] if r in covset])
    va = np.array([r for r in idx['validate'] if r in covset])
    dd = te.dedup_mask(how, va)
    for rep, heads in CFG:
        X = REP[rep]
        for head in heads:
            for n in (None, 1000):
                if n is not None and rep not in ('R217', 'N2_minilm', 'N2_bge', 'N3_minilm', 'N3_bge'):
                    continue
                tag = f'N3sub_{rep}_{head}_{how}' + ('' if n is None else f'_n{n}')
                if os.path.exists(os.path.join(OUT, 'seeds', f'{tag}_val_seeds.csv')):
                    continue
                if n is None:
                    plan = [(s, tr) for s in (te.XGB_SEEDS if head == 'xgb' else te.XGB_SEEDS[:1])]
                else:
                    plan = [(te.XGB_SEEDS[k], np.sort(np.random.default_rng(ss).choice(tr, n, replace=False)))
                            for k, ss in enumerate(te.SUB_SEEDS)]
                preds = []
                with mc.CostMeter('탐색') as c:
                    for seed, rows in plan:
                        m = te.make_head(head, seed).fit(X[rows], y[rows])
                        preds.append(np.asarray(m.predict(X[va]), float))
                P = np.mean(preds, axis=0)
                per = [float(np.mean(np.abs(y[va] - p))) for p in preds]
                if n is None:
                    pd.DataFrame({'row': va, 'y': y[va], 'pred': P}).to_csv(os.path.join(OUT, f'{tag}_val.csv'), index=False)
                pd.DataFrame({'row': va, **{f'pred_s{s}': p for (s, _), p in zip(plan, preds)}}).to_csv(
                    os.path.join(OUT, 'seeds', f'{tag}_val_seeds.csv'), index=False)
                mae = float(np.mean(np.abs(y[va] - P)))
                mae_dd = float(np.mean(np.abs(y[va][dd] - P[dd])))
                mc.append_ledger(te.LEDGER, dict(
                    run_id=f'T8_{tag}', track=te.TRACK, model='XGBoost' if head == 'xgb' else 'Ridge', features=rep,
                    data_variant=f'N3 확보분 부분집합 train {len(plan[0][1])}' + ('' if n is None else ' (무작위 부분표본)'),
                    objective='MAE(validation 확보분)', split=how, seed=','.join(str(s) for s, _ in plan),
                    params='BASE_XGB n_jobs=8' if head == 'xgb' else 'RidgeCV 5-fold(train)', val_MAE=round(mae, 4),
                    n_train=len(plan[0][1]),
                    notes=(f'잠정(N3 확보분만): validation n {len(va)}, 중복제거 MAE {mae_dd:.3f} (n {int(dd.sum())}); '
                           f'시드별 {np.mean(per):.3f}±{(np.std(per, ddof=1) if len(per) > 1 else 0):.3f}. 전체 validation 값과 비교 불가'),
                    **c.as_row()))
                print(f'{tag:45s} MAE {mae:.3f} seeds {np.mean(per):.3f} n_val {len(va)} n_tr {len(plan[0][1])} cpu {c.cpu_core_h:.3f}', flush=True)
print('DONE', flush=True)
