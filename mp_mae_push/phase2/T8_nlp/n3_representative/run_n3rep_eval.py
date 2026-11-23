"""run_n3rep_eval.py : N3 보강(대표 표본) 비교 — random 분할만, validation 전체(n 1,995)와 중복 제거(n 1,922).

학습 집합 (i) ALL = N3 있는 train 전부(기존 비대표 3,325 + 새 무작위), (ii) NEW = 새 무작위 표본만.
각 학습 집합에서 같은 분자로 R217, M2(R217+F5), N2/N3 × (MiniLM, bge-small), R217+N3, R217+N2(대조)를 적합.
head: XGBoost(mc.BASE_XGB, n_jobs=8, seed 0,1,2 예측 평균), Ridge(t8eval.RidgeHead, alpha 는 train 안 5-fold CV).
test 행은 쓰지 않는다. 출력은 T8_nlp_n3rep/ 아래.
"""
import os
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('mp_mae_push/common'))
sys.path.insert(0, os.path.abspath('T8_nlp_n3rep'))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import mpcommon as mc  # noqa: E402

D = 'T8_nlp_n3rep'
LEDGER = os.path.join(D, 'ledger.csv')
PRED = os.path.join(D, 'preds')
os.makedirs(os.path.join(PRED, 'seeds'), exist_ok=True)
SEEDS = (0, 1, 2)

y = pd.read_csv(os.path.join(mc.WEEK2, 'data', 'mp.csv'))['value'].to_numpy(float)
N = len(y)
idx, _ = mc.load_split('random')
dd_tab = pd.read_csv('mp_mae_push/phase2/prereg/dedup_rows_random.csv')
DEDUP = set(dd_tab.loc[(dd_tab['set'] == 'validate') & (dd_tab['overlap'] == 0), 'row'])


def ridge_head():
    # t8eval.RidgeHead 와 같은 전처리·CV (원본 모듈을 그대로 불러 씀)
    import importlib.util
    spec = importlib.util.spec_from_file_location('t8eval_ref', os.path.join(D, 'in_t8eval.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.RidgeHead()


def full(A, rows):
    X = np.full((N, A.shape[1]), np.nan, np.float32)
    X[rows] = A
    return X


old_rows = pd.read_csv(os.path.join(D, 'in_n3_texts_covered.csv'))['row'].to_numpy()
new_tab = pd.read_csv(os.path.join(D, 'n3rep_texts.csv'))
new_rows = new_tab['row'].to_numpy()
N3 = {
    'minilm': full(np.vstack([np.load(os.path.join(D, 'in_n3cov_minilm.npy')), np.load(os.path.join(D, 'n3rep_minilm.npy'))]),
                   np.r_[old_rows, new_rows]),
    'bge': full(np.vstack([np.load(os.path.join(D, 'in_n3cov_bge.npy')), np.load(os.path.join(D, 'n3rep_bge.npy'))]),
                np.r_[old_rows, new_rows]),
}
X0 = np.load('mp_mae_push/cache/X_desc217.npy').astype(np.float32)
F5 = np.load(os.path.join(D, 'in_F5_rules.npy')).astype(np.float32)
REP = {
    'R217': X0,
    'M2_R217F5': np.hstack([X0, F5]),
    'N2_minilm': np.load(os.path.join(D, 'in_n2_minilm.npy')).astype(np.float32),
    'N2_bge': np.load(os.path.join(D, 'in_n2_bge.npy')).astype(np.float32),
    'N3_minilm': N3['minilm'],
    'N3_bge': N3['bge'],
}
REP['C_R217N3minilm'] = np.hstack([X0, REP['N3_minilm']])
REP['C_R217N3bge'] = np.hstack([X0, REP['N3_bge']])
REP['C_R217N2bge'] = np.hstack([X0, REP['N2_bge']])

has3 = set(np.r_[old_rows, new_rows].tolist())
tr_all = np.array([r for r in idx['train'] if r in has3])      # load_split 순서 유지
newset = set(new_rows.tolist())
tr_new = np.array([r for r in idx['train'] if r in newset])
va = np.array([r for r in idx['validate'] if r in has3])
TRAINSETS = {'ALL': tr_all, 'NEW': tr_new}
print('train ALL', len(tr_all), 'NEW', len(tr_new), 'val', len(va), 'val dedup', sum(r in DEDUP for r in va), flush=True)

API = pd.read_csv(os.path.join(D, 'n3rep_api_summary.csv')).iloc[0]

for ts, tr in TRAINSETS.items():
    for rep, X in REP.items():
        for head in ('xgb', 'ridge'):
            tag = f'{rep}_{head}_{ts}'
            if os.path.exists(os.path.join(PRED, f'{tag}_val.csv')):
                continue
            assert not np.isnan(X[np.r_[tr, va]][:, -1]).any() or rep in ('R217', 'M2_R217F5')
            preds, maes, alphas = [], [], []
            with mc.CostMeter('탐색') as c:
                for s in (SEEDS if head == 'xgb' else SEEDS[:1]):
                    if head == 'xgb':
                        from xgboost import XGBRegressor
                        m = XGBRegressor(**mc.BASE_XGB, n_jobs=8, random_state=s)
                    else:
                        m = ridge_head()
                    m.fit(X[tr], y[tr])
                    t0 = time.perf_counter()
                    p = np.asarray(m.predict(X[va]), float)
                    inf = (time.perf_counter() - t0) / len(va) * 1000
                    preds.append(p)
                    maes.append(float(np.mean(np.abs(y[va] - p))))
                    if head == 'ridge':
                        alphas.append(m.alpha_)
            P = np.mean(preds, axis=0)
            dd = np.array([r in DEDUP for r in va])
            met, met_dd = mc.metrics(y[va], P), mc.metrics(y[va][dd], P[dd])
            pd.DataFrame({'row': va, 'y': y[va], 'pred': P}).to_csv(os.path.join(PRED, f'{tag}_val.csv'), index=False)
            pd.DataFrame({'row': va, **{f'pred_s{i}': p for i, p in enumerate(preds)}}).to_csv(
                os.path.join(PRED, 'seeds', f'{tag}_val_seeds.csv'), index=False)
            is_n3 = 'N3' in rep
            mc.append_ledger(LEDGER, dict(
                run_id=f'T8n3rep_{tag}', track='T8_nlp_n3rep', model='XGBoost' if head == 'xgb' else 'Ridge',
                features=rep,
                data_variant=(f'train N3 보유 전체 {len(tr)} (기존 비대표 + 새 무작위)' if ts == 'ALL'
                              else f'train 새 무작위 표본 {len(tr)}'),
                objective='MAE(validation)', split='random', seed=','.join(map(str, SEEDS if head == 'xgb' else SEEDS[:1])),
                params=('BASE_XGB n_jobs=8' if head == 'xgb' else f'RidgeCV 5-fold(train) alpha={np.round(alphas, 3).tolist()}'),
                val_MAE=round(met['MAE'], 4), val_RMSE=round(met['RMSE'], 4), n_train=len(tr),
                api_tokens=0, api_cost=0,
                infer_s_per_1k=round(inf, 4),
                notes=(f'validation n {len(va)}; 중복제거 MAE {met_dd["MAE"]:.3f} (n {int(dd.sum())}); '
                       f'시드별 {np.mean(maes):.3f}±{(np.std(maes, ddof=1) if len(maes) > 1 else 0):.3f}'
                       + ('; N3 생성 API 비용은 T8n3rep_generation 행에 한 번만 기록(중복 합산 방지)' if is_n3 else '')),
                **c.as_row()))
            print(f'{tag:28s} MAE {met["MAE"]:.3f} dd {met_dd["MAE"]:.3f} wall {c.wall_s:.0f}s', flush=True)
print('DONE', flush=True)
