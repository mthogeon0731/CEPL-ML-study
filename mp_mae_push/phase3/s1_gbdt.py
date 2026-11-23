"""s1_gbdt.py : 최종 후보·목표 C 멤버(GBDT·kNN 제외 전부)를 train 전용으로 재학습 → validation·test 예측.

라벨은 p3lib.y_masked(how) (분할 how 의 test 행 = NaN) 만 쓴다. test 예측은 y 없이 P3/preds/test_blind/ 에 저장.
멤버 레시피 (2단계 스크립트와 같은 함수·인자):
  lgb_r1_F0F3F5  t1blib.make_model('lgb_r1') 나무 = round(best_iter × n_fit / n_inner) (3,688 / 2,978), F0+F3+F5, seed 0,1,2
  lgb_r1_F0F5    같은 설정, F0+F5 (random 만: 후보 A 멤버)
  h06_yalk_F0F3F5 s2_yalk.fit/predict, huber, a 65/40, b ln 2.435, σ sigma3d_max_log, 나무 4,588 / 3,304, seed 0,1,2
  xgb600_F0F1O3  t1lib.xgb = BASE_XGB, colsample 0.5 (T1 CV 선택값, random), F0+F1+O3, seed 0,1,2 (random 만)
  xgb600_F0_patent1x_scaf  T5 fitlib.fit_predict, train + 특허 무작위 n_train 분자 (표본 seed = 모델 seed), seed 42,43,44 (scaffold 만)
  xgb600_F0F5_M2 T8 make_head('xgb') = BASE_XGB, F0+F5, seed 42,0,1
  baseline_xgb600_F0  BASE_XGB seed 42, F0 (목표 C ens_sd 멤버; sandbox 기준선 재현 점검)
  lgb_r1_F0 / xgb_r1_F0  T1 s3_refit (seed 0,1,2; 나무 3,688/2,978, 7,057/3,489) — 목표 C ens_sd 멤버
XGBoost n_jobs 8, LightGBM num_threads 6.
사용: python P3/scripts/s1_gbdt.py --members a,b,... [--splits random,scaffold] [--fit train|trainval]
"""
import argparse
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import p3lib as P  # noqa: E402
WS = P.WS
sys.path.insert(0, os.path.join(WS, 'T1b_combined', 'scripts'))
import t1blib as L  # noqa: E402
import s2_yalk as Y  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from xgboost import XGBRegressor  # noqa: E402

mc = P.mc
L.LGB_NT = P.LGB_NT
L.XGB_NJ = P.XGB_NJ
CACHE = os.path.join(WS, 'mp_mae_push', 'cache')
N_INNER = {'random': 13561, 'scaffold': 14573}          # T1 내부 85/15 학습 행 수 (train_candidate.py)
H06 = {'random': dict(a=65.0, best_iter=3900), 'scaffold': dict(a=40.0, best_iter=3018)}
H06_N_INNER = {'random': 13561, 'scaffold': 14573}
B_H06 = float(np.log(2.435))


def X_of(fs):
    parts = {'F0': lambda: np.load(os.path.join(CACHE, 'X_desc217.npy')).astype(np.float32),
             'F3': lambda: np.load(os.path.join(CACHE, 'X_phys.npy')).astype(np.float32),
             'F5': lambda: np.load(os.path.join(CACHE, 'F5_rules.npy')).astype(np.float32),
             'F1': lambda: np.load(os.path.join(CACHE, 'X_morgan_count_r2_2048.npy')).astype(np.float32),
             'O3': lambda: np.load(os.path.join(CACHE, 'X_stereo_o3.npy')).astype(np.float32)}
    return np.hstack([parts[p]() for p in fs.split('+')])


def base_xgb(seed, **kw):
    return XGBRegressor(**{**mc.BASE_XGB, **kw}, n_jobs=P.XGB_NJ, random_state=seed)


def patent_pick(seed, n):
    """T5 add1x: 특허 정제본 194,801행에서 비복원 무작위 n (표본 seed = 모델 seed). 재구성 — validation 재현으로 확인."""
    return np.random.default_rng(seed).choice(194801, n, replace=False)


MEMBERS = {
    'lgb_r1_F0F3F5': dict(kind='lgb', fs='F0+F3+F5', seeds=(0, 1, 2), splits=('random', 'scaffold')),
    'lgb_r1_F0F5': dict(kind='lgb', fs='F0+F5', seeds=(0, 1, 2), splits=('random',)),
    'h06_yalk_F0F3F5': dict(kind='h06', fs='F0+F3+F5', seeds=(0, 1, 2), splits=('random', 'scaffold')),
    'xgb600_F0F1O3': dict(kind='xgb600', fs='F0+F1+O3', seeds=(0, 1, 2), splits=('random',), kw=dict(colsample_bytree=0.5)),
    'xgb600_F0_patent1x_scaf': dict(kind='patent', fs='F0', seeds=(42, 43, 44), splits=('scaffold',)),
    'xgb600_F0F5_M2': dict(kind='xgb600', fs='F0+F5', seeds=(42, 0, 1), splits=('random', 'scaffold')),
    'baseline_xgb600_F0': dict(kind='xgb600', fs='F0', seeds=(42,), splits=('random', 'scaffold')),
    'lgb_r1_F0': dict(kind='lgb', fs='F0', seeds=(0, 1, 2), splits=('random', 'scaffold')),
    'xgb_r1_F0': dict(kind='xgbr1', fs='F0', seeds=(0, 1, 2), splits=('random', 'scaffold')),
}


def fit_member(name, how, fit='train'):
    spec = MEMBERS[name]
    tag = name if fit == 'train' else f'{name}__trainval'
    out_t = os.path.join(P.TEST_DIR, f'{tag}_{how}_test.csv')
    if os.path.exists(out_t):
        print('skip (exists)', tag, how, flush=True)
        return
    idx, _ = P.split(how)
    tr = idx['train'] if fit == 'train' else np.concatenate([idx['train'], idx['validate']])
    va, te = idx['validate'], idx['test']
    y = P.y_masked(how)
    assert np.isfinite(y[tr]).all()
    X = X_of(spec['fs'])
    PV, PT, seeds = [], [], list(spec['seeds'])
    for s in seeds:
        rows_fit, n_est, extra = tr, None, {}
        with mc.CostMeter('최종') as cm:
            if spec['kind'] == 'lgb':
                c = L.CFG['lgb_r1']
                n_est = int(round(c['best_iter'][how] * len(tr) / N_INNER[how]))
                m = L.make_model('lgb_r1', s, how, n_estimators=n_est)
                m.fit(X[tr], y[tr])
                pv, pt = m.predict(X[va]), m.predict(X[te])
                params = dict(c['p'], n_estimators=n_est, threads=P.LGB_NT, cfg='lgb_r1')
            elif spec['kind'] == 'xgbr1':
                c = L.CFG['xgb_r1']
                n_est = int(round(c['best_iter'][how] * len(tr) / N_INNER[how]))
                m = L.make_xgb(c['p'], seed=s, n_estimators=n_est)
                m.fit(X[tr], y[tr])
                pv, pt = m.predict(X[va]), m.predict(X[te])
                params = dict(c['p'], n_estimators=n_est, threads=P.XGB_NJ, cfg='xgb_r1')
            elif spec['kind'] == 'xgb600':
                n_est = 600
                kw = spec.get('kw', {})
                m = base_xgb(s, **kw)
                m.fit(X[tr], y[tr])
                pv, pt = m.predict(X[va]), m.predict(X[te])
                params = dict(mc.BASE_XGB, **kw, n_jobs=P.XGB_NJ)
            elif spec['kind'] == 'patent':
                n_est = 600
                XE = np.load(os.path.join(CACHE, 'X_ext_desc217.npy')).astype(np.float32)
                yE = pd.read_csv(os.path.join(P.INP, 'patent_clean.csv'))['value_K'].to_numpy(float)
                n_add = len(idx['train'])
                ei = np.sort(patent_pick(s, n_add))
                Xtr = np.vstack([X[tr], XE[ei]])
                ytr = np.concatenate([y[tr], yE[ei]])
                del XE
                m = base_xgb(s)
                m.fit(Xtr, ytr)
                del Xtr
                pv, pt = m.predict(X[va]), m.predict(X[te])
                rows_fit = np.arange(len(ytr))
                params = dict(mc.BASE_XGB, n_jobs=P.XGB_NJ, patent_add=n_add, pick='default_rng(seed).choice(194801, n, replace=False)')
                extra = dict(n_patent=n_add)
            elif spec['kind'] == 'h06':
                h = H06[how]
                S = Y.entropy(h['a'], B_H06, 'sigma3d_max_log')
                n_est = int(round(h['best_iter'] * len(tr) / H06_N_INNER[how]))
                bst, _, med = Y.fit(X, y, S, tr, s, n_est, 'huber')
                pv, pt = Y.predict(bst, X, S, va, med), Y.predict(bst, X, S, te, med)
                params = dict(a=h['a'], b=B_H06, sig='sigma3d_max_log', n_estimators=n_est, threads=P.LGB_NT, lgb='lgb_r1 설정',
                              objective='Tm-scale huber(δ60)')
        t0 = time.perf_counter()
        _ = m.predict(X[te]) if spec['kind'] != 'h06' else Y.predict(bst, X, S, te, med)
        inf = (time.perf_counter() - t0) / len(te) * 1000
        pv, pt = np.asarray(pv, float), np.asarray(pt, float)
        vm = float(np.mean(np.abs(y[va] - pv))) if fit == 'train' else np.nan
        P.ledger(dict(run_id=f'P3_{tag}_{how}_s{s}', model=spec['kind'], features=spec['fs'], data_variant=fit,
                      objective=params.get('objective', 'sq'), split=how, seed=s, params=params,
                      val_MAE=(round(vm, 4) if fit == 'train' else ''), n_train=len(rows_fit), infer_s_per_1k=round(inf, 5),
                      notes=f'predict=validate+test(라벨 없이 저장) {extra}', **{k: v for k, v in cm.as_row().items()}))
        print(f'{tag} {how} s{s}: n_fit {len(rows_fit)} n_est {n_est} val MAE {vm:.4f} cpu {cm.cpu_core_h:.4f} core-h '
              f'(P3 누적 {P.used_core_h():.3f})', flush=True)
        PV.append(pv); PT.append(pt)
    if fit == 'train':
        P.save_val(tag, how, va, PV, seeds)
    else:
        os.makedirs(P.VAL_DIR, exist_ok=True)
        pd.DataFrame({'row': va, **{f'pred_s{s}': p for s, p in zip(seeds, PV)}}).to_csv(
            os.path.join(P.VAL_DIR, f'{tag}_{how}_val_insample.csv'), index=False)
    P.save_test_blind(tag, how, te, PT, seeds)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--members', required=True)
    ap.add_argument('--splits', default='random,scaffold')
    ap.add_argument('--fit', default='train', choices=['train', 'trainval'])
    ap.add_argument('--budget', type=float, default=P.BUDGET_CORE_H)
    a = ap.parse_args()
    for name in a.members.split(','):
        for how in a.splits.split(','):
            if how not in MEMBERS[name]['splits']:
                continue
            if P.used_core_h() > a.budget:
                print('예산 상한 — 중단', flush=True)
                sys.exit(0)
            fit_member(name, how, a.fit)
    print('DONE P3 누적 core-h', round(P.used_core_h(), 4), flush=True)
