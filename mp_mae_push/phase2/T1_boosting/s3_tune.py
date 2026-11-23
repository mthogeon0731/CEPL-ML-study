"""3단계: XGBoost / LightGBM Optuna TPE 튜닝 (H01).
목적함수 = train 안 고정 내부 분할(85/15, seed 42; scaffold 는 Murcko 뼈대 그룹 단위)의 내부 15 % MAE.
조기 종료도 내부 15 % (200 라운드, eval_metric MAE). validation 은 쓰지 않는다.
사용: python s3_tune.py --lib xgb --split random --features F0+F1 --trials 50 --budget 8
이어 하기: SQLite(T1_boosting/optuna.db) study 를 불러와 완료 trial 수가 목표에 이를 때까지 진행."""
import argparse, json, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import t1lib as L

ap = argparse.ArgumentParser()
ap.add_argument('--lib', required=True)
ap.add_argument('--split', default='random')
ap.add_argument('--features', default='F0')
ap.add_argument('--trials', type=int, default=50)
ap.add_argument('--tag', default='', help='study 이름 접미사')
ap.add_argument('--budget', type=float, default=8.0, help='이 study 의 CPU 상한 [core-h]; 넘으면 중단')
args = None  # __main__ 에서만 파싱 (import 가능)

C = os.path.join(L.WS, 'mp_mae_push', 'cache')
y = L.y_all()


def feats(fs):
    parts = {'F0': lambda: L.X0(), 'O3': lambda: np.load(os.path.join(C, 'X_stereo_o3.npy')).astype(np.float32),
             'F1': lambda: np.load(os.path.join(C, 'X_morgan_count_r2_2048.npy')).astype(np.float32)}
    return np.hstack([parts[p]() for p in fs.split('+')])


def inner_split(how):
    """train 위치 인덱스 기준 (a=내부 학습 85 %, b=내부 평가 15 %)."""
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    idx, _ = L.split(how)
    tr = idx['train']
    pos = np.arange(len(tr))
    if how == 'random':
        a, b = train_test_split(pos, test_size=0.15, random_state=42)
    else:
        g = L.murcko_groups(how)[tr]
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(pos, groups=g))
    return tr[np.sort(a)], tr[np.sort(b)]


def make_xgb(p, seed=0, n_estimators=10000, es=True, base_score=None):
    from xgboost import XGBRegressor
    kw = dict(n_estimators=n_estimators, learning_rate=p['learning_rate'], max_depth=p['max_depth'],
              min_child_weight=p['min_child_weight'], subsample=p['subsample'], colsample_bytree=p['colsample_bytree'],
              reg_lambda=p['reg_lambda'], reg_alpha=p['reg_alpha'], n_jobs=L.NJ, random_state=seed, eval_metric='mae')
    if p.get('objective', 'sq') == 'phuber60':
        kw.update(objective='reg:pseudohubererror', huber_slope=60.0, base_score=base_score)  # 1단계와 같게 학습 중앙값
    if es:
        kw['early_stopping_rounds'] = 200
    return XGBRegressor(**kw)


def make_lgb(p, seed=0, n_estimators=10000):
    from lightgbm import LGBMRegressor
    kw = dict(n_estimators=n_estimators, learning_rate=p['learning_rate'], num_leaves=p['num_leaves'], max_depth=p['max_depth'],
              min_child_samples=p['min_data_in_leaf'], subsample=p['bagging_fraction'], subsample_freq=1,
              colsample_bytree=p['feature_fraction'], reg_lambda=p['lambda_l2'], reg_alpha=p['lambda_l1'],
              min_child_weight=1e-3, n_jobs=L.NJ, random_state=seed, verbose=-1, deterministic=True, force_row_wise=True)
    if p.get('objective', 'sq') == 'huber60':
        kw.update(objective='huber', alpha=60.0)
    else:
        kw.update(objective='regression')
    return LGBMRegressor(**kw)


def space(trial, lib):
    if lib == 'xgb':
        return dict(objective=trial.suggest_categorical('objective', ['sq', 'phuber60']),
                    learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                    max_depth=trial.suggest_int('max_depth', 4, 10),
                    min_child_weight=trial.suggest_float('min_child_weight', 1, 20, log=True),
                    subsample=trial.suggest_float('subsample', 0.5, 1.0),
                    colsample_bytree=trial.suggest_float('colsample_bytree', 0.2, 0.8),
                    reg_lambda=trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
                    reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 10, log=True))
    return dict(objective=trial.suggest_categorical('objective', ['sq', 'huber60']),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                num_leaves=trial.suggest_int('num_leaves', 15, 255, log=True),
                max_depth=trial.suggest_categorical('max_depth', [-1, 6, 8, 10, 12]),
                min_data_in_leaf=trial.suggest_int('min_data_in_leaf', 5, 100, log=True),
                bagging_fraction=trial.suggest_float('bagging_fraction', 0.5, 1.0),
                feature_fraction=trial.suggest_float('feature_fraction', 0.2, 0.8),
                lambda_l2=trial.suggest_float('lambda_l2', 1e-3, 10, log=True),
                lambda_l1=trial.suggest_float('lambda_l1', 1e-3, 10, log=True))


def fit_inner(lib, p, Xa, ya, Xb, yb, seed=0):
    """내부 학습 → 내부 평가 조기 종료. -> (내부 MAE, best_iter)"""
    if lib == 'xgb':
        m = make_xgb(p, seed, base_score=float(np.median(ya)))
        m.fit(Xa, ya, eval_set=[(Xb, yb)], verbose=False)
        bi = int(m.best_iteration) + 1
    else:
        import lightgbm as lgb
        m = make_lgb(p, seed)
        m.fit(Xa, ya, eval_set=[(Xb, yb)], eval_metric='l1', callbacks=[lgb.early_stopping(200, first_metric_only=True, verbose=False)])
        bi = int(m.best_iteration_)
    pb = m.predict(Xb)
    return float(np.mean(np.abs(yb - pb))), bi


if __name__ == '__main__':
    args = ap.parse_args()
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    how, lib, fs = args.split, args.lib, args.features
    X = feats(fs)
    a, b = inner_split(how)
    print(f'[{lib} {how} {fs}] X {X.shape} inner train {len(a)} / inner eval {len(b)}', flush=True)

    # 기준선(F0, 600그루, seed 42) 내부 MAE — 중단 규칙 기준 (한 번만 계산)
    bpath = os.path.join(L.WORK, f's3_inner_baseline_{how}.json')
    if not os.path.exists(bpath):
        X0 = L.X0()
        with L.mc.CostMeter('탐색') as c:
            m = L.xgb(seed=42).fit(X0[a], y[a])
        bm = float(np.mean(np.abs(y[b] - m.predict(X0[b]))))
        L.ledger(dict(run_id=f's3_inner_baseline_{how}', model='XGBoost', features='F0', data_variant='내부 85 % 학습 / 15 % 평가',
                      objective='reg:squarederror', split=how, seed=42, params=dict(L.mc.BASE_XGB, n_jobs=8), n_train=len(a),
                      notes=f'inner_MAE={bm:.4f} (기준선, 중단 규칙 기준)', **c.as_row()))
        json.dump(dict(inner_MAE=bm, n_a=len(a), n_b=len(b)), open(bpath, 'w'))
    base_inner = json.load(open(bpath))['inner_MAE']
    print('baseline inner MAE', round(base_inner, 4), flush=True)

    storage = 'sqlite:///' + os.path.join(L.TDIR, 'optuna.db').replace('\\', '/')
    name = f'{lib}_{how}_{fs}{args.tag}'
    study = optuna.create_study(study_name=name, storage=storage, direction='minimize', load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=42))
    for t in study.trials:   # 중단된 실행이 남긴 RUNNING trial 은 FAIL 처리
        if t.state == optuna.trial.TrialState.RUNNING:
            study._storage.set_trial_state_values(t._trial_id, optuna.trial.TrialState.FAIL)

    def spent():
        return sum(t.user_attrs.get('cpu_core_h', 0) for t in study.trials)

    def objective(trial):
        p = space(trial, lib)
        with L.mc.CostMeter('탐색') as c:
            mae, bi = fit_inner(lib, p, X[a], y[a], X[b], y[b], seed=0)
        trial.set_user_attr('best_iter', bi)
        trial.set_user_attr('cpu_core_h', c.cpu_core_h)
        trial.set_user_attr('wall_s', c.wall_s)
        L.ledger(dict(run_id=f's3_{name}_t{trial.number}', model='XGBoost' if lib == 'xgb' else 'LightGBM', features=fs,
                      data_variant='내부 85 % 학습 / 15 % 평가 (조기 종료 200)', objective=p['objective'], split=how, seed=0,
                      params=dict(p, n_jobs=8, best_iter=bi), n_train=len(a),
                      notes=f'inner_MAE={mae:.4f}', **c.as_row()))
        print(f't{trial.number}: inner MAE {mae:.4f} best_iter {bi} wall {c.wall_s:.0f}s cpu {c.cpu_core_h:.3f} core-h | {json.dumps(p)}', flush=True)
        return mae

    while True:
        done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(done) >= args.trials:
            break
        if spent() >= args.budget:
            print(f'예산 상한 {args.budget} core-h 도달 — {len(done)} trial 에서 중단', flush=True)
            break
        study.optimize(objective, n_trials=1)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = min(done, key=lambda t: t.value)
    gain = base_inner - best.value
    print(f'완료 {len(done)} trial, 최선 inner MAE {best.value:.4f} (t{best.number}, best_iter {best.user_attrs["best_iter"]}), '
          f'기준선 inner {base_inner:.4f}, 이득 {gain:.3f} K -> {"계속" if gain >= 0.4 else "중단 규칙(이득 < 0.4 K)"}; '
          f'study 비용 {spent():.3f} core-h', flush=True)
