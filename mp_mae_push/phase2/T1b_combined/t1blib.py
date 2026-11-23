"""t1blib.py : T1b 결합 모델 트랙 공통 도구.
튜닝 설정(lgb_r1, xgb_r1; T1_boosting best_configs/best_settings)과 특징 묶음(F0 RDKit217, F3 물리·결정 39열, F5 LLM 규칙 101열).
작업 공간 루트에서 mp_mae_push/ (mpcommon, splits, cache/X_desc217.npy, phase2/prereg) 를 읽는다.
"""
import os, sys, json, time
sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
TDIR = os.path.abspath(os.path.join(_HERE, '..'))                 # T1b_combined
WS = os.path.abspath(os.path.join(TDIR, '..'))                     # 작업 공간 루트
sys.path.insert(0, os.path.join(WS, 'mp_mae_push', 'common'))
os.environ.setdefault('MP_PUSH_ROOT', os.path.join(WS, 'mp_mae_push'))
import mpcommon as mc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRACK = 'T1b'
LEDGER = os.path.join(TDIR, 'ledger.csv')
WORK = os.path.join(TDIR, 'work')
PRED_DIR = os.path.join(TDIR, 'preds')
CACHE = os.path.join(TDIR, 'cache')
XGB_NJ = 8      # PREREG 1.3
LGB_NT = 6      # 동시 실행 트랙(GIN) 때문에 6 스레드 (T1 트랙은 8)

# 튜닝 설정 (T1_boosting best_configs.csv / best_settings.csv 그대로)
CFG = {
    'xgb_r1': dict(lib='xgb', p={"objective": "sq", "learning_rate": 0.010480532554569605, "max_depth": 7,
                                 "min_child_weight": 1.9625542305123942, "subsample": 0.7913319377267739,
                                 "colsample_bytree": 0.4203094086275295, "reg_lambda": 0.04616632262849512,
                                 "reg_alpha": 0.09824107977907075},
                   n_est={'random': 7057, 'scaffold': 3489}, best_iter={'random': 5998, 'scaffold': 3187}),
    'lgb_r1': dict(lib='lgb', p={"objective": "huber60", "learning_rate": 0.013064627044094662, "num_leaves": 68,
                                 "max_depth": 12, "min_data_in_leaf": 12, "bagging_fraction": 0.8258845819979272,
                                 "feature_fraction": 0.6004369194177872, "lambda_l2": 0.02143620308172678,
                                 "lambda_l1": 0.030134019105179627},
                   n_est={'random': 3688, 'scaffold': 2978}, best_iter={'random': 3135, 'scaffold': 2720}),
}

_c = {}


def y_all():
    if 'y' not in _c:
        _c['y'] = pd.read_csv(os.path.join(mc.WEEK2, 'data', 'mp.csv'))['value'].to_numpy(float)
    return _c['y']


def feats(fs):
    """'F0+F3+F5' -> float32 행렬 (19,945행)."""
    key = ('X', fs)
    if key not in _c:
        parts = {'F0': lambda: np.load(os.path.join(WS, 'mp_mae_push', 'cache', 'X_desc217.npy')),
                 'F3': lambda: np.load(os.path.join(CACHE, 'X_phys.npy')),
                 'F5': lambda: np.load(os.path.join(CACHE, 'F5_rules.npy'))}
        _c[key] = np.hstack([parts[p]().astype(np.float32) for p in fs.split('+')])
    return _c[key]


def phys_col(name):
    cols = pd.read_csv(os.path.join(CACHE, 'X_phys_cols.csv'))['col'].tolist()
    return np.load(os.path.join(CACHE, 'X_phys.npy'))[:, cols.index(name)].astype(float)


def split(how):
    if ('split', how) not in _c:
        _c[('split', how)] = mc.load_split(how)
    return _c[('split', how)]


def dedup_mask(how, which='validate'):
    idx, _ = split(how)
    d = pd.read_csv(os.path.join(WS, 'mp_mae_push', 'phase2', 'prereg', f'dedup_rows_{how}.csv'))
    d = d[d['set'] == which].set_index('row')
    return d.loc[idx[which], 'overlap'].to_numpy() == 0


def murcko_groups():
    if 'murcko' not in _c:
        path = os.path.join(CACHE, 'murcko.csv')
        if os.path.exists(path):
            g = pd.read_csv(path, keep_default_na=False)
        else:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            _, s = split('random')
            s = s.sort_values('row')
            out = []
            for smi in s['smiles']:
                m = Chem.MolFromSmiles(smi)
                try:
                    sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m is not None else ''
                except Exception:
                    sc = ''
                out.append(sc if sc else 'ACYCLIC_' + str(smi))
            g = pd.DataFrame({'row': s['row'].to_numpy(), 'scaffold': out})
            g.to_csv(path, index=False)
        codes = pd.factorize(g['scaffold'])[0]
        arr = np.empty(g['row'].max() + 1, dtype=np.int64)
        arr[g['row'].to_numpy()] = codes
        _c['murcko'] = arr
    return _c['murcko']


def inner_split(how, tr=None):
    """T1 s3_tune.inner_split 과 같음 (85/15, seed 42; scaffold 는 Murcko 그룹). tr 를 주면 그 행 집합 안에서."""
    from sklearn.model_selection import train_test_split, GroupShuffleSplit
    if tr is None:
        tr = split(how)[0]['train']
    pos = np.arange(len(tr))
    if how == 'random':
        a, b = train_test_split(pos, test_size=0.15, random_state=42)
    else:
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42).split(pos, groups=murcko_groups()[tr]))
    return tr[np.sort(a)], tr[np.sort(b)]


def train_folds(how, tr=None, k=5):
    """T1 t1lib.train_folds 와 같음: random KFold(shuffle, 42), scaffold GroupKFold(Murcko). 위치 인덱스."""
    from sklearn.model_selection import KFold, GroupKFold
    if tr is None:
        tr = split(how)[0]['train']
    if how == 'random':
        return list(KFold(k, shuffle=True, random_state=42).split(tr))
    return list(GroupKFold(k).split(tr, groups=murcko_groups()[tr]))


def make_xgb(p, seed=0, n_estimators=10000, es=False, base_score=None):
    from xgboost import XGBRegressor
    kw = dict(n_estimators=n_estimators, learning_rate=p['learning_rate'], max_depth=p['max_depth'],
              min_child_weight=p['min_child_weight'], subsample=p['subsample'], colsample_bytree=p['colsample_bytree'],
              reg_lambda=p['reg_lambda'], reg_alpha=p['reg_alpha'], n_jobs=XGB_NJ, random_state=seed, eval_metric='mae')
    if p.get('objective', 'sq') == 'phuber60':
        kw.update(objective='reg:pseudohubererror', huber_slope=60.0, base_score=base_score)
    if es:
        kw['early_stopping_rounds'] = 200
    return XGBRegressor(**kw)


def make_lgb(p, seed=0, n_estimators=10000, objective=None):
    from lightgbm import LGBMRegressor
    kw = dict(n_estimators=n_estimators, learning_rate=p['learning_rate'], num_leaves=p['num_leaves'], max_depth=p['max_depth'],
              min_child_samples=p['min_data_in_leaf'], subsample=p['bagging_fraction'], subsample_freq=1,
              colsample_bytree=p['feature_fraction'], reg_lambda=p['lambda_l2'], reg_alpha=p['lambda_l1'],
              min_child_weight=1e-3, n_jobs=LGB_NT, random_state=seed, verbose=-1, deterministic=True, force_row_wise=True)
    if objective is not None:
        kw.update(objective=objective)
    elif p.get('objective', 'sq') == 'huber60':
        kw.update(objective='huber', alpha=60.0)
    else:
        kw.update(objective='regression')
    return LGBMRegressor(**kw)


def make_model(cfg, seed, how, n_estimators=None, base_score=None):
    c = CFG[cfg]
    n = n_estimators or c['n_est'][how]
    if c['lib'] == 'xgb':
        return make_xgb(c['p'], seed=seed, n_estimators=n, base_score=base_score)
    return make_lgb(c['p'], seed=seed, n_estimators=n)


def ledger(row):
    row = dict(row)
    row.setdefault('track', TRACK)
    if isinstance(row.get('params'), dict):
        row['params'] = json.dumps(row['params'], ensure_ascii=False, sort_keys=True)
    mc.append_ledger(LEDGER, row)


def track_cost():
    if not os.path.exists(LEDGER):
        return 0.0
    return float(pd.read_csv(LEDGER, encoding='utf-8-sig')['cpu_core_h'].sum())


def fit_val(run_id, cfg, fs, how, seed, *, weight_fn=None, oversample=None, stage='탐색', notes='', data_variant='train',
            log=print):
    """train 전체 적합 → validation 예측 (work/<run_id>.npy, 이어 하기). ledger 1줄."""
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, f'{run_id}.npy')
    if os.path.exists(path):
        return np.load(path)
    X, y = feats(fs), y_all()
    idx, _ = split(how)
    tr, va = idx['train'], idx['validate']
    rows = tr
    if oversample is not None:
        rows = np.concatenate([tr, oversample(tr)])
    w = weight_fn(y[rows]) if weight_fn is not None else None
    c = CFG[cfg]
    with mc.CostMeter(stage) as cm:
        m = make_model(cfg, seed, how, base_score=float(np.median(y[tr])))
        m.fit(X[rows], y[rows], sample_weight=w)
    t0 = time.perf_counter()
    p = np.asarray(m.predict(X[va]), float)
    inf = (time.perf_counter() - t0) / len(va) * 1000
    met = mc.metrics(y[va], p)
    ledger(dict(run_id=run_id, model='XGBoost' if c['lib'] == 'xgb' else 'LightGBM', features=fs, data_variant=data_variant,
                objective=c['p']['objective'], split=how, seed=seed,
                params=dict(c['p'], n_estimators=c['n_est'][how], threads=XGB_NJ if c['lib'] == 'xgb' else LGB_NT, cfg=cfg),
                val_MAE=round(met['MAE'], 4), val_RMSE=round(met['RMSE'], 4), n_train=len(rows),
                infer_s_per_1k=round(inf, 5), notes=notes, **cm.as_row()))
    np.save(path, p)
    log(f'{run_id}: val MAE {met["MAE"]:.4f}  wall {cm.wall_s:.0f}s cpu {cm.cpu_core_h:.4f} core-h', flush=True)
    return p


def fit_oof(run_id, cfg, fs, how, seed=0, *, stage='최종', notes='', log=print):
    """train 안 5-fold OOF (T1 fold 규칙). 나무 수 = 분할별 재적합 나무 수 (fold 학습 크기 ≈ 0.8 n_train, T1 s3_oof 와 같은 규칙)."""
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, f'{run_id}_oof.npy')
    if os.path.exists(path):
        return np.load(path)
    X, y = feats(fs), y_all()
    tr = split(how)[0]['train']
    oof = np.zeros(len(tr))
    c = CFG[cfg]
    n_inner = len(inner_split(how)[0])
    with mc.CostMeter(stage) as cm:
        for a, b in train_folds(how):
            n_est = int(round(c['best_iter'][how] * len(a) / n_inner))   # T1 s3_oof 규칙
            m = make_model(cfg, seed, how, n_estimators=n_est, base_score=float(np.median(y[tr[a]])))
            m.fit(X[tr[a]], y[tr[a]])
            oof[b] = m.predict(X[tr[b]])
    cvm = mc.metrics(y[tr], oof)['MAE']
    ledger(dict(run_id=run_id, model='XGBoost' if c['lib'] == 'xgb' else 'LightGBM', features=fs, data_variant='train 5-fold OOF',
                objective=c['p']['objective'], split=how, seed=seed,
                params=dict(c['p'], n_estimators=c['n_est'][how], threads=XGB_NJ if c['lib'] == 'xgb' else LGB_NT, cfg=cfg),
                n_train=len(tr), notes=f'trainCV_MAE={cvm:.4f} ' + notes, **cm.as_row()))
    np.save(path, oof)
    log(f'{run_id}: OOF MAE {cvm:.4f}  cpu {cm.cpu_core_h:.4f} core-h', flush=True)
    return oof


def evaluate(how, p, ref):
    idx, _ = split(how)
    y = y_all()[idx['validate']]
    m = dedup_mask(how)
    p, ref = np.asarray(p, float), np.asarray(ref, float)
    d, lo, hi = mc.paired_bootstrap(y, p, ref)
    d2, lo2, hi2 = mc.paired_bootstrap(y[m], p[m], ref[m])
    return dict(val_MAE=mc.metrics(y, p)['MAE'], dedup_MAE=mc.metrics(y[m], p[m])['MAE'],
                dMAE=d, ci_lo=lo, ci_hi=hi, dMAE_dedup=d2, ci_lo_dedup=lo2, ci_hi_dedup=hi2)


def load_ref_val(name, how):
    """ref/<name>_<how>_val.csv -> validate 순서 예측."""
    idx, _ = split(how)
    r = pd.read_csv(os.path.join(TDIR, 'ref', f'{name}_{how}_val.csv')).set_index('row').loc[idx['validate']]
    return r['pred'].to_numpy(float)


def load_ref_oof(name, how):
    idx, _ = split(how)
    r = pd.read_csv(os.path.join(TDIR, 'ref', f'oof_{name}_{how}_train.csv')).set_index('row').loc[idx['train']]
    return r['pred'].to_numpy(float)


def save_pred(name, how, which, pred):
    idx, _ = split(how)
    rows = idx['validate'] if which == 'val' else idx['train']
    os.makedirs(PRED_DIR, exist_ok=True)
    path = os.path.join(PRED_DIR, f'{name}_{how}_{which}.csv')
    pd.DataFrame({'row': rows, 'y': y_all()[rows], 'pred': np.asarray(pred, float)}).to_csv(path, index=False)
    return path
