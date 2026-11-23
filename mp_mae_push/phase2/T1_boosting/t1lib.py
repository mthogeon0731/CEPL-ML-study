"""t1lib.py : T1 트리 부스팅 트랙 공통 도구 (데이터·평가·예측 저장·ledger)."""
import os, sys, json
sys.dont_write_bytecode = True
_HERE = os.path.dirname(os.path.abspath(__file__))
WS = os.path.abspath(os.path.join(_HERE, '..', '..'))          # 작업 공간 루트
sys.path.insert(0, os.path.join(WS, 'mp_mae_push', 'common'))
os.environ.setdefault('MP_PUSH_ROOT', os.path.join(WS, 'mp_mae_push'))
import mpcommon as mc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TRACK = 'T1_boosting'
TDIR = os.path.join(WS, TRACK)
LEDGER = os.path.join(TDIR, 'ledger.csv')
RESULTS = os.path.join(TDIR, 'results_val.csv')
PRED_DIR = os.path.join(TDIR, 'preds')
OOF_DIR = os.path.join(TDIR, 'oof')
NJ = 8   # XGBoost n_jobs 고정 (PREREG 1.3)

_cache = {}


def y_all():
    if 'y' not in _cache:
        _cache['y'] = pd.read_csv(os.path.join(mc.WEEK2, 'data', 'mp.csv'))['value'].to_numpy(float)
    return _cache['y']


def X0():
    if 'X0' not in _cache:
        _cache['X0'] = np.load(os.path.join(WS, 'mp_mae_push', 'cache', 'X_desc217.npy')).astype(np.float32)
    return _cache['X0']


def split(how):
    if ('split', how) not in _cache:
        idx, s = mc.load_split(how)
        _cache[('split', how)] = (idx, s)
    return _cache[('split', how)]


def ref_val(how):
    """기준선 n_jobs=8 validation 예측 (validate 행 순서로 정렬)."""
    idx, _ = split(how)
    r = pd.read_csv(os.path.join(WS, 'mp_mae_push', 'phase1', 'preds', f'baseline_xgb_desc217_{how}_val.csv'))
    r = r.set_index('row').loc[idx['validate']]
    return r['pred'].to_numpy(float)


def dedup_mask(how):
    """validate 행 순서의 bool 배열: True = 중복 제거 validation 에 포함."""
    idx, _ = split(how)
    d = pd.read_csv(os.path.join(WS, 'mp_mae_push', 'phase2', 'prereg', f'dedup_rows_{how}.csv'))
    d = d[d['set'] == 'validate'].set_index('row')
    return (d.loc[idx['validate'], 'overlap'].to_numpy() == 0)


def murcko_groups(how):
    """분할 파일 smiles 의 Murcko 뼈대 문자열 (행 번호 -> 그룹 id 배열, 전체 19,945)."""
    key = ('murcko',)
    if key not in _cache:
        path = os.path.join(TDIR, 'cache_murcko.csv')
        if os.path.exists(path):
            g = pd.read_csv(path)
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
                out.append(sc if sc else 'ACYCLIC_' + str(smi))  # 고리 없는 분자는 자기 자신이 그룹
            g = pd.DataFrame({'row': s['row'].to_numpy(), 'scaffold': out})
            g.to_csv(path, index=False)
        codes = pd.factorize(g['scaffold'])[0]
        arr = np.empty(g['row'].max() + 1, dtype=np.int64)
        arr[g['row'].to_numpy()] = codes
        _cache[key] = arr
    return _cache[key]


def evaluate(how, pred_val, ref=None):
    """validation 전체·중복 제거 기준 지표와 기준선(또는 ref) 대비 짝지은 부트스트랩."""
    idx, _ = split(how)
    y = y_all()[idx['validate']]
    p = np.asarray(pred_val, float)
    r = ref_val(how) if ref is None else np.asarray(ref, float)
    m = dedup_mask(how)
    full = mc.metrics(y, p)
    dd = mc.metrics(y[m], p[m])
    d, lo, hi = mc.paired_bootstrap(y, p, r)
    d2, lo2, hi2 = mc.paired_bootstrap(y[m], p[m], r[m])
    return dict(val_MAE=full['MAE'], val_RMSE=full['RMSE'], val_R2=full['R2'], n_val=len(y),
                dedup_MAE=dd['MAE'], n_dedup=int(m.sum()),
                dMAE=d, ci_lo=lo, ci_hi=hi, dMAE_dedup=d2, ci_lo_dedup=lo2, ci_hi_dedup=hi2)


def save_val_pred(name, how, pred_val):
    idx, _ = split(how)
    os.makedirs(PRED_DIR, exist_ok=True)
    path = os.path.join(PRED_DIR, f'{name}_{how}_val.csv')
    pd.DataFrame({'row': idx['validate'], 'y': y_all()[idx['validate']], 'pred': np.asarray(pred_val, float)}).to_csv(path, index=False)
    return path


def ledger(row):
    row = dict(row)
    row.setdefault('track', 'T1')
    if isinstance(row.get('params'), dict):
        row['params'] = json.dumps(row['params'], ensure_ascii=False, sort_keys=True)
    mc.append_ledger(LEDGER, row)


def result(row):
    """후보 요약 표 (results_val.csv) 한 줄 추가."""
    new = not os.path.exists(RESULTS)
    pd.DataFrame([row]).to_csv(RESULTS, mode='a', header=new, index=False, encoding='utf-8-sig')


WORK = os.path.join(TDIR, 'work')


def fit_record(run_id, make_model, Xtr, ytr, Xva, how, *, model='XGBoost', features='F0', objective='reg:squarederror',
               seed='', params=None, stage='탐색', notes='', data_variant='train', fit_kw=None, log=print):
    """적합 1회 = ledger 1줄. validation 예측은 work/<run_id>.npy 로 저장 (이미 있으면 건너뜀 = 이어 하기)."""
    import time
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, f'{run_id}.npy')
    if os.path.exists(path):
        return np.load(path)
    with mc.CostMeter(stage) as c:
        m = make_model()
        m.fit(Xtr, ytr, **(fit_kw or {}))
    t0 = time.perf_counter()
    p = np.asarray(m.predict(Xva), float)
    inf = (time.perf_counter() - t0) / len(Xva) * 1000
    idx, _ = split(how)
    yv = y_all()[idx['validate']]
    met = mc.metrics(yv, p)
    extra = ''
    try:
        bi = getattr(m, 'best_iteration', None)
        if bi is not None:
            extra = f' best_iter={bi}'
    except Exception:
        pass
    ledger(dict(run_id=run_id, model=model, features=features, data_variant=data_variant, objective=objective,
                split=how, seed=seed, params=params or {}, val_MAE=round(met['MAE'], 4), val_RMSE=round(met['RMSE'], 4),
                n_train=len(ytr), infer_s_per_1k=round(inf, 5), notes=(notes + extra).strip(), **c.as_row()))
    np.save(path, p)
    log(f'{run_id}: val MAE {met["MAE"]:.4f} K  wall {c.wall_s:.1f}s  cpu {c.cpu_core_h:.4f} core-h{extra}')
    return p


def train_folds(how, k=5):
    """train 안 5-fold: random = KFold(shuffle, 42), scaffold = GroupKFold(Murcko 뼈대). train 내 위치 인덱스 반환."""
    from sklearn.model_selection import KFold, GroupKFold
    idx, _ = split(how)
    tr = idx['train']
    if how == 'random':
        return list(KFold(k, shuffle=True, random_state=42).split(tr))
    g = murcko_groups(how)[tr]
    return list(GroupKFold(k).split(tr, groups=g))


def cv_record(run_id, make_model, X, how, *, model='XGBoost', features='F0', objective='reg:squarederror', seed='',
              params=None, stage='탐색', notes='', make_fit_kw=None, log=print):
    """train 안 5-fold CV 1회 = ledger 1줄 (val_MAE 칸은 비우고 notes 에 CV MAE). OOF 예측을 work/<run_id>_oof.npy."""
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, f'{run_id}_oof.npy')
    idx, _ = split(how)
    tr = idx['train']
    y = y_all()
    if os.path.exists(path):
        return np.load(path)
    oof = np.zeros(len(tr))
    with mc.CostMeter(stage) as c:
        for a, b in train_folds(how):
            m = make_model(y[tr[a]])
            m.fit(X[tr[a]], y[tr[a]], **((make_fit_kw(a, b) if make_fit_kw else None) or {}))
            oof[b] = m.predict(X[tr[b]])
    cvm = mc.metrics(y[tr], oof)['MAE']
    ledger(dict(run_id=run_id, model=model, features=features, data_variant='train 5-fold CV', objective=objective,
                split=how, seed=seed, params=params or {}, n_train=len(tr),
                notes=f'trainCV_MAE={cvm:.4f} ' + notes, **c.as_row()))
    np.save(path, oof)
    log(f'{run_id}: train-CV MAE {cvm:.4f} K  cpu {c.cpu_core_h:.4f} core-h')
    return oof


def ledger_cost(prefixes):
    """ledger 에서 run_id 접두사 목록에 해당하는 비용 합."""
    lg = pd.read_csv(LEDGER, encoding='utf-8-sig')
    msk = np.zeros(len(lg), bool)
    for pf in prefixes:
        msk |= lg['run_id'].astype(str).str.startswith(pf).to_numpy()
    sub = lg[msk]
    return dict(cpu_core_h=float(sub['cpu_core_h'].sum()), wall_s=float(sub['wall_s'].sum()),
                energy_kwh=float(sub['energy_kwh'].sum()), n_fits=int(len(sub)),
                infer_s_per_1k=float(sub['infer_s_per_1k'].sum()))


def track_cost():
    if not os.path.exists(LEDGER):
        return 0.0
    return float(pd.read_csv(LEDGER, encoding='utf-8-sig')['cpu_core_h'].sum())


def xgb(params=None, seed=42, **kw):
    from xgboost import XGBRegressor
    p = dict(mc.BASE_XGB)
    p.update(params or {})
    p.update(kw)
    return XGBRegressor(n_jobs=NJ, random_state=seed, **p)
