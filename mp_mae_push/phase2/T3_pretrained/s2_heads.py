"""T3 단계 2: 사전학습 임베딩 + head(XGBoost 기준선 설정 / Ridge) 비교 (H14).

실행: 작업 폴더 루트에서 (mp-text). validation만 평가한다. test 행은 예측하지 않는다.
- XGBoost: mc.BASE_XGB, n_jobs=8, seed 0,1,2 (같은 seed의 RDKit217 기준과 짝)
- Ridge: 표준화(train 통계, ±10 클리핑) 후 RidgeCV(alpha 격자, train 안 5-fold CV, KFold shuffle seed 0)
- PCA: train 행만으로 fit
출력: T3_pretrained/work/<run>.npy (validation 예측 캐시, 이어 하기), ledger.csv 한 줄/실행
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.abspath('mp_mae_push/common'))
import numpy as np  # noqa: E402
import mpcommon as mc  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import RidgeCV  # noqa: E402
from sklearn.model_selection import KFold  # noqa: E402

T3 = 'T3_pretrained'
LED = os.path.join(T3, 'ledger.csv')
WORK = os.path.join(T3, 'work')
os.makedirs(WORK, exist_ok=True)
ALPHAS = np.logspace(-2, 4, 13)


def emb(name, pool='mean'):
    return np.load(os.path.join(T3, 'emb', f'{name}_{pool}.npy')).astype(np.float32)


def build(rep, idx, D217):
    """rep 문자열 -> (X_train, X_val). PCA는 train 행만으로 fit."""
    parts_tr, parts_va = [], []
    for tok in rep.split('+'):
        if tok == 'R217':
            parts_tr.append(D217[idx['train']]); parts_va.append(D217[idx['validate']])
            continue
        name, *opt = tok.split(':')            # 예: molformer:pca64, cb_mtr:cls
        pool = 'cls' if 'cls' in opt else 'mean'
        E = emb(name, pool)
        Etr, Eva = E[idx['train']], E[idx['validate']]
        pc = [o for o in opt if o.startswith('pca')]
        if pc:
            k = int(pc[0][3:])
            p = PCA(n_components=k, random_state=0).fit(Etr)
            Etr, Eva = p.transform(Etr).astype(np.float32), p.transform(Eva).astype(np.float32)
        parts_tr.append(Etr); parts_va.append(Eva)
    return np.hstack(parts_tr), np.hstack(parts_va)


def prep_linear(Xtr, Xva):
    Xtr = np.where(np.isfinite(Xtr), Xtr, np.nan).astype(np.float64)
    Xva = np.where(np.isfinite(Xva), Xva, np.nan).astype(np.float64)
    med = np.nanmedian(Xtr, 0); med = np.where(np.isfinite(med), med, 0)
    Xtr = np.where(np.isnan(Xtr), med, Xtr); Xva = np.where(np.isnan(Xva), med, Xva)
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-12] = 1
    return np.clip((Xtr - mu) / sd, -10, 10), np.clip((Xva - mu) / sd, -10, 10)


def run(rep, head, split, seed, y, D217, idx_cache):
    run_id = f'{head}_{rep}_{split}_s{seed}'.replace(':', '-').replace('+', '_')
    cpath = os.path.join(WORK, run_id + '.npy')
    if os.path.exists(cpath):
        return run_id, np.load(cpath), None
    idx = idx_cache[split]
    with mc.CostMeter('s2') as c:
        Xtr, Xva = build(rep, idx, D217)
        ytr = y[idx['train']]
        if head == 'xgb':
            from xgboost import XGBRegressor
            m = XGBRegressor(**mc.BASE_XGB, n_jobs=8, random_state=seed)
            m.fit(Xtr, ytr)
            pred = m.predict(Xva)
            params = f'BASE_XGB n_jobs=8 random_state={seed}'
        else:
            Atr, Ava = prep_linear(Xtr, Xva)
            m = RidgeCV(alphas=ALPHAS, cv=KFold(5, shuffle=True, random_state=0), scoring='neg_mean_absolute_error')
            m.fit(Atr, ytr)
            pred = m.predict(Ava)
            params = f'RidgeCV alpha={m.alpha_:.3g} (grid 1e-2..1e4, 5-fold KFold seed0, MAE); std+clip10; deterministic'
    pred = np.asarray(pred, float)
    np.save(cpath, pred)
    met = mc.metrics(y[idx['validate']], pred)
    mc.append_ledger(LED, dict(run_id=run_id, track='T3', model=head, features=rep, data_variant='original',
                               objective='squared', split=split, seed=seed, params=params + f'; n_feat={Xtr.shape[1]}',
                               val_MAE=round(met['MAE'], 4), val_RMSE=round(met['RMSE'], 4), n_train=len(ytr),
                               notes='validation only; embedding cost in embed_* rows', **c.as_row()))
    return run_id, pred, c
