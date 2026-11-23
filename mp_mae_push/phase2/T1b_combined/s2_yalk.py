"""2단계 H06: Yalkowsky 구조 Tm = ΔHm / ΔSm 를 Tm 척도 손실로 학습 (LightGBM 사용자 정의 목적함수).

  h  = GBDT 출력 (ΔHm, kJ/mol);  S = ΔSm = max(a − R·lnσ + R·b·τ, S_MIN)  [J/(K·mol)];  Tm = 1000·h / S
  손실: Tm 척도 Huber(δ = 60 K) — lgb_r1 과 같은 손실 (기본) 또는 Tm 척도 제곱 (--loss sq)
     r = k·h − y, k = 1000/S;  huber: grad = k·clip(r, ±60), hess = k²;  sq: grad = 2k·r, hess = 2k²
  init_score h0 = median(y_train)·S/1000 (상수 Tm 에서 출발).  대조(control): S ≡ 1000 (k = 1, Tm 직접 회귀, 같은 기계).
  σ: sigma3d_max_log(기본; NaN 9행은 0 = σ 1) 또는 sigma2d_log;  τ: tau_clip0 (음수는 0).
  (a, b) 선택: train 안 내부 분할(85/15, T1 규칙)에서 격자. 격자는 비용 때문에 learning_rate × 3 대리 적합(조기 종료 200)으로
  순위를 내고, 고른 (a, b, σ) 하나만 원래 learning_rate 로 내부 조기 종료해 나무 수를 정한다. validation 은 쓰지 않는다.
사용:
  python s2_yalk.py --stage grid  --fs F0+F3+F5 --splits random,scaffold
  python s2_yalk.py --stage es    --fs ... --splits ...            (grid 결과 best 를 읽음)
  python s2_yalk.py --stage refit --fs ... --seeds 0,1,2
  python s2_yalk.py --stage oof   --fs ... --seeds 0
"""
import argparse, json, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import lightgbm as lgb
import t1blib as L

R = 8.314
S_MIN = 10.0
DELTA = 60.0
P = L.CFG['lgb_r1']['p']


def entropy(a, b, sig_col):
    sig = np.nan_to_num(L.phys_col(sig_col), nan=0.0)
    tau = L.phys_col('tau_clip0')
    return np.maximum(a - R * sig + R * b * tau, S_MIN)


def make_obj(k_tr, y_tr, loss):
    def fobj(preds, data):
        r = k_tr * preds - y_tr
        if loss == 'huber':
            return k_tr * np.clip(r, -DELTA, DELTA), k_tr ** 2
        return 2 * k_tr * r, 2 * k_tr ** 2
    return fobj


def params(seed, lr_mult=1.0):
    return dict(objective='none', learning_rate=P['learning_rate'] * lr_mult, num_leaves=P['num_leaves'], max_depth=P['max_depth'],
                min_data_in_leaf=P['min_data_in_leaf'], bagging_fraction=P['bagging_fraction'], bagging_freq=1,
                feature_fraction=P['feature_fraction'], lambda_l2=P['lambda_l2'], lambda_l1=P['lambda_l1'],
                min_sum_hessian_in_leaf=1e-3, num_threads=L.LGB_NT, seed=seed, verbose=-1, deterministic=True,
                force_row_wise=True)


def fit(X, y, S, tr, seed, n_rounds, loss, ev=None, lr_mult=1.0):
    """tr 행으로 학습. ev(행) 이 있으면 Tm MAE 조기 종료 200. -> (booster, best_iter)"""
    k = 1000.0 / S
    med = float(np.median(y[tr]))
    p = params(seed, lr_mult)
    p['objective'] = make_obj(k[tr], y[tr], loss)
    dtr = lgb.Dataset(X[tr], label=y[tr], init_score=med * S[tr] / 1000.0, free_raw_data=False)
    kw = {}
    if ev is not None:
        dev = lgb.Dataset(X[ev], label=y[ev], init_score=med * S[ev] / 1000.0, reference=dtr)
        kev, yev = k[ev], y[ev]

        def feval(preds, data):
            return 'tm_mae', float(np.mean(np.abs(kev * preds - yev))), False
        kw = dict(valid_sets=[dev], feval=feval, callbacks=[lgb.early_stopping(200, verbose=False)])
    b = lgb.train(p, dtr, num_boost_round=n_rounds, **kw)
    return b, (b.best_iteration if ev is not None else n_rounds), med


def predict(b, X, S, rows, med):
    raw = b.predict(X[rows], raw_score=True) + med * S[rows] / 1000.0
    return 1000.0 * raw / S[rows]


def grid_list():
    out = [dict(tag='ctrl', a=None, b=None, sig='none')]
    for a in (40.0, 50.0, 65.0):
        for c in (2.435, 2.85):
            out.append(dict(tag=f'a{a:g}_c{c:g}_s3d', a=a, b=float(np.log(c)), sig='sigma3d_max_log'))
    return out


def S_of(g, n):
    if g['a'] is None:
        return np.full(n, 1000.0)
    return entropy(g['a'], g['b'], g['sig'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True)
    ap.add_argument('--fs', default='F0+F3+F5')
    ap.add_argument('--splits', default='random,scaffold')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--loss', default='huber')
    ap.add_argument('--budget', type=float, default=99.0)
    a = ap.parse_args()
    X, y = L.feats(a.fs), L.y_all()
    fsn = a.fs.replace('+', '')
    os.makedirs(L.WORK, exist_ok=True)
    for how in a.splits.split(','):
        idx, _ = L.split(how)
        tr, va = idx['train'], idx['validate']
        ia, ib = L.inner_split(how)
        gpath = os.path.join(L.WORK, f'h06_grid_{fsn}_{a.loss}_{how}.csv')
        if a.stage == 'grid':
            done = pd.read_csv(gpath) if os.path.exists(gpath) else pd.DataFrame(columns=['tag'])
            gl = grid_list()
            for g in gl + ['sig2d']:
                if g == 'sig2d':   # 3D σ 최선 (a, b) 에 2D σ 대안
                    d = pd.read_csv(gpath)
                    best = d[d['tag'].str.endswith('_s3d')].sort_values('inner_MAE').iloc[0]
                    g = dict(tag=best['tag'].replace('_s3d', '_s2d'), a=float(best['a']), b=float(best['b']), sig='sigma2d_log')
                if g['tag'] in set(done['tag']):
                    continue
                if L.track_cost() > a.budget:
                    print('예산 상한, 중단', flush=True); sys.exit(0)
                S = S_of(g, len(y))
                with L.mc.CostMeter('탐색') as cm:
                    bst, bi, med = fit(X, y, S, ia, 0, 20000, a.loss, ev=ib, lr_mult=3.0)
                mae = float(np.mean(np.abs(predict(bst, X, S, ib, med) - y[ib])))
                L.ledger(dict(run_id=f'h06_grid_{g["tag"]}_{fsn}_{a.loss}_{how}', model='LightGBM+Yalkowsky', features=a.fs,
                              data_variant='내부 85/15 (lr x3 대리, 조기 종료 200)', objective=f'Tm-scale {a.loss}', split=how, seed=0,
                              params=dict(a=g['a'], b=g['b'], sig=g['sig'], S_min=S_MIN, lr_mult=3, best_iter=bi, threads=L.LGB_NT),
                              n_train=len(ia), notes=f'inner_MAE={mae:.4f} S_floor_frac={np.mean(S <= S_MIN):.4f}', **cm.as_row()))
                row = dict(g, inner_MAE=mae, best_iter=bi, cpu_core_h=cm.cpu_core_h)
                done = pd.concat([done, pd.DataFrame([row])], ignore_index=True)
                done.to_csv(gpath, index=False)
                print(f'{how} {g["tag"]}: inner MAE {mae:.4f} it {bi} cpu {cm.cpu_core_h:.4f}', flush=True)
            continue
        d = pd.read_csv(gpath)
        best = d[d['tag'] != 'ctrl'].sort_values('inner_MAE').iloc[0]
        variants = [dict(tag='ctrl', a=None, b=None, sig='none'),
                    dict(tag='yalk', a=float(best['a']), b=float(best['b']), sig=best['sig'], grid_tag=best['tag'])]
        for g in variants:
            S = S_of(g, len(y))
            espath = os.path.join(L.WORK, f'h06_es_{g["tag"]}_{fsn}_{a.loss}_{how}.json')
            if a.stage == 'es':
                if os.path.exists(espath) or g['tag'] == 'ctrl':
                    continue   # 대조는 lgb_r1 나무 수 규칙 그대로 (T1 값)
                with L.mc.CostMeter('탐색') as cm:
                    bst, bi, med = fit(X, y, S, ia, 0, 20000, a.loss, ev=ib)
                mae = float(np.mean(np.abs(predict(bst, X, S, ib, med) - y[ib])))
                json.dump(dict(g, best_iter=int(bi), inner_MAE=mae, n_inner=len(ia)), open(espath, 'w'))
                L.ledger(dict(run_id=f'h06_es_{g["tag"]}_{fsn}_{a.loss}_{how}', model='LightGBM+Yalkowsky', features=a.fs,
                              data_variant='내부 85/15 조기 종료 200 (나무 수 결정)', objective=f'Tm-scale {a.loss}', split=how, seed=0,
                              params=dict(a=g['a'], b=g['b'], sig=g['sig'], best_iter=bi, threads=L.LGB_NT), n_train=len(ia),
                              notes=f'inner_MAE={mae:.4f}', **cm.as_row()))
                print(f'{how} ES {g["tag"]}: best_iter {bi} inner MAE {mae:.4f} cpu {cm.cpu_core_h:.4f}', flush=True)
                continue
            if g['tag'] == 'ctrl':
                continue   # 대조 = 1단계 lgb_r1 같은 특징 (내장 huber) — 따로 적합하지 않음
            bi = json.load(open(espath))['best_iter']
            n_inner = len(ia)
            if a.stage == 'refit':
                n_est = int(round(bi * len(tr) / n_inner))
                for s in [int(v) for v in a.seeds.split(',')]:
                    rid = f'h06_{g["tag"]}_{fsn}_{a.loss}_{how}_s{s}'
                    path = os.path.join(L.WORK, rid + '.npy')
                    if os.path.exists(path):
                        continue
                    if L.track_cost() > a.budget:
                        print('예산 상한, 중단', flush=True); sys.exit(0)
                    with L.mc.CostMeter('탐색') as cm:
                        bst, _, med = fit(X, y, S, tr, s, n_est, a.loss)
                    t0 = time.perf_counter()
                    p = predict(bst, X, S, va, med)
                    inf = (time.perf_counter() - t0) / len(va) * 1000
                    met = L.mc.metrics(y[va], p)
                    L.ledger(dict(run_id=rid, model='LightGBM+Yalkowsky', features=a.fs, data_variant='train', objective=f'Tm-scale {a.loss}',
                                  split=how, seed=s, params=dict(a=g['a'], b=g['b'], sig=g['sig'], n_estimators=n_est, threads=L.LGB_NT,
                                                                 lgb='lgb_r1 설정'), val_MAE=round(met['MAE'], 4),
                                  val_RMSE=round(met['RMSE'], 4), n_train=len(tr), infer_s_per_1k=round(inf, 5),
                                  notes=f'S_floor_frac_val={np.mean(S[va] <= S_MIN):.4f}', **cm.as_row()))
                    np.save(path, p)
                    print(f'{rid}: val MAE {met["MAE"]:.4f} cpu {cm.cpu_core_h:.4f}', flush=True)
            if a.stage == 'oof':
                for s in [int(v) for v in a.seeds.split(',')]:
                    rid = f'h06_{g["tag"]}_{fsn}_{a.loss}_{how}_s{s}'
                    path = os.path.join(L.WORK, rid + '_oof.npy')
                    if os.path.exists(path):
                        continue
                    oof = np.zeros(len(tr))
                    with L.mc.CostMeter('최종') as cm:
                        for fa, fb in L.train_folds(how):
                            n_est = int(round(bi * len(fa) / n_inner))
                            bst, _, med = fit(X, y, S, tr[fa], s, n_est, a.loss)
                            oof[fb] = predict(bst, X, S, tr[fb], med)
                    mae = float(np.mean(np.abs(oof - y[tr])))
                    L.ledger(dict(run_id=rid + '_oof', model='LightGBM+Yalkowsky', features=a.fs, data_variant='train 5-fold OOF',
                                  objective=f'Tm-scale {a.loss}', split=how, seed=s,
                                  params=dict(a=g['a'], b=g['b'], sig=g['sig'], tree_rule=f'best_iter {bi} x n_fold/{n_inner}',
                                              threads=L.LGB_NT), n_train=len(tr), notes=f'trainCV_MAE={mae:.4f}', **cm.as_row()))
                    np.save(path, oof)
                    print(f'{rid} OOF MAE {mae:.4f} cpu {cm.cpu_core_h:.4f}', flush=True)
    print('track cost', round(L.track_cost(), 4), flush=True)
