"""s2_ensemble.py : H10 앙상블. 멤버 목록은 ENS/members.csv (member, track, split, kind[val|OOF], version_id, path).
새 멤버(T3 등)는 members.csv 에 줄을 더하고 다시 돌리면 된다.

방법 (모두 비음수):
  wavg_mae  : 단체(simplex, w>=0, sum 1) 가중 평균, 목적 MAE (선형계획, HiGHS)
  wavg_mse  : 단체 가중 평균, 목적 MSE (NNLS + 합 1 제약 벌점)
  mean_top{k}: 학습 fold 안 단일 MAE 상위 k개 단순 평균 (k = 2,3,4,5), mean_all: 전체 평균
  mean_topauto: 학습 fold 안에서 k 를 고름 (학습 fold 내 MAE 최소)
  ridge_pos : Ridge(positive=True, 절편 있음), alpha 는 학습 fold 안 5-fold CV
  fwd{m}_mae: m = 2,3,4 멤버 축소 앙상블. 학습 fold 안 전진 선택(기준 = MSE 가중치의 학습 fold MAE), 최종 가중치는 MAE 단체
정직한 validation 예측: validation 안 5-fold 교차 적합 × 10회 반복(fold seed 0–9). 반복별 예측을 평균한 것을 '정직한 예측'으로 저장.
최종 가중치: validation 전체로 한 번 적합해 기록.
확인용: OOF 가 있는 멤버만으로 train OOF 에서 가중치를 적합해 validation 에 적용.
"""
import os, sys, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enslib as E
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.optimize import linprog, nnls
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

BEST = 'lgb_r1_F0F3F5'
BASE = 'baseline_xgb600_F0'
N_REP, N_FOLD = 10, 5
ALPHAS = (1e-2, 1e-1, 1, 10, 100, 1000, 1e4)
# scaffold train OOF 의 fold 규칙이 멤버마다 다르다 (T8 M2 는 무작위 KFold -> OOF 가 낙관적). scaffold OOF 가중치 적합에서 뺀다.
OOF_EXCLUDE = {'scaffold': {'xgb600_F0F5_M2'}}


# ---------------- 적합기 ----------------
def _w_mae2(P, y):
    """m = 2 정확해: min_w Σ|r_i − w d_i|, r = y − p2, d = p1 − p2 -> 가중 중앙값(r/d, 가중 |d|) 을 [0,1] 로 자름."""
    r, d = y - P[:, 1], P[:, 0] - P[:, 1]
    k = np.abs(d) > 1e-12
    if not k.any():
        return np.array([0.5, 0.5])
    q, wt = r[k] / d[k], np.abs(d[k])
    o = np.argsort(q); c = np.cumsum(wt[o])
    w = float(np.clip(q[o][np.searchsorted(c, 0.5 * c[-1])], 0, 1))
    return np.array([w, 1 - w])


def w_mae(P, y):
    n, m = P.shape
    if m == 1:
        return np.ones(1)
    if m == 2:
        return _w_mae2(P, y)
    c = np.r_[np.zeros(m), np.ones(2 * n)]
    A = sp.vstack([sp.hstack([sp.csr_matrix(P), sp.eye(n), -sp.eye(n)]),
                   sp.csr_matrix(np.r_[np.ones(m), np.zeros(2 * n)][None, :])]).tocsr()
    r = linprog(c, A_eq=A, b_eq=np.r_[y, 1.0], bounds=(0, None), method='highs')
    w = np.clip(r.x[:m], 0, None)
    return w / w.sum()


def w_mse(P, y, lam=1e4):
    A = np.vstack([P, lam * np.ones((1, P.shape[1]))]); b = np.r_[y, lam]
    w, _ = nnls(A, b, maxiter=5000)
    return w / w.sum()


class Lin:
    def __init__(self, w, b=0.0):
        self.w, self.b = np.asarray(w, float), float(b)

    def __call__(self, P):
        return P @ self.w + self.b


def fit_method(name, P, y, names):
    m = P.shape[1]
    if name == 'wavg_mae':
        return Lin(w_mae(P, y))
    if name == 'wavg_mse':
        return Lin(w_mse(P, y))
    if name == 'mean_all':
        return Lin(np.ones(m) / m)
    if name.startswith('mean_top'):
        order = np.argsort(np.abs(P - y[:, None]).mean(0))
        if name == 'mean_topauto':
            best = min(range(1, m + 1), key=lambda k: np.abs(P[:, order[:k]].mean(1) - y).mean())
            k = best
        else:
            k = min(int(name[len('mean_top'):]), m)
        w = np.zeros(m); w[order[:k]] = 1.0 / k
        return Lin(w)
    if name == 'ridge_pos':
        kf = KFold(5, shuffle=True, random_state=123)
        sc = {}
        for a in ALPHAS:
            e = []
            for i, j in kf.split(P):
                r = Ridge(alpha=a, positive=True).fit(P[i], y[i]); e.append(np.abs(r.predict(P[j]) - y[j]).mean())
            sc[a] = np.mean(e)
        a = min(sc, key=sc.get)
        r = Ridge(alpha=a, positive=True).fit(P, y)
        f = Lin(r.coef_, r.intercept_); f.alpha = a
        return f
    if name.startswith('fwd'):
        mm = int(name[3]); sel = []
        for _ in range(min(mm, m)):
            best, bs = None, np.inf
            for j in range(m):
                if j in sel:
                    continue
                cols = sel + [j]
                w = w_mse(P[:, cols], y)
                s = np.abs(P[:, cols] @ w - y).mean()
                if s < bs:
                    best, bs = j, s
            sel.append(best)
        w = np.zeros(m); w[sel] = w_mae(P[:, sel], y)
        f = Lin(w); f.sel = [names[s] for s in sel]
        return f
    raise ValueError(name)


METHODS = ['wavg_mae', 'wavg_mse', 'mean_all', 'mean_top2', 'mean_top3', 'mean_top4', 'mean_top5', 'mean_topauto',
           'ridge_pos', 'fwd2_mae', 'fwd3_mae', 'fwd4_mae']


def crossfit(method, P, y, names):
    """-> 반복별 예측 (N_REP x n), 반복별 MAE"""
    H = np.zeros((N_REP, len(y)))
    for r in range(N_REP):
        for i, j in KFold(N_FOLD, shuffle=True, random_state=r).split(P):
            H[r, j] = fit_method(method, P[i], y[i], names)(P[j])
    return H


def members_for(how, kind, reg, exclude=()):
    sub = reg[(reg['split'] == how) & (reg['kind'] == kind) & (~reg['member'].isin(exclude))]
    return {r.member: (r.path if os.path.isabs(r.path) else os.path.join(E.WS, r.path)) for r in sub.itertuples()}


def compare_rows(y, p, ref, dm, label):
    out = {}
    out[f'd_vs_{label}'] = E.fmt_ci(E.pboot(y, p, ref))
    out[f'd_vs_{label}_dedup'] = E.fmt_ci(E.pboot(y[dm], p[dm], ref[dm]))
    t = E.pboot(y, p, ref); td = E.pboot(y[dm], p[dm], ref[dm])
    out[f'd_vs_{label}_num'], out[f'd_vs_{label}_dedup_num'] = t[0], td[0]
    out[f'd_vs_{label}_sig'] = int(t[2] < 0); out[f'd_vs_{label}_dedup_sig'] = int(td[2] < 0)
    return out


def main(tag=''):
    reg = pd.read_csv(os.path.join(E.ENS, E.MEMBERS_FILE))
    os.makedirs(os.path.join(E.ENS, 'preds', 'ens'), exist_ok=True)
    rows, wrows = [], []
    for how in E.SPLITS:
        with E.mc.CostMeter('탐색') as cm:
            mv, y = E.load_members(how, 'val', members_for(how, 'val', reg))
            names = list(mv); P = np.column_stack([mv[n] for n in names])
            dm = E.dedup_mask(how)
            best, base = mv[BEST], mv[BASE]
            # 오차 상관
            R = P - y[:, None]
            pd.DataFrame(np.corrcoef(R.T), index=names, columns=names).round(4).to_csv(
                os.path.join(E.ENS, 'tables', f'error_corr_{how}{tag}.csv'))
            # 단일 멤버
            for n in names:
                p = mv[n]
                rows.append(dict(split=how, method='single', member_set=n, n_members=1, MAE=E.mae(y, p), MAE_dedup=E.mae(y[dm], p[dm]),
                                 rep_mean=np.nan, rep_sd=np.nan, n=len(y), n_dedup=int(dm.sum()), has_O2=int(n == 'O2_knn'),
                                 **compare_rows(y, p, best, dm, 'best'), **compare_rows(y, p, base, dm, 'base')))
            sets = {'all': names, 'noO2': [n for n in names if n != 'O2_knn']}
            for sname, snames in sets.items():
                cols = [names.index(n) for n in snames]
                Ps = P[:, cols]
                for meth in METHODS:
                    t0 = time.time()
                    H = crossfit(meth, Ps, y, snames)
                    h = H.mean(0)
                    repm = [E.mae(y, H[r]) for r in range(N_REP)]
                    fin = fit_method(meth, Ps, y, snames)
                    wd = {snames[i]: round(float(fin.w[i]), 4) for i in range(len(snames)) if abs(fin.w[i]) > 1e-4}
                    key = f'{meth}|{sname}'
                    pd.DataFrame({'row': E.split(how)[0]['validate'], 'y': y, 'pred': h,
                                  **{f'pred_rep{r}': H[r] for r in range(N_REP)}}).to_csv(
                        os.path.join(E.ENS, 'preds', 'ens', f'{how}__{meth}__{sname}{tag}_val_honest.csv'), index=False)
                    rows.append(dict(split=how, method=meth, member_set=sname, n_members=int(np.sum(np.abs(fin.w) > 1e-4)),
                                     MAE=E.mae(y, h), MAE_dedup=E.mae(y[dm], h[dm]), rep_mean=np.mean(repm), rep_sd=np.std(repm, ddof=1),
                                     n=len(y), n_dedup=int(dm.sum()), has_O2=int(wd.get('O2_knn', 0) > 0) if sname == 'all' else 0,
                                     insample_MAE_final=E.mae(y, fin(Ps)),
                                     **compare_rows(y, h, best, dm, 'best'), **compare_rows(y, h, base, dm, 'base')))
                    wrows.append(dict(split=how, method=meth, member_set=sname, intercept=fin.b, alpha=getattr(fin, 'alpha', ''),
                                      selected=';'.join(getattr(fin, 'sel', [])), weights=json.dumps(wd)))
                    print(how, key, f'{rows[-1]["MAE"]:.3f} dedup {rows[-1]["MAE_dedup"]:.3f} rep {np.mean(repm):.3f}±{np.std(repm, ddof=1):.3f}',
                          f'{time.time() - t0:.1f}s', flush=True)
            # 확인용: train OOF 에서 가중치 적합 -> validation 적용
            oof_m = members_for(how, 'OOF', reg, exclude=OOF_EXCLUDE.get(how, set()))
            mo, yo = E.load_members(how, 'OOF', oof_m)
            on = list(mo); Po = np.column_stack([mo[n] for n in on]); Pv = np.column_stack([mv[n] for n in on])
            for sname, sn in {'oofset_all': on, 'oofset_noO2': [n for n in on if n != 'O2_knn']}.items():
                ci = [on.index(n) for n in sn]
                for meth in ['wavg_mae', 'wavg_mse', 'ridge_pos']:
                    f = fit_method(meth, Po[:, ci], yo, sn); h = f(Pv[:, ci])
                    wd = {sn[i]: round(float(f.w[i]), 4) for i in range(len(sn)) if abs(f.w[i]) > 1e-4}
                    rows.append(dict(split=how, method=meth + '_trainOOF', member_set=sname, n_members=len(wd), MAE=E.mae(y, h),
                                     MAE_dedup=E.mae(y[dm], h[dm]), rep_mean=np.nan, rep_sd=np.nan, n=len(y), n_dedup=int(dm.sum()),
                                     has_O2=int(wd.get('O2_knn', 0) > 0), insample_MAE_final=E.mae(yo, f(Po[:, ci])),
                                     **compare_rows(y, h, best, dm, 'best'), **compare_rows(y, h, base, dm, 'base')))
                    wrows.append(dict(split=how, method=meth + '_trainOOF', member_set=sname, intercept=f.b, alpha=getattr(f, 'alpha', ''),
                                      selected='', weights=json.dumps(wd)))
                    pd.DataFrame({'row': E.split(how)[0]['validate'], 'y': y, 'pred': h}).to_csv(
                        os.path.join(E.ENS, 'preds', 'ens', f'{how}__{meth}_trainOOF__{sname}{tag}_val.csv'), index=False)
                    # 같은 멤버 집합의 validation 교차 적합 (비교용)
                    H = crossfit(meth, Pv[:, ci], y, sn); hv = H.mean(0)
                    rows.append(dict(split=how, method=meth, member_set=sname + '_valCV', n_members=len(sn), MAE=E.mae(y, hv),
                                     MAE_dedup=E.mae(y[dm], hv[dm]), rep_mean=np.mean([E.mae(y, H[r]) for r in range(N_REP)]),
                                     rep_sd=np.std([E.mae(y, H[r]) for r in range(N_REP)], ddof=1), n=len(y), n_dedup=int(dm.sum()),
                                     has_O2=int('O2_knn' in sn), **compare_rows(y, hv, best, dm, 'best'), **compare_rows(y, hv, base, dm, 'base')))
                    print(how, meth, sname, f'trainOOF {E.mae(y, h):.3f} / valCV {E.mae(y, hv):.3f}', flush=True)
        E.ledger(dict(run_id=f'H10_ensemble_{how}{tag}', model='앙상블 (가중 평균·스태킹)', features=f'{len(names)} 멤버 val 예측',
                      data_variant='validation 5-fold×10 교차 적합 + train OOF 확인', objective='MAE/MSE', split=how, seed='fold seed 0-9',
                      params=dict(methods=METHODS, members=names), n_train=len(y), notes='예측 결합만(모델 재학습 없음)', **cm.as_row()))
    res = pd.DataFrame(rows); res.to_csv(os.path.join(E.ENS, f'ensemble_results{tag}.csv'), index=False)
    pd.DataFrame(wrows).to_csv(os.path.join(E.ENS, 'tables', f'ensemble_final_weights{tag}.csv'), index=False)
    return res


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '')
