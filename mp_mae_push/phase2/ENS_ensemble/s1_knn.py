"""s1_knn.py : O2 Tanimoto 최근접 이웃 회귀 + 유사도 특징.
- Morgan 반경 2, 2,048비트 (RDKit rdFingerprintGenerator), train·validation 행만 계산 (test 행은 지문도 만들지 않음).
- train 참조만 쓴다. k ∈ {1,3,5,10,20}, 가중 ∈ {uniform, sim, sim^2, sim^4} 를 train 안 5-fold CV(T1 fold 규칙:
  random KFold shuffle 42 / scaffold GroupKFold Murcko)로 고른다. fold 밖 train 행만 이웃으로 쓴 OOF 예측을 함께 저장.
- 유사도 특징: validation 행은 train 전체 대비, train 행은 (a) 자기 자신 제외 (b) 같은 fold 제외 두 가지.
  nn1 = 최근접 Tanimoto, n07 = Tanimoto ≥ 0.7 이웃 수, nn5mean = 상위 5개 평균.
사용: python ENS/s1_knn.py
"""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enslib as E
import numpy as np, pandas as pd

KS = (1, 3, 5, 10, 20)
WS_ = {'uniform': 0, 'sim': 1, 'sim2': 2, 'sim4': 4}
KMAX = 20


def fps_for(rows, smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    X = np.zeros((len(rows), 2048), dtype=np.float32)
    for i, r in enumerate(rows):
        m = Chem.MolFromSmiles(smiles[r])
        X[i] = gen.GetFingerprintAsNumPy(m).astype(np.float32)
    return X


def topk_sims(Q, R, excl=None, k=KMAX, chunk=1500):
    """Q (nq x 2048), R (nr x 2048) 이진. excl(i) -> 제외할 R 열 마스크 함수 정보: ('self', None) 또는 ('fold', fq, fr).
    -> idx (nq x k), sim (nq x k) 내림차순, nn1, n07"""
    cq, cr = Q.sum(1), R.sum(1)
    I = np.zeros((len(Q), k), np.int64); S = np.zeros((len(Q), k), np.float32)
    n07 = np.zeros(len(Q), np.int64)
    for a in range(0, len(Q), chunk):
        b = min(a + chunk, len(Q))
        inter = Q[a:b] @ R.T
        sim = inter / (cq[a:b, None] + cr[None, :] - inter + 1e-12)
        if excl is not None:
            kind = excl[0]
            if kind == 'self':
                sim[np.arange(b - a), np.arange(a, b)] = -1.0
            elif kind == 'fold':
                fq, fr = excl[1], excl[2]
                sim[fq[a:b, None] == fr[None, :]] = -1.0
        n07[a:b] = (sim >= 0.7).sum(1)
        part = np.argpartition(-sim, k, axis=1)[:, :k]
        ps = np.take_along_axis(sim, part, 1)
        o = np.argsort(-ps, axis=1, kind='stable')
        I[a:b] = np.take_along_axis(part, o, 1); S[a:b] = np.take_along_axis(ps, o, 1)
    return I, S, n07


def knn_pred(I, S, ytr, k, p):
    s = np.clip(S[:, :k].astype(float), 0, None)
    w = np.ones_like(s) if p == 0 else (s ** p + 1e-12)
    return (w * ytr[I[:, :k]]).sum(1) / w.sum(1)


def main():
    y = E.y_all()
    _, sdf = E.split('random')
    smiles = sdf.sort_values('row')['smiles'].to_numpy()
    os.makedirs(os.path.join(E.ENS, 'preds'), exist_ok=True); os.makedirs(os.path.join(E.ENS, 'tables'), exist_ok=True)
    os.makedirs(os.path.join(E.ENS, 'cache'), exist_ok=True)
    grid_rows, chosen = [], []
    for how in E.SPLITS:
        with E.mc.CostMeter('탐색') as cm:
            idx, _ = E.split(how)
            tr, va = idx['train'], idx['validate']
            Xtr, Xva = fps_for(tr, smiles), fps_for(va, smiles)
            fold = E.train_folds(how)
            ytr = y[tr]
            # train OOF 이웃 (fold 밖) / train 자기 제외 / validation
            If, Sf, n07f = topk_sims(Xtr, Xtr, ('fold', fold, fold))
            Is, Ss, n07s = topk_sims(Xtr, Xtr, ('self', None))
            Iv, Sv, n07v = topk_sims(Xva, Xtr)
            res = {}
            for k in KS:
                for wn, p in WS_.items():
                    oof = knn_pred(If, Sf, ytr, k, p)
                    fm = [E.mae(ytr[fold == f], oof[fold == f]) for f in range(5)]
                    res[(k, wn)] = np.mean(fm)
                    grid_rows.append(dict(split=how, k=k, weight=wn, cv_MAE_mean_fold=np.mean(fm), cv_MAE_sd_fold=np.std(fm, ddof=1),
                                          cv_MAE_pooled=E.mae(ytr, oof), n_train=len(tr)))
            kb, wb = min(res, key=res.get)
            oof = knn_pred(If, Sf, ytr, kb, WS_[wb]); pv = knn_pred(Iv, Sv, ytr, kb, WS_[wb])
        pd.DataFrame({'row': tr, 'y': ytr, 'pred': oof, 'fold': fold}).to_csv(os.path.join(E.ENS, 'preds', f'O2_knn_{how}_train_oof.csv'), index=False)
        pd.DataFrame({'row': va, 'y': y[va], 'pred': pv}).to_csv(os.path.join(E.ENS, 'preds', f'O2_knn_{how}_val.csv'), index=False)
        dm = E.dedup_mask(how)
        sim_tr = pd.DataFrame({'row': tr, 'set': 'train', 'fold': fold,
                               'nn1_self': Ss[:, 0], 'n07_self': n07s, 'nn5mean_self': Ss[:, :5].mean(1),
                               'nn1_fold': Sf[:, 0], 'n07_fold': n07f, 'nn5mean_fold': Sf[:, :5].mean(1)})
        sim_va = pd.DataFrame({'row': va, 'set': 'validate', 'fold': -1,
                               'nn1_self': Sv[:, 0], 'n07_self': n07v, 'nn5mean_self': Sv[:, :5].mean(1),
                               'nn1_fold': Sv[:, 0], 'n07_fold': n07v, 'nn5mean_fold': Sv[:, :5].mean(1),
                               'dedup_keep': dm.astype(int)})
        pd.concat([sim_tr, sim_va]).to_csv(os.path.join(E.ENS, 'preds', f'similarity_features_{how}.csv'), index=False)
        np.save(os.path.join(E.ENS, 'cache', f'knn_val_nbrs_{how}.npy'), np.stack([Iv, Sv.astype(np.float64)]))
        cv = res[(kb, wb)]
        chosen.append(dict(split=how, k=kb, weight=wb, cv_MAE=cv, val_MAE=E.mae(y[va], pv),
                           val_MAE_dedup=E.mae(y[va][dm], pv[dm]), oof_MAE=E.mae(ytr, oof), n_val=len(va), n_dedup=int(dm.sum()),
                           val_nn1_median=float(np.median(Sv[:, 0])), val_frac_nn1_ge07=float((Sv[:, 0] >= 0.7).mean()),
                           train_nn1_self_median=float(np.median(Ss[:, 0])), train_nn1_fold_median=float(np.median(Sf[:, 0]))))
        E.ledger(dict(run_id=f'O2_knn_{how}', model='Tanimoto kNN', features='Morgan r2 2048 bit', data_variant='train',
                      objective='MAE (CV 선택)', split=how, seed='-', params=dict(k=kb, weight=wb, grid_k=KS, grid_w=list(WS_),
                      cv='T1 fold 규칙 5-fold', fold_cv_MAE=round(cv, 3)),
                      val_MAE=round(E.mae(y[va], pv), 4), val_RMSE=round(float(np.sqrt(np.mean((y[va] - pv) ** 2))), 4),
                      n_train=len(tr), notes='지문 계산 + train×train 유사도 2회 + OOF + val 예측; dedup val MAE '
                      f'{E.mae(y[va][dm], pv[dm]):.3f}', **cm.as_row()))
        print(how, chosen[-1], f'core-h {cm.cpu_core_h:.3f}', flush=True)
    pd.DataFrame(grid_rows).to_csv(os.path.join(E.ENS, 'tables', 'O2_knn_cv_grid.csv'), index=False)
    pd.DataFrame(chosen).to_csv(os.path.join(E.ENS, 'tables', 'O2_knn_chosen.csv'), index=False)


if __name__ == '__main__':
    main()
