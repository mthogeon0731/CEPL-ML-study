"""3단계 OOF: 최종 후보 설정의 train 5-fold out-of-fold 예측 (random: KFold shuffle 42, scaffold: GroupKFold Murcko).
fold 마다 나무 수 = best_iter(해당 분할) x (fold 학습 n / 내부 학습 n). seed 여러 개면 평균.
출력: T1_boosting/oof/oof_<name>_<split>_train.csv (row, y, pred)
사용: python s3_oof.py --cfg cfgA --name xgb_tuned1 --splits random,scaffold --seeds 0"""
import argparse, json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import t1lib as L
import s3_tune as T

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', required=True)
ap.add_argument('--name', required=True)
ap.add_argument('--splits', default='random,scaffold')
ap.add_argument('--seeds', default='0')
ap.add_argument('--features', default='F0')
args = ap.parse_args()

y = L.y_all()
X = T.feats(args.features)
c = pd.read_csv(os.path.join(L.TDIR, 'best_configs.csv')).set_index('cfg_id').loc[args.cfg]
p, lib = json.loads(c['params_json']), c['lib']
os.makedirs(L.OOF_DIR, exist_ok=True)
for how in args.splits.split(','):
    idx, _ = L.split(how)
    tr = idx['train']
    a, _b = T.inner_split(how)
    bi = json.load(open(os.path.join(L.WORK, f's3_bi_{args.cfg}_{how}.json')))['best_iter']
    preds = []
    for s in [int(v) for v in args.seeds.split(',')]:
        path = os.path.join(L.WORK, f's3_oof_{args.name}_s{s}_{how}.npy')
        if os.path.exists(path):
            preds.append(np.load(path)); continue
        oof = np.zeros(len(tr))
        with L.mc.CostMeter('최종') as cm:
            for fa, fb in L.train_folds(how):
                n_est = int(round(bi * len(fa) / len(a)))
                med = float(np.median(y[tr[fa]]))
                m = (T.make_xgb(p, seed=s, n_estimators=n_est, es=False, base_score=med) if lib == 'xgb'
                     else T.make_lgb(p, seed=s, n_estimators=n_est))
                m.fit(X[tr[fa]], y[tr[fa]])
                oof[fb] = m.predict(X[tr[fb]])
        mae = float(np.mean(np.abs(y[tr] - oof)))
        L.ledger(dict(run_id=f's3_oof_{args.name}_s{s}_{how}', model='XGBoost' if lib == 'xgb' else 'LightGBM',
                      features=args.features, data_variant='train 5-fold OOF', objective=p['objective'], split=how, seed=s,
                      params=dict(p, n_jobs=8, tree_rule=f'best_iter {bi} x n_fold/{len(a)}'), n_train=len(tr),
                      notes=f'OOF_MAE={mae:.4f}; folds={"KFold shuffle 42" if how == "random" else "GroupKFold Murcko"}',
                      **cm.as_row()))
        np.save(path, oof); preds.append(oof)
        print(f'{args.name} {how} seed {s}: OOF MAE {mae:.4f}  cpu {cm.cpu_core_h:.3f} core-h', flush=True)
    pm = np.mean(preds, 0)
    pd.DataFrame({'row': tr, 'y': y[tr], 'pred': pm}).to_csv(os.path.join(L.OOF_DIR, f'oof_{args.name}_{how}_train.csv'), index=False)
    print(f'{args.name} {how}: OOF (seed 평균 {len(preds)}) MAE {np.mean(np.abs(y[tr] - pm)):.4f}', flush=True)
