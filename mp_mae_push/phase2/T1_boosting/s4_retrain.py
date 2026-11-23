"""4단계 (O10): train 87.5 % 무작위 부분집합 3개(부분집합 k 는 모델 seed k) 로 학습한 validation MAE 와
train 100 % (seed 0-2) 의 차이로 train+validation 재학습 효과를 추정한다.
기준선(BASE_XGB 600그루 고정, 100 % 예측은 1단계 s1_sq_s0-2 재사용)과 튜닝 후보(나무 수 = best_iter x n/n_inner) 모두.
사용: python s4_retrain.py --cfg cfgA --name xgb_tuned1"""
import argparse, json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import t1lib as L
import s3_tune as T

ap = argparse.ArgumentParser()
ap.add_argument('--cfg', default='')
ap.add_argument('--name', default='')
ap.add_argument('--splits', default='random,scaffold')
ap.add_argument('--frac', type=float, default=0.875)
args = ap.parse_args()

y = L.y_all()
X = L.X0()
for how in args.splits.split(','):
    idx, _ = L.split(how)
    tr, va = idx['train'], idx['validate']
    for k in range(3):
        rng = np.random.default_rng(1000 + k)
        sub = np.sort(rng.choice(tr, int(round(args.frac * len(tr))), replace=False))
        # 기준선
        L.fit_record(f's4_base_f875_k{k}_{how}', lambda k=k: L.xgb(seed=k), X[sub], y[sub], X[va], how,
                     data_variant=f'train {args.frac:.3f} 부분집합 k={k} (rng 1000+k)', seed=k,
                     params=dict(L.mc.BASE_XGB, n_jobs=8), stage='탐색', notes='O10 기준선')
        if args.cfg:
            c = pd.read_csv(os.path.join(L.TDIR, 'best_configs.csv')).set_index('cfg_id').loc[args.cfg]
            p, lib = json.loads(c['params_json']), c['lib']
            a, _b = T.inner_split(how)
            bi = json.load(open(os.path.join(L.WORK, f's3_bi_{args.cfg}_{how}.json')))['best_iter']
            n_est = int(round(bi * len(sub) / len(a)))
            med = float(np.median(y[sub]))
            mk = ((lambda k=k: T.make_xgb(p, seed=k, n_estimators=n_est, es=False, base_score=med)) if lib == 'xgb'
                  else (lambda k=k: T.make_lgb(p, seed=k, n_estimators=n_est)))
            L.fit_record(f's4_{args.name}_f875_k{k}_{how}', mk, X[sub], y[sub], X[va], how,
                         model='XGBoost' if lib == 'xgb' else 'LightGBM', objective=p['objective'],
                         data_variant=f'train {args.frac:.3f} 부분집합 k={k} (rng 1000+k)', seed=k,
                         params=dict(p, n_estimators=n_est, n_jobs=8), stage='탐색', notes=f'O10 {args.cfg}')
print('track cost so far', round(L.track_cost(), 4), 'core-h')
