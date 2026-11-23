"""3단계 재적합: 튜닝 상위 설정을 train 전체에 seed 0-2 로 다시 적합 → validation 예측.
나무 수 = 해당 분할 내부 분할의 조기 종료 best_iter x (n_train / n_inner_train)  (random 15,955/13,561 = 1.177 ≈ 1/0.85).
random 에서 찾은 설정을 scaffold 에 적용할 때는 scaffold 내부 분할(Murcko 그룹)에서 같은 설정으로 조기 종료를 1회(seed 0) 돌려 best_iter 를 얻는다.
입력: T1_boosting/best_configs.csv (cfg_id, lib, study, trial, params_json)
사용: python s3_refit.py --cfgs cfgA,cfgB --splits random,scaffold --seeds 0,1,2"""
import argparse, json, os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import t1lib as L
import s3_tune as T

ap = argparse.ArgumentParser()
ap.add_argument('--cfgs', required=True)
ap.add_argument('--splits', default='random,scaffold')
ap.add_argument('--seeds', default='0,1,2')
ap.add_argument('--features', default='F0')
args = ap.parse_args()

y = L.y_all()
X = T.feats(args.features)
cfgs = pd.read_csv(os.path.join(L.TDIR, 'best_configs.csv')).set_index('cfg_id')
for cid in args.cfgs.split(','):
    c = cfgs.loc[cid]
    p = json.loads(c['params_json'])
    lib = c['lib']
    for how in args.splits.split(','):
        idx, _ = L.split(how)
        tr, va = idx['train'], idx['validate']
        a, b = T.inner_split(how)
        # 나무 수 결정
        bi_path = os.path.join(L.WORK, f's3_bi_{cid}_{how}.json')
        if os.path.exists(bi_path):
            bi = json.load(open(bi_path))['best_iter']
        elif how == c['split'] and not pd.isna(c.get('best_iter', np.nan)):
            bi = int(c['best_iter'])
            json.dump(dict(best_iter=bi, source=f'study {c["study"]} t{c["trial"]}'), open(bi_path, 'w'))
        else:
            with L.mc.CostMeter('탐색') as cm:
                mae, bi = T.fit_inner(lib, p, X[a], y[a], X[b], y[b], seed=0)
            L.ledger(dict(run_id=f's3_es_{cid}_{how}', model='XGBoost' if lib == 'xgb' else 'LightGBM', features=args.features,
                          data_variant=f'{how} 내부 분할 조기 종료 (나무 수 결정)', objective=p['objective'], split=how, seed=0,
                          params=dict(p, n_jobs=8, best_iter=bi), n_train=len(a), notes=f'inner_MAE={mae:.4f}', **cm.as_row()))
            json.dump(dict(best_iter=bi, source=f'{how} inner ES', inner_MAE=mae), open(bi_path, 'w'))
            print(f'{cid} {how}: inner ES best_iter {bi}, inner MAE {mae:.4f}', flush=True)
        n_est = int(round(bi * len(tr) / len(a)))
        med = float(np.median(y[tr]))
        for s in [int(v) for v in args.seeds.split(',')]:
            if lib == 'xgb':
                mk = lambda s=s: T.make_xgb(p, seed=s, n_estimators=n_est, es=False, base_score=med)
            else:
                mk = lambda s=s: T.make_lgb(p, seed=s, n_estimators=n_est)
            L.fit_record(f's3_fit_{cid}_s{s}_{how}', mk, X[tr], y[tr], X[va], how, model='XGBoost' if lib == 'xgb' else 'LightGBM',
                         features=args.features, objective=p['objective'], seed=s,
                         params=dict(p, n_estimators=n_est, n_jobs=8, tree_rule=f'best_iter {bi} x {len(tr)}/{len(a)}'),
                         stage='최종', notes=f'튜닝 설정 {cid} train 전체 재적합')
print('track cost so far', round(L.track_cost(), 4), 'core-h')
