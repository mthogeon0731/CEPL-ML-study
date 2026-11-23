"""T1b 최종 후보 재현 스크립트 (3단계 test 평가·train+validation 재학습용).

후보 (candidates.csv 와 같음):
  lgb_F0F3F5 : LightGBM lgb_r1 설정, 특징 F0+F3+F5 (RDKit217 + 물리·결정 39 + LLM 규칙 101 = 357열)
  lgb_F0F5   : LightGBM lgb_r1 설정, 특징 F0+F5 (318열; 3D 배좌 계산 불필요)
  h06_yalk_F0F3F5 : Yalkowsky 구조(앙상블 재료 후보; 단독 채택 아님). 이 스크립트가 아니라
                    s2_yalk.py --stage refit 으로 재현 (a, c, σ 와 나무 수는 candidates.csv / work/h06_es_*.json)
나무 수 규칙 (T1 과 같음): n_estimators = round(best_iter[split] × n_fit / n_inner[split]),
  best_iter = random 3135 / scaffold 2720 (T1 내부 85/15 조기 종료), n_inner = random 13,561 / scaffold 14,573.
  → train 만 (n 15,955/15,956) 이면 3688 / 2978 그루 = T1 재적합과 같음. train+validation 이면 비례해 늘어남.
seed 는 인자로 받는다 (기본 0,1,2; 결과로 고르지 않음). 출력 CSV: row, y, pred (seed 평균), pred_s<seed>...
사용 예:
  python train_candidate.py --cand lgb_F0F3F5 --split random --fit train --predict validate --seeds 0,1,2 --out val.csv
  python train_candidate.py --cand lgb_F0F3F5 --split random --fit train+validate --predict test --seeds 0,1,2 --out test.csv
  (행 번호 파일을 직접 줄 수도 있다: --fit-rows rows.csv --predict-rows rows2.csv, 'row' 열)
"""
import argparse, os, sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import t1blib as L

CANDS = {'lgb_F0F3F5': ('lgb_r1', 'F0+F3+F5'), 'lgb_F0F5': ('lgb_r1', 'F0+F5')}
N_INNER = {'random': 13561, 'scaffold': 14573}


def rows_of(how, spec, path):
    if path:
        return pd.read_csv(path)['row'].to_numpy()
    idx, _ = L.split(how)
    return np.concatenate([idx[k] for k in spec.split('+')])


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cand', required=True, choices=sorted(CANDS))
    ap.add_argument('--split', required=True, choices=['random', 'scaffold'])
    ap.add_argument('--fit', default='train')
    ap.add_argument('--predict', default='validate')
    ap.add_argument('--fit-rows', default='')
    ap.add_argument('--predict-rows', default='')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ledger', default='', help='주면 이 ledger.csv 에 적합 1회 = 1줄 기록')
    a = ap.parse_args()
    cfg, fs = CANDS[a.cand]
    c = L.CFG[cfg]
    X, y = L.feats(fs), L.y_all()
    fr, pr = rows_of(a.split, a.fit, a.fit_rows), rows_of(a.split, a.predict, a.predict_rows)
    n_est = int(round(c['best_iter'][a.split] * len(fr) / N_INNER[a.split]))
    out = pd.DataFrame({'row': pr, 'y': y[pr]})
    for s in [int(v) for v in a.seeds.split(',')]:
        with L.mc.CostMeter('최종') as cm:
            m = L.make_model(cfg, s, a.split, n_estimators=n_est)
            m.fit(X[fr], y[fr])
        t0 = time.perf_counter()
        out[f'pred_s{s}'] = m.predict(X[pr])
        inf = (time.perf_counter() - t0) / len(pr) * 1000
        print(f'{a.cand} {a.split} seed {s}: n_fit {len(fr)} n_est {n_est} cpu {cm.cpu_core_h:.4f} core-h', flush=True)
        if a.ledger:
            L.LEDGER = a.ledger
            L.ledger(dict(run_id=f'{a.cand}_{a.split}_{a.fit}_s{s}', model='LightGBM', features=fs, data_variant=a.fit,
                          objective=c['p']['objective'], split=a.split, seed=s,
                          params=dict(c['p'], n_estimators=n_est, threads=L.LGB_NT, cfg=cfg), n_train=len(fr),
                          infer_s_per_1k=round(inf, 5), notes=f'predict={a.predict}', **cm.as_row()))
    out['pred'] = out[[k for k in out if k.startswith('pred_s')]].mean(1)
    out[['row', 'y', 'pred'] + [k for k in out if k.startswith('pred_s')]].to_csv(a.out, index=False)
    print('saved', a.out, flush=True)
