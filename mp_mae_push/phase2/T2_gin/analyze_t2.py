"""analyze_t2.py : T2 GIN 슬롯 validation 결과 요약 (test 행은 쓰지 않는다).

입력: T2_gin/preds/T2_<kind>_<split>_s<seed>_val.csv (row,y,pred), 기준선 phase1/preds/baseline_xgb_desc217_<split>_val.csv,
      phase2/prereg/dedup_rows_<split>.csv.  출력: T2_gin/tables/*.csv, T2_gin/preds/T2_hyb_<split>_seedavg_val.csv
"""
import glob
import os

import numpy as np
import pandas as pd

import mpcommon as mc

T2 = 'T2_gin'
XGB_SEED3 = {'random': 27.20, 'scaffold': 36.32}      # 1단계 기록 (예측 파일 없음 -> MAE 차이만)


def load_runs(kind, how):
    out = {}
    for f in sorted(glob.glob(os.path.join(T2, 'preds', f'T2_{kind}_{how}_s?_val.csv'))):
        seed = int(f.split('_s')[-1].split('_')[0])
        out[seed] = pd.read_csv(f).sort_values('row').reset_index(drop=True)
    return out


def summarize():
    rows, cis, phys = [], [], []
    for how in ('random', 'scaffold'):
        base = pd.read_csv(os.path.join(mc.ROOT, 'phase1', 'preds', f'baseline_xgb_desc217_{how}_val.csv')).sort_values('row')
        dd = pd.read_csv(os.path.join(mc.ROOT, 'phase2', 'prereg', f'dedup_rows_{how}.csv'))
        keep = set(dd.loc[(dd['set'] == 'validate') & (dd['overlap'] == 0), 'row'])
        for kind in ('hyb', 'graphonly'):
            runs = load_runs(kind, how)
            if not runs:
                continue
            ref = runs[min(runs)]
            assert all((r['row'].values == ref['row'].values).all() for r in runs.values())
            assert (base['row'].values == ref['row'].values).all() and np.allclose(base['y'].values, ref['y'].values)
            y = ref['y'].values; m_dd = ref['row'].isin(keep).values
            preds = {f's{s}': r['pred'].values for s, r in runs.items()}
            if len(runs) > 1:
                preds['seedavg'] = np.mean([r['pred'].values for r in runs.values()], axis=0)
                if kind == 'hyb':
                    w = pd.DataFrame({'row': ref['row'], 'y': y, 'pred': preds['seedavg'],
                                      **{f'pred_{k}': v for k, v in preds.items() if k != 'seedavg'}})
                    w.to_csv(os.path.join(T2, 'preds', f'T2_hyb_{how}_seedavg_val.csv'), index=False)
            for k, p in preds.items():
                m = mc.metrics(y, p); md = mc.metrics(y[m_dd], p[m_dd])
                d, lo, hi = mc.paired_bootstrap(y, p, base['pred'].values)
                dd_, lod, hid = mc.paired_bootstrap(y[m_dd], p[m_dd], base['pred'].values[m_dd])
                rows.append(dict(split=how, model=kind, pred=k, n_val=len(y), MAE=m['MAE'], RMSE=m['RMSE'], R2=m['R2'],
                                 n_dedup=int(m_dd.sum()), MAE_dedup=md['MAE'],
                                 dMAE_vs_xgb=d, ci_lo=lo, ci_hi=hi, dMAE_vs_xgb_dedup=dd_, ci_lo_dedup=lod, ci_hi_dedup=hid,
                                 n_lt0=int((p < 0).sum()), n_lt50=int((p < 50).sum()), pred_min=float(p.min()),
                                 n_y_lt50=int((y < 50).sum())))
            seeds = [r for r in rows if r['split'] == how and r['model'] == kind and r['pred'][1:].isdigit()]
            if len(seeds) > 1:
                a = np.array([r['MAE'] for r in seeds]); ad = np.array([r['MAE_dedup'] for r in seeds])
                cis.append(dict(split=how, model=kind, n_seeds=len(seeds), MAE_mean=a.mean(), MAE_sd=a.std(ddof=1),
                                MAE_dedup_mean=ad.mean(), MAE_dedup_sd=ad.std(ddof=1),
                                xgb_single=mc.metrics(y, base['pred'].values)['MAE'],
                                xgb_single_dedup=mc.metrics(y[m_dd], base['pred'].values[m_dd])['MAE'],
                                xgb_seed3_mean=XGB_SEED3[how], diff_vs_xgb_seed3=a.mean() - XGB_SEED3[how]))
        # XGB 기준선 자체의 물리 불가능 예측 수 (비교용)
        pb = base['pred'].values
        phys.append(dict(split=how, model='xgb_baseline_single', n_lt0=int((pb < 0).sum()), n_lt50=int((pb < 50).sum()),
                         pred_min=float(pb.min())))
    os.makedirs(os.path.join(T2, 'tables'), exist_ok=True)
    R = pd.DataFrame(rows); S = pd.DataFrame(cis); P = pd.DataFrame(phys)
    R.to_csv(os.path.join(T2, 'tables', 'val_metrics_by_pred.csv'), index=False)
    S.to_csv(os.path.join(T2, 'tables', 'val_seed_summary.csv'), index=False)
    P.to_csv(os.path.join(T2, 'tables', 'xgb_baseline_phys.csv'), index=False)
    return R, S, P
