"""1단계 요약: (모델, 특징 묶음, 분할)별 seed MAE 평균±SD, seed 평균 예측의 전체·중복 제거 MAE,
같은 모델 F0(T1 재적합 seed 3 평균 예측) 대비와 기준선(XGB 600그루 단일 seed) 대비 짝지은 부트스트랩 95 % CI, 두 기준 순위.
출력: tables/step1_val_summary.csv"""
import os, sys, glob
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import t1blib as L

REF_F0 = {'lgb_r1': 'lgb_r1_tuned_s3', 'xgb_r1': 'xgb_r1_tuned_s3'}


def seed_preds(prefix, how, seeds=(0, 1, 2)):
    ps = [np.load(os.path.join(L.WORK, f'{prefix}_{how}_s{s}.npy')) for s in seeds
          if os.path.exists(os.path.join(L.WORK, f'{prefix}_{how}_s{s}.npy'))]
    return ps


def summarize(rows_spec):
    """rows_spec: list of dict(name, cfg, how, preds(list) or pred_mean, ref_name)."""
    out = []
    for r in rows_spec:
        how = r['how']
        idx, _ = L.split(how)
        y = L.y_all()[idx['validate']]
        ps = r.get('preds')
        pm = np.mean(ps, 0) if ps else r['pred_mean']
        smae = [np.mean(np.abs(p - y)) for p in ps] if ps else []
        ref_same = L.load_ref_val(REF_F0[r['cfg']], how) if r.get('cfg') in REF_F0 else None
        base = L.load_ref_val('baseline_xgb_desc217', how)
        e_same = L.evaluate(how, pm, ref_same) if ref_same is not None else {}
        e_base = L.evaluate(how, pm, base)
        out.append(dict(name=r['name'], model=r.get('cfg', ''), features=r.get('fs', ''), split=how, n_seeds=len(ps) if ps else r.get('n_seeds', ''),
                        seed_MAE_mean=np.mean(smae) if smae else np.nan, seed_MAE_sd=np.std(smae, ddof=1) if len(smae) > 1 else np.nan,
                        val_MAE=e_base['val_MAE'], dedup_MAE=e_base['dedup_MAE'],
                        d_vs_F0=e_same.get('dMAE'), F0_lo=e_same.get('ci_lo'), F0_hi=e_same.get('ci_hi'),
                        d_vs_F0_dedup=e_same.get('dMAE_dedup'), F0_lo_dedup=e_same.get('ci_lo_dedup'), F0_hi_dedup=e_same.get('ci_hi_dedup'),
                        d_vs_base=e_base['dMAE'], base_lo=e_base['ci_lo'], base_hi=e_base['ci_hi'],
                        d_vs_base_dedup=e_base['dMAE_dedup'], base_lo_dedup=e_base['ci_lo_dedup'], base_hi_dedup=e_base['ci_hi_dedup'],
                        note=r.get('note', '')))
    df = pd.DataFrame(out)
    for how in df['split'].unique():
        m = df['split'] == how
        df.loc[m, 'rank_full'] = df.loc[m, 'val_MAE'].rank(method='min').astype(int)
        df.loc[m, 'rank_dedup'] = df.loc[m, 'dedup_MAE'].rank(method='min').astype(int)
    df['rank_differs'] = np.where(df['rank_full'] != df['rank_dedup'], '*', '')
    return df


def step1_spec():
    spec = []
    for how in ('random', 'scaffold'):
        spec.append(dict(name='baseline_xgb600', how=how, pred_mean=L.load_ref_val('baseline_xgb_desc217', how), n_seeds=1,
                         fs='F0', note='기준선 (1단계, 단일 seed)'))
        for cfg in ('lgb_r1', 'xgb_r1'):
            spec.append(dict(name=f'{cfg}_F0', cfg=cfg, fs='F0', how=how, pred_mean=L.load_ref_val(REF_F0[cfg], how), n_seeds=3,
                             note='T1 재적합 예측 재사용 (seed 0-2 평균; LGB 8 스레드)'))
            for fs in ('F0+F3', 'F0+F5', 'F0+F3+F5'):
                ps = seed_preds(f"s1_{cfg}_{fs.replace('+', '')}", how)
                if ps:
                    spec.append(dict(name=f'{cfg}_{fs}', cfg=cfg, fs=fs, how=how, preds=ps))
        ps = seed_preds('s1_lgb_r1_F0', how, seeds=(0,))
        if ps:
            spec.append(dict(name='lgb_r1_F0_rerun6thr', cfg='lgb_r1', fs='F0', how=how, preds=ps, note='재현 점검 (6 스레드, seed 0)'))
    return spec


if __name__ == '__main__':
    df = summarize(step1_spec())
    os.makedirs(os.path.join(L.TDIR, 'tables'), exist_ok=True)
    df.to_csv(os.path.join(L.TDIR, 'tables', 'step1_val_summary.csv'), index=False, encoding='utf-8-sig')
    print(df.round(3).to_string())
