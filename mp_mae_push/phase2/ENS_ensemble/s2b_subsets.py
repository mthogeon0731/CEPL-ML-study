"""s2b_subsets.py : 멤버 수를 줄인 고정 부분집합 앙상블 (2–4개) 평가 + 후보 간 짝지은 비교.
고정 부분집합은 s2 결과(전진 선택에서 GIN 이 늘 뽑힘)를 본 뒤 정한 것이므로 '사후 지정'이다. 가중치는 s2 와 같은
validation 5-fold × 10회 교차 적합으로 정직하게 추정하지만, 부분집합 선택 자체의 낙관은 s2 의 fwd{m}_mae (선택까지 교차 적합) 로 가늠한다.
방법: mean (가중치 없음, 적합 없음), wavg_mae, wavg_mse.
"""
import os, sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enslib as E
import s2_ensemble as S
import numpy as np, pandas as pd

GIN = 'gin_hybrid_seedavg'
FIXED = {
    'best+GIN': ['lgb_r1_F0F3F5', GIN],
    'F0F5+GIN': ['lgb_r1_F0F5', GIN],
    'h06+GIN': ['h06_yalk_F0F3F5', GIN],
    'lgbF0+GIN': ['lgb_r1_F0', GIN],
    'xgbr1+GIN': ['xgb_r1_F0', GIN],
    'M2+GIN': ['xgb600_F0F5_M2', GIN],
    'base+GIN': ['baseline_xgb600_F0', GIN],
    'bag10+GIN': ['xgb600_F0_bag10', GIN],
    'F0F3+GIN': ['xgb600_F0F3', GIN],
    'best+h06+GIN': ['lgb_r1_F0F3F5', 'h06_yalk_F0F3F5', GIN],
    'F0F5+h06+GIN': ['lgb_r1_F0F5', 'h06_yalk_F0F3F5', GIN],
    'h06+patent+GIN': ['h06_yalk_F0F3F5', 'xgb600_F0_patent1x_scaf', GIN],
    'best+patent+GIN': ['lgb_r1_F0F3F5', 'xgb600_F0_patent1x_scaf', GIN],
    'F0F5+patent+GIN': ['lgb_r1_F0F5', 'xgb600_F0_patent1x_scaf', GIN],
    'best+h06': ['lgb_r1_F0F3F5', 'h06_yalk_F0F3F5'],
    'best+F0F5+h06+GIN': ['lgb_r1_F0F3F5', 'lgb_r1_F0F5', 'h06_yalk_F0F3F5', GIN],
}


def main(tag='', extra=None):
    fixed = dict(FIXED)
    if extra:
        fixed.update(extra)
    reg = pd.read_csv(os.path.join(E.ENS, E.MEMBERS_FILE))
    rows, wrows = [], []
    for how in E.SPLITS:
        with E.mc.CostMeter('탐색') as cm:
            mv, y = E.load_members(how, 'val', S.members_for(how, 'val', reg))
            dm = E.dedup_mask(how)
            best, base = mv[S.BEST], mv[S.BASE]
            for sname, names in fixed.items():
                if any(n not in mv for n in names):
                    continue
                P = np.column_stack([mv[n] for n in names])
                for meth in ('mean_all', 'wavg_mae', 'wavg_mse'):
                    H = S.crossfit(meth, P, y, names) if meth != 'mean_all' else np.tile(P.mean(1), (S.N_REP, 1))
                    h = H.mean(0); repm = [E.mae(y, H[r]) for r in range(S.N_REP)]
                    fin = S.fit_method(meth, P, y, names)
                    wd = {names[i]: round(float(fin.w[i]), 4) for i in range(len(names))}
                    meth_l = 'mean' if meth == 'mean_all' else meth
                    pd.DataFrame({'row': E.split(how)[0]['validate'], 'y': y, 'pred': h}).to_csv(
                        os.path.join(E.ENS, 'preds', 'ens', f'{how}__fixed_{meth_l}__{sname}{tag}_val_honest.csv'), index=False)
                    rows.append(dict(split=how, method=f'fixed_{meth_l}', member_set=sname, n_members=len(names), MAE=E.mae(y, h),
                                     MAE_dedup=E.mae(y[dm], h[dm]), rep_mean=np.mean(repm), rep_sd=np.std(repm, ddof=1), n=len(y),
                                     n_dedup=int(dm.sum()), has_O2=0, insample_MAE_final=E.mae(y, fin(P)),
                                     **S.compare_rows(y, h, best, dm, 'best'), **S.compare_rows(y, h, base, dm, 'base')))
                    wrows.append(dict(split=how, method=f'fixed_{meth_l}', member_set=sname, intercept=0.0, alpha='', selected=';'.join(names),
                                      weights=json.dumps(wd)))
        E.ledger(dict(run_id=f'H10_fixed_subsets_{how}{tag}', model='앙상블 (고정 부분집합)', features='val 예측', data_variant='validation 5-fold×10 교차 적합',
                      objective='MAE/MSE', split=how, seed='fold seed 0-9', params=dict(subsets=list(fixed)), n_train=len(y),
                      notes='사후 지정 부분집합', **cm.as_row()))
    pd.DataFrame(rows).to_csv(os.path.join(E.ENS, 'tables', f'ensemble_fixed_subsets{tag}.csv'), index=False)
    pd.DataFrame(wrows).to_csv(os.path.join(E.ENS, 'tables', f'ensemble_fixed_weights{tag}.csv'), index=False)
    return pd.DataFrame(rows)
