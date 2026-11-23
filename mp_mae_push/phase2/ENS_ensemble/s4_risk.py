"""s4_risk.py : 목표 C (O5) 오차 예측 메타모델과 위험–커버리지 곡선.
점예측 대상: 최선 단일(lgb_r1_F0F3F5), 최선 앙상블(wavg_mse|noO2, 정직한 교차 적합 예측), 그 밖에 targets 인자로 받은 것.
메타모델 입력 (분자별):
  ens_sd   : OOF 가 있는 멤버들 예측의 표준편차 (random 6개; scaffold 5개 — M2 scaffold OOF 는 무작위 fold 라 뺌). O2 는 넣지 않음
  knn_gap  : |최선 단일 예측 − O2 kNN 예측|
  nn1, n07, nn5mean : 최근접 train Tanimoto, ≥0.7 이웃 수, 상위 5개 평균. train 행은 자기 자신 제외(주) / 같은 fold 제외(민감도)
  n_meas   : 반복측정 수 (phase1 mp_replicate_stats.csv)
  pred     : 최선 단일 예측값 (MP 구간 대리)
  heavy    : 무거운 원자 수 (mol_flags)
메타 학습기: LightGBM 회귀(l2, |오차| 대상), num_threads 8, seed 0.
방식 A (우선): train OOF 로 학습 (대상 = 최선 단일 OOF 의 |오차|) → validation 에 적용. 앙상블 대상에는 같은 점수를 대리로 쓴다.
방식 B: validation 5-fold × 10회 교차 적합 (대상 = 그 점예측의 |오차|).
기준 점수: ens_sd 단독, −nn1 단독.
임계값: validation 위험–커버리지에서 누적 MAE ≤ 10 K 인 최대 커버리지(최소 30분자)의 점수 = τ. 고정해 기록.
  CI: (i) 그 부분집합 행 부트스트랩, (ii) validation 전체 행 부트스트랩 + τ 고정(커버리지와 MAE 함께), (iii) 교차 검증 임계값
  (5-fold × 10: 4/5 에서 τ 선택 → 1/5 에 적용, 합친 커버리지·MAE) = test 에서 기대할 값의 정직한 추정.
"""
import os, sys, json
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import enslib as E
import s2_ensemble as S
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

TARGET_MAE = 10.0
MIN_N = 30
FEATS = ['ens_sd', 'knn_gap', 'nn1', 'n07', 'nn5mean', 'n_meas', 'pred', 'heavy']
LGBP = dict(objective='l2', learning_rate=0.03, n_estimators=400, num_leaves=15, min_child_samples=100, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, n_jobs=8, verbose=-1)
LGBP_VAL = dict(LGBP, min_child_samples=40, n_estimators=300)
BINS = [-np.inf, 250, 350, 450, 550, np.inf]
BIN_LAB = ['<250', '250-350', '350-450', '450-550', '>550']
FLAG_COLS = ['multi_fragment', 'charged_any', 'charged_excl_nitro', 'stereo_annotated', 'stereo_possible_unspecified',
             'dblbond_stereo_unspecified', 'acyclic', 'polycyclic_ge3', 'halogen_heavy', 'carboxylic_acid', 'amide_urea',
             'hbd_ge3', 'rigid_rotb0', 'sym2d_high']


def side_tables():
    rs = pd.read_csv(os.path.join(E.ENS, 'inputs', 'mp_replicate_stats.csv')).set_index('row')
    fl = pd.read_csv(os.path.join(E.WS, 'mp_mae_push', 'phase2', 'prereg', 'mol_flags.csv')).set_index('row')
    return rs, fl


def features(how, reg, kind, sim_mode='self'):
    idx, _ = E.split(how)
    rows = idx['validate' if kind == 'val' else 'train']
    oofm = S.members_for(how, 'OOF', reg, exclude=S.OOF_EXCLUDE.get(how, set()) | {'O2_knn'})
    src = S.members_for(how, kind, reg)
    mem, y = E.load_members(how, kind, {k: src[k] for k in oofm})
    o2, _ = E.load_members(how, kind, {'O2_knn': src['O2_knn']})
    sim = pd.read_csv(os.path.join(E.ENS, 'preds', f'similarity_features_{how}.csv'))
    sim = sim[sim['set'] == ('validate' if kind == 'val' else 'train')].set_index('row').loc[rows]
    rs, fl = side_tables()
    P = np.column_stack(list(mem.values()))
    X = pd.DataFrame({'ens_sd': P.std(1, ddof=1), 'knn_gap': np.abs(mem[S.BEST] - o2['O2_knn']),
                      'nn1': sim[f'nn1_{sim_mode}'].to_numpy(), 'n07': sim[f'n07_{sim_mode}'].to_numpy(),
                      'nn5mean': sim[f'nn5mean_{sim_mode}'].to_numpy(), 'n_meas': rs.loc[rows, 'n_meas'].fillna(1).to_numpy(),
                      'pred': mem[S.BEST], 'heavy': fl.loc[rows, 'heavy'].to_numpy()})
    return X, y, mem[S.BEST], list(mem)


def rc_curve(score, err):
    o = np.argsort(score, kind='stable')
    cum = np.cumsum(err[o]) / np.arange(1, len(err) + 1)
    return o, cum


def pick_tau(score, err):
    o, cum = rc_curve(score, err)
    ok = np.where((cum <= TARGET_MAE) & (np.arange(1, len(err) + 1) >= MIN_N))[0]
    if len(ok) == 0:
        return -np.inf, 0
    k = ok.max()
    return float(score[o][k]), k + 1


def threshold_report(score, err, seed=0):
    tau, k = pick_tau(score, err)
    sel = score <= tau
    out = dict(tau=tau, n_sel=int(sel.sum()), coverage=float(sel.mean()), MAE_sel=float(err[sel].mean()) if sel.any() else np.nan)
    if sel.sum() >= 2:
        out['MAE_sel_CI_rows'] = E.mc.boot_ci(err[sel])
        rng = np.random.default_rng(seed); cov, m = [], []
        for _ in range(2000):
            b = rng.integers(0, len(err), len(err)); s = score[b] <= tau
            cov.append(s.mean()); m.append(err[b][s].mean() if s.any() else np.nan)
        out['coverage_CI_fixtau'] = tuple(np.nanpercentile(cov, [2.5, 97.5]))
        out['MAE_sel_CI_fixtau'] = tuple(np.nanpercentile(m, [2.5, 97.5]))
    # 교차 검증 임계값
    covs, maes, ns = [], [], []
    for r in range(10):
        sel_all = np.zeros(len(err), bool)
        for i, j in KFold(5, shuffle=True, random_state=r).split(err):
            t, _ = pick_tau(score[i], err[i]); sel_all[j] = score[j] <= t
        covs.append(sel_all.mean()); maes.append(err[sel_all].mean() if sel_all.any() else np.nan); ns.append(sel_all.sum())
    out.update(cv_coverage=float(np.mean(covs)), cv_coverage_sd=float(np.std(covs, ddof=1)), cv_MAE_sel=float(np.nanmean(maes)),
               cv_MAE_sel_sd=float(np.nanstd(maes, ddof=1)), cv_frac_rep_le10=float(np.mean(np.array(maes) <= TARGET_MAE)))
    return out


def cov_mae(score, err, covs=(0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)):
    o, cum = rc_curve(score, err)
    return {f'MAE@{int(c * 100)}%': float(cum[max(int(round(c * len(err))) - 1, 0)]) for c in covs} | {'AURC': float(cum.mean())}


def main(targets=None, tag=''):
    reg = pd.read_csv(os.path.join(E.ENS, E.MEMBERS_FILE))
    rs, fl = side_tables()
    rows, crows, srows, thr_rows, imp_rows = [], [], [], [], []
    for how in E.SPLITS:
        with E.mc.CostMeter('탐색') as cm:
            idx, _ = E.split(how)
            va = idx['validate']
            Xv, yv, pbest_v, oofnames = features(how, reg, 'val')
            tg = {'best_single': pbest_v,
                  'best_ens': pd.read_csv(os.path.join(E.ENS, 'preds', 'ens', f'{how}__wavg_mse__noO2_val_honest.csv'))['pred'].to_numpy()}
            if targets:
                for k, f in targets.items():
                    tg[k] = pd.read_csv(os.path.join(E.ENS, 'preds', 'ens', f.format(how=how)))['pred'].to_numpy()
            scores = {}
            for sim_mode in ('self', 'fold'):
                Xt, yt, pbest_t, _ = features(how, reg, 'OOF', sim_mode)
                et = np.abs(yt - pbest_t)
                m = lgb.LGBMRegressor(**LGBP).fit(Xt[FEATS], et)
                scores[f'metaA_trainOOF_{sim_mode}'] = m.predict(Xv[FEATS])
                imp_rows += [dict(split=how, model=f'metaA_trainOOF_{sim_mode}', feature=f, gain=float(g))
                             for f, g in zip(FEATS, m.booster_.feature_importance('gain'))]
            scores['ens_sd_only'] = Xv['ens_sd'].to_numpy()
            scores['neg_nn1_only'] = -Xv['nn1'].to_numpy()
            for tname, p in tg.items():
                err = np.abs(yv - p)
                sc = dict(scores)
                H = np.zeros((10, len(yv)))
                for r in range(10):
                    for i, j in KFold(5, shuffle=True, random_state=r).split(Xv):
                        H[r, j] = lgb.LGBMRegressor(**LGBP_VAL).fit(Xv.iloc[i][FEATS], err[i]).predict(Xv.iloc[j][FEATS])
                sc['metaB_valCV'] = H.mean(0)
                sc['oracle'] = err
                for sname, s in sc.items():
                    o, cum = rc_curve(s, err)
                    rows.append(dict(split=how, target=tname, score=sname, MAE_all=float(err.mean()), n=len(err),
                                     spearman_score_err=float(pd.Series(s).corr(pd.Series(err), method='spearman')), **cov_mae(s, err)))
                    step = max(1, len(err) // 200)
                    for k in list(range(step - 1, len(err), step)) + [len(err) - 1]:
                        crows.append(dict(split=how, target=tname, score=sname, coverage=(k + 1) / len(err), MAE=float(cum[k])))
                    if sname != 'oracle':
                        tr = threshold_report(s, err)
                        thr_rows.append(dict(split=how, target=tname, score=sname, **{k: (json.dumps([round(v, 3) for v in vv]) if isinstance(vv, tuple) else vv)
                                                                                       for k, vv in tr.items()}))
                # 분자별 점수 저장
                pd.DataFrame({'row': va, 'y': yv, 'pred': p, 'abs_err': err, **{f'score_{k}': v for k, v in sc.items() if k != 'oracle'},
                              **{f'feat_{c}': Xv[c].to_numpy() for c in FEATS}}).to_csv(
                    os.path.join(E.ENS, 'preds', f'risk_scores_{how}_{tname}{tag}.csv'), index=False)
                # PREREG 7절 부분집합
                dm = E.dedup_mask(how)
                subs = {'전체': np.ones(len(yv), bool), '중복 제거': dm}
                ybin = pd.cut(yv, BINS, labels=BIN_LAB); pbin = pd.cut(p, BINS, labels=BIN_LAB)
                for b in BIN_LAB:
                    subs[f'MP구간(y) {b}'] = np.asarray(ybin == b)
                    subs[f'MP구간(예측) {b}'] = np.asarray(pbin == b)
                for c in FLAG_COLS:
                    subs[f'flag:{c}'] = fl.loc[va, c].astype(bool).to_numpy()
                r_ = rs.loc[va]
                nm, rg, dpg = r_['n_meas'].fillna(1).to_numpy(), r_['range_K'].fillna(0).to_numpy(), r_['in_dpg'].astype(bool).to_numpy()
                subs['품질1 정제판(DPG)'] = dpg
                subs['품질2 반복·범위<5K(DPG 외)'] = (~dpg) & (nm >= 2) & (rg < 5)
                subs['품질3 반복·범위5–10K'] = (~dpg) & (nm >= 2) & (rg >= 5) & (rg < 10)
                subs['품질4 반복·범위≥10K'] = (~dpg) & (nm >= 2) & (rg >= 10)
                subs['품질5 측정 1개'] = (~dpg) & (nm < 2)   # DPG 목록에는 반복 수 1인 분자가 몇 개 있다 → DPG 우선
                subs['nn1≥0.9'] = Xv['nn1'].to_numpy() >= 0.9
                subs['nn1≥0.7'] = Xv['nn1'].to_numpy() >= 0.7
                for sn, mk in subs.items():
                    if mk.sum() == 0:
                        continue
                    lo, hi = E.mc.boot_ci(err[mk]) if mk.sum() > 1 else (np.nan, np.nan)
                    srows.append(dict(split=how, target=tname, subset=sn, n=int(mk.sum()), MAE=float(err[mk].mean()), CI_lo=lo, CI_hi=hi,
                                      le10=int(err[mk].mean() <= TARGET_MAE), ci_hi_le10=int(hi <= TARGET_MAE)))
            print(how, 'done', flush=True)
        E.ledger(dict(run_id=f'O5_risk_{how}{tag}', model='LightGBM |오차| 메타모델', features=','.join(FEATS), data_variant='train OOF 학습 / validation 교차 적합',
                      objective='l2 on |err|', split=how, seed=0, params=dict(LGBP, threads=8, oof_members=oofnames), n_train=len(yv), **cm.as_row()))
    out = {k: pd.DataFrame(v) for k, v in dict(summary=rows, curve=crows, subsets=srows, thresholds=thr_rows, importance=imp_rows).items()}
    for k, d in out.items():
        d.to_csv(os.path.join(E.ENS, 'tables', f'risk_{k}{tag}.csv'), index=False)
    return out
