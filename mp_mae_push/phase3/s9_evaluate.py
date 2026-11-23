"""s9_evaluate.py : test 1회 평가 — test 라벨을 처음 읽는 유일한 스크립트.

전제: P3/tables/test_pred_sha256_main.csv 에 기록된 test 예측 파일의 sha256 이 지금 파일과 같아야 한다(다르면 중단).
평가 대상: 후보 A·B·C (+ GIN 재현 점검 실패 시 B_alt), 기준선 2개(원본 예측 파일 random / sandbox 재적합 seed 42 두 분할).
산출(P3/tables): test_results.csv, bootstrap_ci.csv, ranks.csv, mp_bin_residuals.csv, goalC_test.csv, goalC_subsets_test.csv,
  goalC_curve_test.csv, seed_metrics_test.csv; P3/preds/test/<이름>_<분할>_test.csv (row, y, pred[, pred_s*]).
안전 규칙: 후보 test MAE 가 20 K 미만이면 결과 표를 쓰지 않고 P3/notes/STOP_lt20K.txt 만 쓴 뒤 멈춘다(지시서 9절).
사용: python P3/scripts/s9_evaluate.py [--with-balt]
"""
import json
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as P  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

mc = P.mc
OUTP = os.path.join(P.P3, 'preds', 'test')
NB, BSEED = 2000, 0
BINS = [-np.inf, 250, 350, 450, 550, np.inf]
BIN_LAB = ['<250', '250-350', '350-450', '450-550', '>550']
CANDS = {'A': 'A_wavg_mse_noO2', 'B': 'B_fixed_mean_best_GIN', 'C': 'C_xgb600_F0F5_M2'}
TAU = {'A': 14.675346, 'B': 14.61442, 'C': 13.337261}     # random validation 고정값 (risk_thresholds_cands / _M2)
FLAG_COLS = ['multi_fragment', 'charged_any', 'charged_excl_nitro', 'stereo_annotated', 'stereo_possible_unspecified',
             'dblbond_stereo_unspecified', 'acyclic', 'polycyclic_ge3', 'halogen_heavy', 'carboxylic_acid', 'amide_urea',
             'hbd_ge3', 'rigid_rotb0', 'sym2d_high']
# seed 대응 조합 (seed 평균 ± SD 보고용): 멤버의 i 번째 seed 끼리 조합
SEED_MEMBERS = {
    ('A', 'random'): {'h06_yalk_F0F3F5': 0.4696, 'gin_hybrid_seedavg': 0.3877, 'lgb_r1_F0F5': 0.1232, 'xgb600_F0F1O3': 0.0196},
    ('A', 'scaffold'): {'gin_hybrid_seedavg': 0.3722, 'xgb600_F0_patent1x_scaf': 0.3178, 'h06_yalk_F0F3F5': 0.3100},
    ('B', 'random'): {'lgb_r1_F0F3F5': 0.5, 'gin_hybrid_seedavg': 0.5},
    ('B', 'scaffold'): {'lgb_r1_F0F3F5': 0.5, 'gin_hybrid_seedavg': 0.5},
    ('C', 'random'): {'xgb600_F0F5_M2': 1.0}, ('C', 'scaffold'): {'xgb600_F0F5_M2': 1.0},
    ('B_alt', 'random'): {'lgb_r1_F0F3F5': 1.0}, ('B_alt', 'scaffold'): {'lgb_r1_F0F3F5': 1.0},
}


def verify_sha():
    m = pd.read_csv(os.path.join(P.TAB, 'test_pred_sha256_main.csv'))
    bad = [r['file'] for _, r in m.iterrows() if P.sha256(os.path.join(P.TEST_DIR, r['file'])) != r['sha256']]
    if bad:
        raise SystemExit(f'sha256 불일치 — 중단: {bad}')
    return len(m)


def met(y, p):
    d = mc.metrics(y, p)
    return dict(MAE=d['MAE'], RMSE=d['RMSE'], R2=d['R2'], n=len(y))


def pb(y, a, b):
    return mc.paired_bootstrap(y, a, b, n_boot=NB, seed=BSEED)


def main():
    with_balt = '--with-balt' in sys.argv
    cands = dict(CANDS, **({'B_alt': 'Balt_lgb_r1_F0F3F5'} if with_balt else {}))
    n_files = verify_sha()
    y_all = pd.read_csv(os.path.join(mc.WEEK2, 'data', 'mp.csv'))['value'].to_numpy(float)   # ← test 라벨을 처음 읽는 곳
    preds, ys, keep = {}, {}, {}
    for how in P.SPLITS:
        idx, sdf = P.split(how)
        te = idx['test']
        ys[how] = y_all[te]
        keep[how] = P.dedup_keep(how, 'test')
        for cid, nm in cands.items():
            preds[(cid, how)] = pd.read_csv(os.path.join(P.TEST_DIR, f'{nm}_{how}_test.csv')).set_index('row').loc[te, 'pred'].to_numpy(float)
        sb = pd.read_csv(os.path.join(P.INP, 'base', f'sbx_{how}_test.csv')).set_index('row').loc[te]
        assert np.abs(sb['y'].to_numpy() - ys[how]).max() < 1e-6
        preds[('base_sbx', how)] = sb['pred'].to_numpy(float)
        if how == 'random':
            o = pd.read_csv(os.path.join(P.INP, 'base', 'orig_random_test_pred.csv'))
            smi = sdf.set_index('row').loc[te, 'smiles'].to_numpy()
            assert (o['smiles'].to_numpy() == smi).all(), 'SMILES 순서 불일치'
            assert np.abs(o['y'].to_numpy() - ys[how]).max() < 1e-6, '원본 라벨 불일치'
            preds[('base_orig', how)] = o['pred'].to_numpy(float)
    # 20 K 안전 규칙
    low = {f'{c}|{h}': mc.metrics(ys[h], preds[(c, h)])['MAE'] for (c, h) in preds if c in cands}
    if min(low.values()) < 20.0:
        os.makedirs(os.path.join(P.P3, 'notes'), exist_ok=True)
        open(os.path.join(P.P3, 'notes', 'STOP_lt20K.txt'), 'w', encoding='utf-8').write(json.dumps(low, indent=1))
        raise SystemExit('test MAE < 20 K — 결과 표를 쓰지 않고 중단 (누설 감사 먼저)')
    names = dict(cands, base_orig='원본 예측 파일(수업, XGB+RDKit217)', base_sbx='sandbox 재적합 기준선(n_jobs 8, seed 42)')
    # ---- 1. 지표
    res, ci, rk = [], [], []
    for how in P.SPLITS:
        y, k = ys[how], keep[how]
        keys = [c for (c, h) in preds if h == how]
        for c in keys:
            p = preds[(c, how)]
            for basis, m in (('전체', np.ones(len(y), bool)), ('중복 제거', k)):
                r = dict(cand=c, name=names[c], split=how, basis=basis, **met(y[m], p[m]),
                         mean_resid=float(np.mean(p[m] - y[m])))
                res.append(r)
        # 부트스트랩
        refs = [('base_sbx', 'sandbox 재적합 기준선')] + ([('base_orig', '원본 예측 파일')] if how == 'random' else [])
        pairs = [(c, r_) for c in cands for r_ in refs] + [(('A', 'B'), None), (('B', 'C'), None), (('A', 'C'), None)]
        for basis, m in (('전체', np.ones(len(y), bool)), ('중복 제거', k)):
            for item in pairs:
                if item[1] is None:
                    a, b = item[0]
                    lab = f'{a}−{b}'
                    refname = b
                else:
                    a, (b, refname) = item[0], item[1]
                    lab = f'{a}−{b}'
                d, lo, hi = pb(y[m], preds[(a, how)][m], preds[(b, how)][m])
                ci.append(dict(split=how, basis=basis, comparison=lab, new=a, ref=b, ref_name=refname, n=int(m.sum()), dMAE=d,
                               ci_lo=lo, ci_hi=hi, significant=bool(hi < 0 or lo > 0)))
        # 순위 (두 기준)
        for basis in ('전체', '중복 제거'):
            sub = [r for r in res if r['split'] == how and r['basis'] == basis]
            order = sorted(sub, key=lambda r: r['MAE'])
            for i, r in enumerate(order):
                rk.append(dict(split=how, basis=basis, cand=r['cand'], rank=i + 1, MAE=r['MAE']))
    R = pd.DataFrame(res)
    C = pd.DataFrame(ci)
    K = pd.DataFrame(rk).pivot_table(index=['split', 'cand'], columns='basis', values='rank').reset_index()
    K['rank_differs'] = K['전체'] != K['중복 제거']
    # ---- 2. seed 평균 ± SD (seed 대응 조합)
    srows = []
    for (cid, how), w in SEED_MEMBERS.items():
        if cid not in cands:
            continue
        te = P.split(how)[0]['test']
        tot = sum(w.values())
        cols = None
        mats = {}
        for mname, v in w.items():
            f = pd.read_csv(os.path.join(P.TEST_DIR, f'{mname}_{how}_test.csv')).set_index('row').loc[te]
            sc = [c for c in f.columns if c.startswith('pred_s')]
            mats[mname] = (v / tot, f[sc].to_numpy(float), sc)
        nseed = min(M.shape[1] for _, M, _ in mats.values())
        maes = []
        for i in range(nseed):
            p = sum(v * M[:, i] for v, M, _ in mats.values())
            maes.append(mc.metrics(ys[how], p)['MAE'])
        srows.append(dict(cand=cid, split=how, n_seed_combos=nseed, seed_MAE_mean=float(np.mean(maes)), seed_MAE_sd=float(np.std(maes, ddof=1)),
                          seed_MAEs=';'.join(f'{x:.4f}' for x in maes),
                          seed_columns=' | '.join(f'{m}:{",".join(c[6:] for c in sc)}' for m, (_, _, sc) in mats.items())))
    S = pd.DataFrame(srows)
    # ---- 3. MP 구간
    brows = []
    for how in P.SPLITS:
        y = ys[how]
        b = pd.cut(y, BINS, labels=BIN_LAB)
        for c in [c_ for (c_, h) in preds if h == how]:
            p = preds[(c, how)]
            for lab in BIN_LAB:
                m = np.asarray(b == lab)
                brows.append(dict(split=how, cand=c, bin=lab, n=int(m.sum()), MAE=float(np.mean(np.abs(p[m] - y[m]))),
                                  mean_resid=float(np.mean(p[m] - y[m])), CI_lo=mc.boot_ci(np.abs(p[m] - y[m]))[0],
                                  CI_hi=mc.boot_ci(np.abs(p[m] - y[m]))[1]))
    B = pd.DataFrame(brows)
    # ---- 4. 목표 C
    grows, crows, subrows = [], [], []
    rs = pd.read_csv(os.path.join(P.INP, 'ens', 'mp_replicate_stats.csv')).set_index('row')
    fl = pd.read_csv(os.path.join(P.INP, 'mol_flags.csv')).set_index('row')
    for how in P.SPLITS:
        te = P.split(how)[0]['test']
        y = ys[how]
        sc = pd.read_csv(os.path.join(P.TEST_DIR, f'goalC_score_{how}_test.csv')).set_index('row').loc[te]
        score = sc['score'].to_numpy(float)
        for cid in CANDS:
            err = np.abs(preds[(cid, how)] - y)
            o = np.argsort(score, kind='stable')
            cum = np.cumsum(err[o]) / np.arange(1, len(err) + 1)
            step = max(1, len(err) // 200)
            for kk in list(range(step - 1, len(err), step)) + [len(err) - 1]:
                crows.append(dict(split=how, cand=cid, coverage=(kk + 1) / len(err), MAE=float(cum[kk])))
            r = dict(split=how, cand=cid, n_test=len(err), AURC=float(cum.mean()),
                     **{f'MAE@{int(c * 100)}%': float(cum[max(int(round(c * len(err))) - 1, 0)]) for c in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0)})
            if how == 'random':
                tau = TAU[cid]
                sel = score <= tau
                r.update(tau=tau, n_sel=int(sel.sum()), coverage=float(sel.mean()), MAE_sel=float(err[sel].mean()) if sel.any() else np.nan)
                if sel.sum() >= 2:
                    lo, hi = mc.boot_ci(err[sel])
                    rng = np.random.default_rng(0)
                    cv_, mv_ = [], []
                    for _ in range(NB):
                        bi = rng.integers(0, len(err), len(err)); s_ = score[bi] <= tau
                        cv_.append(s_.mean()); mv_.append(err[bi][s_].mean() if s_.any() else np.nan)
                    r.update(MAE_sel_CI_rows_lo=lo, MAE_sel_CI_rows_hi=hi,
                             coverage_CI_fixtau_lo=float(np.nanpercentile(cv_, 2.5)), coverage_CI_fixtau_hi=float(np.nanpercentile(cv_, 97.5)),
                             MAE_sel_CI_fixtau_lo=float(np.nanpercentile(mv_, 2.5)), MAE_sel_CI_fixtau_hi=float(np.nanpercentile(mv_, 97.5)),
                             goalC_met_point=bool(err[sel].mean() <= 10), goalC_met_CIhi=bool(hi <= 10))
            else:
                r.update(tau='미고정(사전 예측: 미달)', n_sel=0, coverage=0.0)
            grows.append(r)
            # PREREG 7절 부분집합
            dm = keep[how]
            subs = {'전체': np.ones(len(y), bool), '중복 제거': dm}
            ybin, pbin = pd.cut(y, BINS, labels=BIN_LAB), pd.cut(preds[(cid, how)], BINS, labels=BIN_LAB)
            for lab in BIN_LAB:
                subs[f'MP구간(y) {lab}'] = np.asarray(ybin == lab)
                subs[f'MP구간(예측) {lab}'] = np.asarray(pbin == lab)
            for c in FLAG_COLS:
                subs[f'flag:{c}'] = fl.loc[te, c].astype(bool).to_numpy()
            r_ = rs.loc[te]
            nm_, rg, dpg = r_['n_meas'].fillna(1).to_numpy(), r_['range_K'].fillna(0).to_numpy(), r_['in_dpg'].astype(bool).to_numpy()
            subs['품질1 정제판(DPG)'] = dpg
            subs['품질2 반복·범위<5K(DPG 외)'] = (~dpg) & (nm_ >= 2) & (rg < 5)
            subs['품질3 반복·범위5–10K'] = (~dpg) & (nm_ >= 2) & (rg >= 5) & (rg < 10)
            subs['품질4 반복·범위≥10K'] = (~dpg) & (nm_ >= 2) & (rg >= 10)
            subs['품질5 측정 1개'] = (~dpg) & (nm_ < 2)
            subs['nn1≥0.9'] = sc['nn1'].to_numpy() >= 0.9
            subs['nn1≥0.7'] = sc['nn1'].to_numpy() >= 0.7
            for sn, mk in subs.items():
                if mk.sum() == 0:
                    continue
                lo, hi = mc.boot_ci(err[mk]) if mk.sum() > 1 else (np.nan, np.nan)
                subrows.append(dict(split=how, cand=cid, subset=sn, n=int(mk.sum()), MAE=float(err[mk].mean()), CI_lo=lo, CI_hi=hi,
                                    le10=bool(err[mk].mean() <= 10), ci_hi_le10=bool(hi <= 10)))
    G, GC, GS = pd.DataFrame(grows), pd.DataFrame(crows), pd.DataFrame(subrows)
    # ---- 5. 저장
    R.to_csv(os.path.join(P.TAB, 'test_results.csv'), index=False)
    C.to_csv(os.path.join(P.TAB, 'bootstrap_ci.csv'), index=False)
    K.to_csv(os.path.join(P.TAB, 'ranks.csv'), index=False)
    S.to_csv(os.path.join(P.TAB, 'seed_metrics_test.csv'), index=False)
    B.to_csv(os.path.join(P.TAB, 'mp_bin_residuals.csv'), index=False)
    G.to_csv(os.path.join(P.TAB, 'goalC_test.csv'), index=False)
    GC.to_csv(os.path.join(P.TAB, 'goalC_curve_test.csv'), index=False)
    GS.to_csv(os.path.join(P.TAB, 'goalC_subsets_test.csv'), index=False)
    os.makedirs(OUTP, exist_ok=True)
    for f in sorted(os.listdir(P.TEST_DIR)):
        if not f.endswith('_test.csv') or f.startswith('goalC_score'):
            continue
        how = 'random' if f.endswith('_random_test.csv') else 'scaffold'
        te = P.split(how)[0]['test']
        d = pd.read_csv(os.path.join(P.TEST_DIR, f)).set_index('row').loc[te].reset_index()
        d.insert(1, 'y', y_all[te])
        d.to_csv(os.path.join(OUTP, f), index=False)
    print(f'sha256 확인 {n_files}개 파일 일치', flush=True)
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print(R.round(3).to_string())
        print(C.round(3).to_string())
        print(K.to_string())
        print(S.round(3).to_string())
        print(G.round(3).to_string())


if __name__ == '__main__':
    with mc.CostMeter('평가') as cm:
        main()
    P.ledger(dict(run_id='P3_s9_evaluate', model='-', features='-', data_variant='test 1회 평가', split='both',
                  params={'n_boot': NB, 'seed': BSEED}, notes='test 라벨 처음 읽음; 지표·CI·목표 C', **cm.as_row()))
