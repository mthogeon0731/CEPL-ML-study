"""s1_label_noise.py : 라벨 잡음 하한 추정 (네트워크 없이).

(a) README 통계 3개(반복측정 4,679종 중 범위 >= 10 K 410종, >= 50 K 86종; 평균 반복 수)에
    측정 오차 분포를 맞춰 단일 측정 / 중앙값 병합 라벨의 기대 |오차| 와 test MAE 하한을 계산.
(b) mp.csv 안의 직접 표본: InChIKey 전체가 같은데 SMILES 가 달라 따로 남은 행들의 라벨 차이,
    첫 블록만 같은 쌍(입체 이성질체 등)의 라벨 차이.
출력: phase1/tables/label_noise_floor.csv, label_noise_indata.csv, label_noise_indata_pairs.csv
"""
import itertools
import os

import numpy as np
import pandas as pd
from scipy import optimize, stats

import mpcommon as mc

OUT = os.path.join(mc.ROOT, 'phase1', 'tables')
LEDGER = os.path.join(mc.ROOT, 'phase1', 'ledger_errdecomp.csv')
os.makedirs(OUT, exist_ok=True)

# ---- README 통계 ---------------------------------------------------------------------------
N_RAW, N_MERGED, N_REP = 27976, 19945, 4679
N_DUP_ROWS = N_RAW - N_MERGED                 # 8,031 병합된 행
N_MEAS_REP = N_REP + N_DUP_ROWS               # 반복측정 화합물의 측정 수 합 = 12,710
K_MEAN = N_MEAS_REP / N_REP                   # 2.7164
P10, P50 = 410 / N_REP, 86 / N_REP            # 0.08763, 0.01838
F_SINGLE = (N_MERGED - N_REP) / N_MERGED      # 측정 1개뿐인 화합물 비율 0.7654
BASE_MAE = 27.4995                            # 원본 기준선 random test MAE [K]

# ---- 격자 위의 분포 (수치 적분) ---------------------------------------------------------------
U = np.linspace(-np.arcsinh(3000.0), np.arcsinh(3000.0), 40001)
XG = np.sinh(U)                               # 0 근처 촘촘, +-3000 K 까지


class Dist:
    """격자 위의 대칭 오차 분포. pdf 값을 받아 정규화하고 cdf 를 누적 사다리꼴로 만든다."""

    def __init__(self, pdf):
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(XG))])
        self.pdf, self.cdf = pdf / cdf[-1], cdf / cdf[-1]

    def F(self, x):
        return np.interp(x, XG, self.cdf, left=0.0, right=1.0)

    def p_range_ge(self, r, k):
        """k 개 독립 측정의 범위가 r 이상일 확률: 1 - k * int f(x) [F(x+r)-F(x)]^(k-1) dx"""
        g = self.pdf * (self.F(XG + r) - self.cdf) ** (k - 1)
        return 1.0 - k * np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(XG))

    def mean_abs(self):
        g = self.pdf * np.abs(XG)
        return float(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(XG)))

    def rms(self):
        g = self.pdf * XG ** 2
        return float(np.sqrt(np.sum(0.5 * (g[1:] + g[:-1]) * np.diff(XG))))

    def sample(self, rng, size):
        return np.interp(rng.random(size), self.cdf, XG)


def kdist(kind):
    """반복 수 k 의 분포 (평균 = K_MEAN). k <= 12 에서 자른 뒤 재정규화."""
    lam = K_MEAN - 2.0
    ks = np.arange(2, 13)
    if kind == 'two_point':          # k = 2 또는 3
        return {2: 1 - lam, 3: lam}
    if kind == 'poisson':            # k = 2 + Poisson(lam)
        p = stats.poisson.pmf(ks - 2, lam)
    elif kind == 'geometric':        # k = 2 + Geometric(평균 lam)
        q = 1.0 / (1.0 + lam)
        p = q * (1 - q) ** (ks - 2)
    else:
        raise ValueError(kind)
    return dict(zip(ks.tolist(), (p / p.sum()).tolist()))


def p_range(dist, r, kd):
    return sum(w * dist.p_range_ge(r, k) for k, w in kd.items())


def make_dist(family, theta, fixed):
    """theta = 풀어야 하는 두 모수(변환된 값)."""
    if family == 'normal_mix':       # (1-w) N(0, s1^2) + w N(0, s2^2); s1 고정
        w, s2 = 1 / (1 + np.exp(-theta[0])), np.exp(theta[1])
        pdf = (1 - w) * stats.norm.pdf(XG, scale=fixed['s1']) + w * stats.norm.pdf(XG, scale=s2)
        par = dict(w_tail=w, core_sd_K=fixed['s1'], tail_scale_K=s2)
    elif family == 'normal_laplace':  # (1-w) N(0, s1^2) + w Laplace(b)
        w, b = 1 / (1 + np.exp(-theta[0])), np.exp(theta[1])
        pdf = (1 - w) * stats.norm.pdf(XG, scale=fixed['s1']) + w * stats.laplace.pdf(XG, scale=b)
        par = dict(w_tail=w, core_sd_K=fixed['s1'], tail_scale_K=b)
    elif family == 'student_t_trunc':  # 척도 s, 자유도 nu 의 t 분포를 +-T 에서 자름
        nu, s = np.exp(theta[0]), np.exp(theta[1])
        pdf = stats.t.pdf(XG / s, df=nu) / s
        pdf = np.where(np.abs(XG) <= fixed['T'], pdf, 0.0)
        par = dict(w_tail=np.nan, core_sd_K=s, tail_scale_K=nu)
    else:
        raise ValueError(family)
    return Dist(pdf), par


def fit(family, fixed, kd, copy_frac, x0):
    """독립인 반복 세트에서 P(range>=10), P(range>=50) 이 README 값/(1-copy_frac) 이 되도록 푼다."""
    t10, t50 = P10 / (1 - copy_frac), P50 / (1 - copy_frac)

    def resid(theta):
        d, _ = make_dist(family, theta, fixed)
        return [np.log(max(p_range(d, 10.0, kd), 1e-12) / t10), np.log(max(p_range(d, 50.0, kd), 1e-12) / t50)]

    sol = optimize.least_squares(resid, x0, xtol=1e-12, ftol=1e-12, max_nfev=400)
    d, par = make_dist(family, sol.x, fixed)
    ok = bool(np.max(np.abs(sol.fun)) < 1e-4)
    return d, par, ok, p_range(d, 10.0, kd) * (1 - copy_frac), p_range(d, 50.0, kd) * (1 - copy_frac)


def label_errors(dist, kd, copy_frac, rng, n=1_000_000):
    """단일 측정 오차와 중앙값 병합 라벨 오차의 표본.
    copy_frac: 반복 세트 중 '같은 값을 베껴 적은 사본'(독립 아님)의 비율 -> 병합해도 단일 측정과 같다."""
    e1 = dist.sample(rng, n)
    med = []
    for k, w in kd.items():
        nk = int(round(n * w))
        if nk:
            med.append(np.median(dist.sample(rng, (nk, k)), axis=1))
    med = np.concatenate(med)
    n_copy = int(round(len(med) * copy_frac))
    if n_copy:
        med[:n_copy] = dist.sample(rng, n_copy)
    return e1, med


def noise_free_mae(e_label, rng, shape):
    """기준선 오차 = (모델의 참값 대비 오차 m) + (test 라벨 오차 e), 서로 독립이라 가정.
    E|m - e| = BASE_MAE 가 되는 m 의 척도를 찾아 E|m| (= 라벨이 무잡음일 때의 기준선 MAE) 를 돌려준다."""
    z = rng.laplace(size=len(e_label)) if shape == 'laplace' else rng.normal(size=len(e_label))
    f = lambda s: np.mean(np.abs(s * z - e_label)) - BASE_MAE
    s = optimize.brentq(f, 1.0, 100.0)
    return float(np.mean(np.abs(s * z)))


SCENARIOS = [  # (id, family, fixed, k 분포, 사본 비율, 초기값, 가정 설명)
    ('A1', 'normal_mix', dict(s1=0.5), 'poisson', 0.0, [-2.5, 3.3], '정규 2성분 혼합, 핵 SD 0.5 K 고정'),
    ('A2', 'normal_mix', dict(s1=1.0), 'poisson', 0.0, [-2.5, 3.3], '정규 2성분 혼합, 핵 SD 1 K 고정'),
    ('A3', 'normal_mix', dict(s1=2.0), 'poisson', 0.0, [-2.5, 3.3], '정규 2성분 혼합, 핵 SD 2 K 고정 (기준 가정)'),
    ('A4', 'normal_mix', dict(s1=3.0), 'poisson', 0.0, [-3.0, 3.6], '정규 2성분 혼합, 핵 SD 3 K 고정'),
    ('A5', 'normal_mix', dict(s1=2.0), 'two_point', 0.0, [-2.5, 3.3], 'A3 + 반복 수 k 가 2 또는 3'),
    ('A6', 'normal_mix', dict(s1=2.0), 'geometric', 0.0, [-2.5, 3.3], 'A3 + 반복 수 k = 2 + 기하분포'),
    ('B1', 'normal_laplace', dict(s1=2.0), 'poisson', 0.0, [-2.5, 3.0], '정규 핵(SD 2 K) + Laplace 꼬리'),
    ('C1', 'student_t_trunc', dict(T=200.0), 'poisson', 0.0, [0.0, 0.5], 'Student-t (두꺼운 꼬리), |오차| <= 200 K 에서 자름'),
    ('C2', 'student_t_trunc', dict(T=100.0), 'poisson', 0.0, [0.0, 0.5], 'Student-t (두꺼운 꼬리), |오차| <= 100 K 에서 자름'),
    ('D1', 'normal_mix', dict(s1=2.0), 'poisson', 0.25, [-2.2, 3.3], 'A3 + 반복 세트의 25 % 는 같은 값을 옮겨 적은 사본(독립 아님)'),
    ('D2', 'normal_mix', dict(s1=2.0), 'poisson', 0.50, [-1.8, 3.3], 'A3 + 반복 세트의 50 % 는 사본'),
    ('D3', 'normal_mix', dict(s1=2.0), 'poisson', 0.75, [-1.0, 3.3], 'A3 + 반복 세트의 75 % 는 사본 (극단 가정)'),
]

rows = []
with mc.CostMeter('탐색') as cost:
    rng = np.random.default_rng(0)
    for sid, fam, fixed, kkind, cfrac, x0, desc in SCENARIOS:
        kd = kdist(kkind)
        d, par, ok, f10, f50 = fit(fam, fixed, kd, cfrac, x0)
        e1, med = label_errors(d, kd, cfrac, rng)
        n1 = int(round(len(med) * F_SINGLE / (1 - F_SINGLE)))
        e_label = np.concatenate([d.sample(rng, n1), med])     # test 라벨 오차 표본 (단일 76.5 % + 병합 23.5 %)
        e_label = rng.permutation(e_label)
        e_a, e_b = d.sample(rng, 500_000), d.sample(rng, 500_000)
        rows.append(dict(
            scenario=sid, family=fam, assumption=desc, k_dist=kkind, copy_frac=cfrac, fit_ok=ok,
            fit_P_range_ge10=f10, fit_P_range_ge50=f50, target_P_range_ge10=P10, target_P_range_ge50=P50,
            w_tail=par['w_tail'], core_sd_or_t_scale_K=par['core_sd_K'], tail_scale_K_or_t_df=par['tail_scale_K'],
            single_meas_mean_abs_err_K=d.mean_abs(), single_meas_rms_err_K=d.rms(),
            median_label_mean_abs_err_K=float(np.mean(np.abs(med))),
            floor_MAE_K=F_SINGLE * d.mean_abs() + (1 - F_SINGLE) * float(np.mean(np.abs(med))),
            floor_RMSE_K=float(np.sqrt(F_SINGLE * d.rms() ** 2 + (1 - F_SINGLE) * np.mean(med ** 2))),
            two_indep_meas_mean_abs_diff_K=float(np.mean(np.abs(e_a - e_b))),
            two_indep_meas_P_diff_ge10=float(np.mean(np.abs(e_a - e_b) >= 10)),
            two_indep_meas_P_diff_ge50=float(np.mean(np.abs(e_a - e_b) >= 50)),
            baseline_MAE_if_noise_free_laplace_K=noise_free_mae(e_label, rng, 'laplace'),
            baseline_MAE_if_noise_free_normal_K=noise_free_mae(e_label, rng, 'normal'),
            n_rep_compounds=N_REP, mean_k=K_MEAN, frac_single_label=F_SINGLE, mc_n=len(e_label), mc_seed=0))
        r = rows[-1]
        print(f"{sid} ok={ok} P10={f10:.4f} P50={f50:.4f} w={par['w_tail']:.4f} s/core={par['core_sd_K']:.3f} "
              f"tail={par['tail_scale_K']:.2f} | E|e1|={r['single_meas_mean_abs_err_K']:.2f} "
              f"E|med|={r['median_label_mean_abs_err_K']:.2f} floor={r['floor_MAE_K']:.2f} K "
              f"| noise-free base {r['baseline_MAE_if_noise_free_laplace_K']:.2f}/{r['baseline_MAE_if_noise_free_normal_K']:.2f}")

    # ---- (b) 데이터 안 직접 표본 -------------------------------------------------------------
    df, mols, y = mc.load_data()
    s = pd.read_csv(os.path.join(mc.SPLIT_DIR, 'random.csv')).sort_values('row').reset_index(drop=True)
    assert (s['smiles'].values == df['smiles'].values).all()
    s['y'] = y
    s['block1'] = s['inchikey'].str[:14]
    pairs = []
    for key, g in s.groupby('inchikey'):
        if len(g) > 1:
            for (i, a), (j, b) in itertools.combinations(g[['row', 'y', 'smiles']].iterrows(), 2):
                pairs.append(dict(kind='same_full_inchikey', key=key, row_a=int(a['row']), row_b=int(b['row']),
                                  smiles_a=a['smiles'], smiles_b=b['smiles'], y_a=a['y'], y_b=b['y'],
                                  abs_diff_K=abs(a['y'] - b['y'])))
    for key, g in s.groupby('block1'):
        if g['inchikey'].nunique() > 1:
            recs = g[['row', 'y', 'smiles', 'inchikey']].to_dict('records')
            for a, b in itertools.combinations(recs, 2):
                if a['inchikey'] != b['inchikey']:
                    pairs.append(dict(kind='same_block1_diff_full', key=key, row_a=int(a['row']), row_b=int(b['row']),
                                      smiles_a=a['smiles'], smiles_b=b['smiles'], y_a=a['y'], y_b=b['y'],
                                      abs_diff_K=abs(a['y'] - b['y'])))
    P = pd.DataFrame(pairs)
    P.to_csv(os.path.join(OUT, 'label_noise_indata_pairs.csv'), index=False, encoding='utf-8-sig')
    summ = []
    for kind, g in P.groupby('kind'):
        d_ = g['abs_diff_K'].values
        lo, hi = mc.boot_ci(d_)
        summ.append(dict(kind=kind, n_pairs=len(g), n_groups=g['key'].nunique(), mean_abs_diff_K=d_.mean(),
                         mean_ci_lo=lo, mean_ci_hi=hi, median_abs_diff_K=np.median(d_),
                         q75_K=np.quantile(d_, 0.75), q90_K=np.quantile(d_, 0.90), max_K=d_.max(),
                         frac_lt1=np.mean(d_ < 1), frac_ge10=np.mean(d_ >= 10), frac_ge50=np.mean(d_ >= 50),
                         implied_single_label_MAE_lo_K=d_.mean() / 2, implied_single_label_MAE_normal_K=d_.mean() / np.sqrt(2),
                         implied_single_label_MAE_hi_K=d_.mean()))
    S = pd.DataFrame(summ)
    S['n_unique_full_inchikey'] = s['inchikey'].nunique()
    S['n_unique_block1'] = s['block1'].nunique()
    S['n_rows'] = len(s)
    S.to_csv(os.path.join(OUT, 'label_noise_indata.csv'), index=False, encoding='utf-8-sig')
    print(S.T.to_string())

T = pd.DataFrame(rows)
T.to_csv(os.path.join(OUT, 'label_noise_floor.csv'), index=False, encoding='utf-8-sig')
mc.append_ledger(LEDGER, dict(run_id='errdecomp_s1_label_noise', track='errdecomp', model='noise-model (no ML fit)',
                              features='-', data_variant='README stats + mp.csv InChIKey duplicates', objective='-',
                              split='-', seed=0, params='{"scenarios": %d, "mc_n": 1000000}' % len(rows),
                              n_train=0, notes='라벨 잡음 하한 추정; 모델 적합 없음', **cost.as_row()))
print(f'K_MEAN={K_MEAN:.4f} P10={P10:.5f} P50={P50:.5f} F_SINGLE={F_SINGLE:.4f}')
print(f"floor range (fit_ok): {T.loc[T.fit_ok, 'floor_MAE_K'].min():.2f} - {T.loc[T.fit_ok, 'floor_MAE_K'].max():.2f} K")
