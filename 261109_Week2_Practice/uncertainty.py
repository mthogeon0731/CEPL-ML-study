"""uncertainty.py : 3주차. 예측의 불확실성(σ)을 재고, 그 σ 로 다음에 잴 분자를 고른다.

    from uncertainty import split_smiles, morgan, rf_sigma, gp_sigma, calibration, max_similarity, active_learning
    d = split_smiles('logP', max_n=3000, how='random')     # train.run 과 같은 분할 (같은 seed, 같은 함수)
    Xtr, Xte = morgan(d['train'].smiles), morgan(d['test'].smiles)
    mu, sd = gp_sigma(Xtr, d['train'].y, Xte)               # Tanimoto 커널 GP

σ 를 내는 방법 세 가지
    1. rf_sigma   : Random Forest 나무 300그루의 예측 퍼짐 (앙상블)
    2. BGNN-GIN   : train.run(bayes=True) 의 마지막 층 가중치 표본 100개 퍼짐 (Bayes by Backprop)
    3. gp_sigma   : 가우시안 과정. 학습 분자와 닮은 정도(Tanimoto)로 σ 가 정해진다
기존 파일(feature.py, datamanager.py, models.py, train.py)은 고치지 않고 import 만 한다.
"""
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import norm, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, ConstantKernel, WhiteKernel

from datamanager import load_data, gen_dataset, split_dataset, LOG_PROPS

OKABE = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442', '#000000']
_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


# ---------------------------------------------------------------- 1. 데이터 (train.run 과 같은 분할)
def split_smiles(prop='logP', max_n=3000, how='random', seed=42):
    """train.run 과 같은 load_data -> gen_dataset -> split_dataset 를 거쳐 분자 목록만 돌려준다.
    그래서 여기의 test 분자 = BGNN 의 test 분자 (순서까지 같다)."""
    df = load_data(prop, max_n=max_n, seed=seed)
    sets = split_dataset(gen_dataset(df, prop), how=how, seed=seed)
    return {k: pd.DataFrame({'smiles': [g.smiles for g in v], 'y': [float(g.y) for g in v]}) for k, v in sets.items()}


def morgan(smiles):
    """Morgan 지문 (반경 2, 2048 비트), 0/1 float 배열. 1주차와 같은 설정."""
    out = np.zeros((len(smiles), 2048), dtype=np.float64)
    for i, s in enumerate(smiles):
        DataStructs.ConvertToNumpyArray(_GEN.GetFingerprint(Chem.MolFromSmiles(s)), out[i])
    return out


# ---------------------------------------------------------------- 2. σ 를 내는 방법
def rf_sigma(Xtr, ytr, Xte, n_trees=300, seed=42):
    """앙상블: 나무마다 다른 부트스트랩 표본으로 학습 -> 나무 예측의 표준편차 = σ."""
    rf = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    P = np.stack([t.predict(Xte) for t in rf.estimators_])
    return P.mean(0), P.std(0)


class Tanimoto(Kernel):
    """k(a, b) = |a ∩ b| / |a ∪ b|. 두 분자의 Morgan 비트가 얼마나 겹치나. 조절할 값이 없는 커널."""

    def __init__(self):   # sklearn 규칙: 조절값이 없어도 __init__ 을 명시한다
        pass

    def __call__(self, X, Y=None, eval_gradient=False):
        Y = X if Y is None else Y
        xy = X @ Y.T
        K = xy / (X.sum(1)[:, None] + Y.sum(1)[None, :] - xy + 1e-9)
        return (K, np.empty((X.shape[0], X.shape[0], 0))) if eval_gradient else K

    def diag(self, X):
        return np.ones(X.shape[0])

    def is_stationary(self):
        return False


def gp_model(kernel=None):
    """σ_f² · Tanimoto + 잡음. 잡음(WhiteKernel) = 같은 분자를 다시 재도 남는 흩어짐(우연 불확실성)의 추정."""
    kernel = kernel or ConstantKernel(1.0, (1e-2, 1e2)) * Tanimoto() + WhiteKernel(0.1, (1e-4, 10))
    return GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=0)


def gp_sigma(Xtr, ytr, Xte, return_model=False):
    gp = gp_model().fit(Xtr, ytr)
    mu, sd = gp.predict(Xte, return_std=True)      # sd 는 잡음 항까지 포함한 예측 표준편차
    return (mu, sd, gp) if return_model else (mu, sd)


def gp_noise_sd(gp):
    """학습된 잡음 표준편차 (원래 y 단위). normalize_y=True 라서 y 표준편차를 곱한다."""
    return float(np.sqrt(gp.kernel_.k2.noise_level) * gp._y_train_std)


# ---------------------------------------------------------------- 3. 보정 (σ 가 맞는가)
def calibration(y, mu, sd, n_bins=8):
    """2σ 적중률 · Spearman(|오차|, σ) · σ 구간별 RMSE.

    정규분포면 |오차| <= 2σ 인 비율이 0.954. 더 낮으면 σ 를 작게(자신 있게) 잡은 것.
    bins: σ 크기 순으로 같은 개수씩 나눈 구간마다 (평균 σ, 실제 RMSE). 대각선 위에 있으면 σ 가 맞다."""
    err = np.abs(y - mu)
    q = np.quantile(sd, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(q, sd, side='right') - 1, 0, n_bins - 1)
    bins = pd.DataFrame({'sd_mean': [sd[b == k].mean() for k in range(n_bins)],
                         'rmse': [np.sqrt(np.mean(err[b == k] ** 2)) for k in range(n_bins)],
                         'n': [int((b == k).sum()) for k in range(n_bins)]})
    return dict(MAE=float(err.mean()), mean_sd=float(sd.mean()),
                coverage_2sd=float(np.mean(err <= 2 * sd)),
                spearman=float(spearmanr(err, sd).correlation)), bins


# ---------------------------------------------------------------- 4. 적용범위 (학습 세트와 얼마나 닮았나)
def max_similarity(train_smiles, query_smiles):
    """query 분자마다 학습 세트에서 가장 닮은 분자와의 Tanimoto. 1 = 같은 비트, 0 = 겹치는 비트 없음."""
    fp_tr = [_GEN.GetFingerprint(Chem.MolFromSmiles(s)) for s in train_smiles]
    return np.array([max(DataStructs.BulkTanimotoSimilarity(_GEN.GetFingerprint(Chem.MolFromSmiles(s)), fp_tr))
                     for s in query_smiles])


def error_by_similarity(sim, err, edges=(0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0001)):
    """닮은 정도 구간별 평균 |오차| 와 분자 수."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (sim >= lo) & (sim < hi)
        rows.append(dict(lo=lo, hi=min(hi, 1.0), n=int(m.sum()), mae=float(err[m].mean()) if m.any() else np.nan))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 5. 결정: 능동학습 시뮬레이션
def expected_improvement(mu, sd, best, xi=0.0):
    """최댓값 찾기. 지금까지 잰 최고값 best 를 얼마나 넘을지의 기댓값."""
    sd = np.maximum(sd, 1e-9)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)


def ucb(mu, sd, kappa=2.576):
    """상한 신뢰. 연구실 cjh_bayes_optimization.py 가 쓴 kappa = 2.576 (bayes_opt 기본값)."""
    return mu + kappa * sd


def active_learning(X_pool, y_pool, X_test, y_test, strategy='random', start=0.05, step=0.05, rounds=10,
                    seed=0, top_frac=0.01):
    """풀(pool)의 y 를 '아직 안 잰 실험값'으로 숨겨 두고, 한 라운드마다 step 만큼 골라 '잰다'.

    strategy: 'random' (무작위), 'sigma' (σ 가 가장 큰 분자 = 모르는 곳 메우기),
              'ei' (최댓값 찾기, 기대 개선), 'ucb' (최댓값 찾기, μ + 2.576σ)
    기록: 잰 개수, test MAE (모델이 얼마나 정확해졌나), 지금까지 찾은 최고 y, 풀 상위 1% 중 찾은 비율.
    한 라운드에 여러 개를 고를 때 점수 상위 k 개를 한꺼번에 고른다 (단순화: 서로 닮은 분자를 함께 고를 수 있다)."""
    rng = np.random.default_rng(seed)
    N = len(y_pool); k0, k = int(round(start * N)), int(round(step * N))
    top = set(np.argsort(y_pool)[-max(1, int(round(top_frac * N))):])
    labeled = list(rng.choice(N, k0, replace=False))
    kernel, log, t0 = None, [], time.time()
    for r in range(rounds + 1):
        gp = gp_model(kernel).fit(X_pool[labeled], y_pool[labeled])
        kernel = gp.kernel_                                      # 다음 라운드는 이 값에서 출발 (속도)
        mu_t = gp.predict(X_test)
        log.append(dict(strategy=strategy, seed=seed, round=r, n_labeled=len(labeled),
                        test_MAE=float(np.mean(np.abs(mu_t - y_test))), best_found=float(y_pool[labeled].max()),
                        top_found=len(top & set(labeled)) / len(top)))
        if r == rounds:
            break
        rest = np.setdiff1d(np.arange(N), labeled)
        if strategy == 'random':
            pick = rng.choice(rest, k, replace=False)
        else:
            mu, sd = gp.predict(X_pool[rest], return_std=True)
            score = {'sigma': sd, 'ucb': ucb(mu, sd),
                     'ei': expected_improvement(mu, sd, y_pool[labeled].max())}[strategy]
            pick = rest[np.argsort(score)[-k:]]
        labeled += list(pick)
    out = pd.DataFrame(log); out['sec'] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------- 6. 그림 점검 (하우스 스타일)
def check_fig(fig, data_ranges=()):
    """네 방향 테두리, 축이 데이터를 자르지 않는지 확인. data_ranges = [(ax, xs, ys), ...]."""
    for ax in fig.axes:
        if ax.axison:
            assert all(s.get_visible() for s in ax.spines.values()), 'spine missing'
            assert not ax.texts or all(t.get_text() in 'abcdef' for t in ax.texts), 'annotation text inside plot'
    for ax, xs, ys in data_ranges:
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        assert np.nanmin(xs) >= x0 and np.nanmax(xs) <= x1 and np.nanmin(ys) >= y0 and np.nanmax(ys) <= y1, 'axis clips data'


def panel(ax, s):
    ax.text(-0.02, 1.03, s, transform=ax.transAxes, fontsize=15, fontweight='bold', ha='right', va='bottom')
