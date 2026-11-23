"""mpcommon.py : mp_mae_push 공통 도구 — 데이터·분할·특징·지표·비용 측정·ledger·짝지은 부트스트랩.

원본 수업 코드(261109_Week2_Practice/train_xgb.py)의 load/split/featurize/metrics 를 그대로 import 해서 쓴다.
원본 폴더에는 아무것도 쓰지 않는다 (PYTHONDONTWRITEBYTECODE=1 로 실행).
"""
import csv
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil

WEEK2 = os.environ.get('MP_WEEK2_DIR', r'C:\Users\mthog\CEPL ML\261109_Week2_Practice')
ROOT = os.environ.get('MP_PUSH_ROOT', os.path.join(os.getcwd(), 'mp_mae_push'))
SPLIT_DIR = os.environ.get('MP_SPLIT_DIR', os.path.join(ROOT, 'splits'))
CACHE_DIR = os.path.join(ROOT, 'cache')
SEED = 42

# 기준선 설정 (train_xgb.run 과 동일)
BASE_XGB = dict(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.5)

# 에너지 추정 가정: energy_kwh = (cpu_core_s / 논리 코어 수) x CPU 패키지 전력[W] / 3.6e6
#   = 평균 CPU 사용률 x TDP x 시간.  TDP 값은 cost_assumptions.md 에 근거와 함께 기록.
CPU_TDP_W = float(os.environ.get('MP_CPU_TDP_W', '25'))
N_LOGICAL = psutil.cpu_count(logical=True) or 1

LEDGER_COLS = ['run_id', 'track', 'model', 'features', 'data_variant', 'objective', 'split', 'seed', 'params',
               'val_MAE', 'val_RMSE', 'test_MAE', 'test_RMSE', 'test_R2', 'n_train',
               'wall_s', 'cpu_core_h', 'gpu_h', 'peak_ram_gb', 'energy_kwh', 'api_tokens', 'api_cost',
               'infer_s_per_1k', 'stage', 'notes', 'energy_method']


def _train_xgb():
    if WEEK2 not in sys.path:
        sys.path.insert(0, WEEK2)
    import train_xgb  # noqa: PLC0415
    return train_xgb


def load_data():
    """mp.csv 전체 (train_xgb.load 와 같은 거르기). -> df, mols, y[K]"""
    return _train_xgb().load('mp', None, SEED)


def make_split(smiles, how):
    """train_xgb.split 그대로. -> train, validate, test 인덱스"""
    return _train_xgb().split(list(smiles), how, SEED)


def load_split(how):
    """고정 분할 파일(splits/<how>.csv)에서 인덱스를 읽는다. 모든 트랙은 이것만 쓴다."""
    s = pd.read_csv(os.path.join(SPLIT_DIR, f'{how}.csv'))
    # pos = train_xgb.split 이 돌려준 세트 안 순서. 이 순서로 읽어야 subsample 이 있는 모델이 원본과 같은 수치를 낸다.
    return {k: s.loc[s['set'] == k].sort_values('pos')['row'].to_numpy() for k in ('train', 'validate', 'test')}, s


def desc217(mols, cache=True):
    """RDKit 기술자 217개 (train_xgb.featurize 'desc'). 작업 폴더 cache 에 저장해 재사용."""
    path = os.path.join(CACHE_DIR, 'X_desc217.npy')
    if cache and os.path.exists(path):
        X = np.load(path)
        if len(X) == len(mols):
            return X
    X = _train_xgb().featurize(mols, 'desc')
    if cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, X)
    return X


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(MAE=float(np.mean(np.abs(y - p))), RMSE=float(np.sqrt(np.mean((y - p) ** 2))),
                R2=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)))


def paired_bootstrap(y, p_new, p_ref, n_boot=2000, seed=0):
    """같은 test 에서 MAE(new) - MAE(ref) 의 짝지은 부트스트랩. -> (차이, 95% CI 하한, 상한)"""
    y = np.asarray(y, float)
    d = np.abs(y - np.asarray(p_new, float)) - np.abs(y - np.asarray(p_ref, float))
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def boot_ci(x, n_boot=2000, seed=0, stat=np.mean):
    """표본 통계량의 부트스트랩 95% CI."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    v = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n_boot)])
    lo, hi = np.percentile(v, [2.5, 97.5])
    return float(lo), float(hi)


class CostMeter:
    """with CostMeter('최종') as c: ...  -> c.wall_s, c.cpu_core_h, c.peak_ram_gb, c.energy_kwh

    cpu_core_h : 이 프로세스 + 자식 프로세스의 user+sys CPU 시간 합 / 3600 (psutil, 자식은 0.2 s 간격 표본)
    peak_ram_gb: 같은 간격으로 본 RSS 합의 최댓값
    energy_kwh : 평균 CPU 사용률 x TDP 추정 (CodeCarbon 미사용)
    """

    def __init__(self, stage='', interval=0.2):
        self.stage, self.interval = stage, interval

    def __enter__(self):
        self.proc = psutil.Process()
        self.t_wall0, self.t0 = time.time(), time.perf_counter()
        ct = self.proc.cpu_times()
        self.cpu0 = ct.user + ct.system
        self._kids = {}
        self.peak_rss = self.proc.memory_info().rss
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        return self

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._poll()

    def _poll(self):
        rss = 0
        try:
            rss = self.proc.memory_info().rss
            kids = self.proc.children(recursive=True)
        except psutil.Error:
            kids = []
        for k in kids:
            try:
                ct = k.cpu_times()
                key, cpu = (k.pid, k.create_time()), ct.user + ct.system
                first = 0.0 if k.create_time() >= self.t_wall0 - 0.5 else cpu
                self._kids.setdefault(key, [first, cpu])[1] = cpu
                rss += k.memory_info().rss
            except psutil.Error:
                pass
        self.peak_rss = max(self.peak_rss, rss)

    def __exit__(self, *exc):
        self._stop.set()
        self._th.join()
        self._poll()
        self.wall_s = time.perf_counter() - self.t0
        ct = self.proc.cpu_times()
        cpu_s = (ct.user + ct.system - self.cpu0) + sum(last - first for first, last in self._kids.values())
        self.cpu_core_h = cpu_s / 3600.0
        self.peak_ram_gb = self.peak_rss / 2 ** 30
        self.energy_kwh = (cpu_s / N_LOGICAL) * CPU_TDP_W / 3.6e6
        self.energy_method = f'cpu_util x TDP({CPU_TDP_W:g} W, {N_LOGICAL} threads)'
        return False

    def as_row(self):
        return dict(wall_s=round(self.wall_s, 3), cpu_core_h=round(self.cpu_core_h, 6), gpu_h=0,
                    peak_ram_gb=round(self.peak_ram_gb, 3), energy_kwh=round(self.energy_kwh, 8),
                    energy_method=self.energy_method, stage=self.stage)


def append_ledger(path, row):
    """ledger.csv 에 실행 하나 = 한 줄. 정해진 열 밖의 키는 오류."""
    extra = set(row) - set(LEDGER_COLS)
    if extra:
        raise KeyError(f'ledger 에 없는 열: {sorted(extra)}')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in LEDGER_COLS})
