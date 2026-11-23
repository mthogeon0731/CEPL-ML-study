"""p3lib.py : 3단계 최종 평가(P3) 공통 도구.

- 작업 공간 루트에서 실행한다 (mp_mae_push/ 와 P3/ 가 루트 바로 아래).
- test 라벨 차단: y_masked(how) 는 분할 how 의 test 행을 NaN 으로 가린 라벨이다.
  학습·예측 스크립트는 y_masked(how) 만 쓴다. 진짜 test 라벨은 평가 스크립트(s9_evaluate.py)만 읽는다.
- test 예측 파일은 P3/preds/test_blind/<member>_<split>_test.csv (row, pred, pred_s*) — y 열 없음.
"""
import hashlib
import json
import os
import sys

sys.dont_write_bytecode = True
WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.join(WS, 'mp_mae_push', 'common'))
os.environ.setdefault('MP_PUSH_ROOT', os.path.join(WS, 'mp_mae_push'))
import mpcommon as mc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

P3 = os.path.join(WS, 'P3')
LEDGER = os.path.join(P3, 'ledger.csv')
VAL_DIR = os.path.join(P3, 'preds', 'val')
TEST_DIR = os.path.join(P3, 'preds', 'test_blind')
TAB = os.path.join(P3, 'tables')
INP = os.path.join(P3, 'inputs')
SPLITS = ('random', 'scaffold')
TRACK = 'P3'
BUDGET_CORE_H = 18.0
XGB_NJ, LGB_NT, GIN_NT = 8, 6, 1
_c = {}


def split(how):
    if ('split', how) not in _c:
        _c[('split', how)] = mc.load_split(how)
    return _c[('split', how)]


def _y_raw():
    if 'y' not in _c:
        _c['y'] = pd.read_csv(os.path.join(mc.WEEK2, 'data', 'mp.csv'))['value'].to_numpy(float)
    return _c['y']


def y_masked(how):
    """분할 how 의 test 행을 NaN 으로 가린 라벨 (19,945). 학습·예측 단계는 이것만 쓴다."""
    y = _y_raw().copy()
    y[split(how)[0]['test']] = np.nan
    return y


def dedup_keep(how, which):
    """which 세트 순서의 bool: True = 중복 제거 세트에 포함 (InChIKey 첫 블록이 다른 세트와 겹치지 않음)."""
    idx, _ = split(how)
    d = pd.read_csv(os.path.join(INP, f'dedup_{how}.csv'))
    d = d[d['set'] == which].set_index('row')
    return d.loc[idx[which], 'overlap'].to_numpy() == 0


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def ledger(row):
    row = dict(row)
    row.setdefault('track', TRACK)
    if isinstance(row.get('params'), dict):
        row['params'] = json.dumps(row['params'], ensure_ascii=False, sort_keys=True)
    mc.append_ledger(LEDGER, row)


def used_core_h():
    if not os.path.exists(LEDGER):
        return 0.0
    return float(pd.read_csv(LEDGER, encoding='utf-8-sig')['cpu_core_h'].fillna(0).sum())


def save_val(member, how, rows, preds_by_seed, seeds):
    """validation 예측 저장: row, y, pred(seed 평균), pred_s<seed>. (validation 라벨은 공개 대상)"""
    os.makedirs(VAL_DIR, exist_ok=True)
    y = y_masked(how)
    P = np.asarray(preds_by_seed, float)
    d = pd.DataFrame({'row': rows, 'y': y[rows], 'pred': P.mean(0)})
    for s, p in zip(seeds, P):
        d[f'pred_s{s}'] = p
    path = os.path.join(VAL_DIR, f'{member}_{how}_val.csv')
    d.to_csv(path, index=False)
    return path


def save_test_blind(member, how, rows, preds_by_seed, seeds):
    """test 예측 저장 (y 없음): row, pred(seed 평균), pred_s<seed>."""
    os.makedirs(TEST_DIR, exist_ok=True)
    assert set(rows) == set(split(how)[0]['test'])
    P = np.asarray(preds_by_seed, float)
    assert np.isfinite(P).all(), (member, how, 'NaN in test pred')
    d = pd.DataFrame({'row': rows, 'pred': P.mean(0)})
    for s, p in zip(seeds, P):
        d[f'pred_s{s}'] = p
    path = os.path.join(TEST_DIR, f'{member}_{how}_test.csv')
    d.to_csv(path, index=False)
    return path
