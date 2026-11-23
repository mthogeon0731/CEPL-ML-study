"""enslib.py : 앙상블 트랙(2-C, ENS/) 공통 도구.
작업 공간 루트에서 mp_mae_push/ (mpcommon, splits, cache, phase2/prereg) 와 ENS/inputs/ 를 읽는다.
test 행은 어떤 형태로도 읽지 않는다 (라벨·예측·특징 통계 모두). 지문·유사도도 train·validation 행만 계산한다.
"""
import os, sys, json, time
sys.dont_write_bytecode = True
WS = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(WS, 'mp_mae_push', 'common'))
os.environ.setdefault('MP_PUSH_ROOT', os.path.join(WS, 'mp_mae_push'))
import mpcommon as mc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ENS = os.path.join(WS, 'ENS')
LEDGER = os.path.join(ENS, 'ledger.csv')
TRACK = 'ENS'
SPLITS = ('random', 'scaffold')
MEMBERS_FILE = os.environ.get('ENS_MEMBERS', 'members.csv')  # 멤버 등록부 (T3 도착 후 members_T3.csv)
_c = {}


def split(how):
    if ('split', how) not in _c:
        _c[('split', how)] = mc.load_split(how)
    return _c[('split', how)]


def y_all():
    """라벨: splits 파일과 같은 행 순서의 mp.csv value (K). train·validation 행만 쓴다."""
    if 'y' not in _c:
        _c['y'] = pd.read_csv(os.path.join(mc.WEEK2, 'data', 'mp.csv'))['value'].to_numpy(float)
    return _c['y']


def dedup_mask(how):
    idx, _ = split(how)
    d = pd.read_csv(os.path.join(WS, 'mp_mae_push', 'phase2', 'prereg', f'dedup_rows_{how}.csv'))
    d = d[d['set'] == 'validate'].set_index('row')
    return d.loc[idx['validate'], 'overlap'].to_numpy() == 0


def murcko_groups():
    """t1blib.murcko_groups 와 같은 규칙 (Murcko, 입체 무시, 고리 없으면 ACYCLIC_+SMILES)."""
    if 'murcko' not in _c:
        path = os.path.join(ENS, 'cache', 'murcko.csv')
        if os.path.exists(path):
            g = pd.read_csv(path, keep_default_na=False)
        else:
            from rdkit import Chem
            from rdkit.Chem.Scaffolds import MurckoScaffold
            _, s = split('random')
            s = s.sort_values('row')
            out = []
            for smi in s['smiles']:
                m = Chem.MolFromSmiles(smi)
                try:
                    sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) if m is not None else ''
                except Exception:
                    sc = ''
                out.append(sc if sc else 'ACYCLIC_' + str(smi))
            g = pd.DataFrame({'row': s['row'].to_numpy(), 'scaffold': out})
            os.makedirs(os.path.dirname(path), exist_ok=True)
            g.to_csv(path, index=False)
        codes = pd.factorize(g['scaffold'])[0]
        arr = np.empty(g['row'].max() + 1, dtype=np.int64)
        arr[g['row'].to_numpy()] = codes
        _c['murcko'] = arr
    return _c['murcko']


def train_folds(how, k=5):
    """T1·T1b 와 같은 규칙: random KFold(5, shuffle, 42), scaffold GroupKFold(5, Murcko). 위치 인덱스 -> fold 번호 배열."""
    from sklearn.model_selection import KFold, GroupKFold
    tr = split(how)[0]['train']
    sp = (KFold(k, shuffle=True, random_state=42).split(tr) if how == 'random'
          else GroupKFold(k).split(tr, groups=murcko_groups()[tr]))
    f = np.full(len(tr), -1)
    for i, (_, b) in enumerate(sp):
        f[b] = i
    return f


def ledger(row):
    row = dict(row)
    row.setdefault('track', TRACK)
    if isinstance(row.get('params'), dict):
        row['params'] = json.dumps(row['params'], ensure_ascii=False)
    mc.append_ledger(LEDGER, row)


# ---------------- 지표 ----------------
def mae(y, p):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(p, float))))


def pboot(y, p_new, p_ref, n_boot=2000, seed=0):
    return mc.paired_bootstrap(y, p_new, p_ref, n_boot=n_boot, seed=seed)


def fmt_ci(t):
    d, lo, hi = t
    return f'{d:+.2f} [{lo:+.2f}, {hi:+.2f}]'


# ---------------- 후보 예측 ----------------
def load_members(how, kind, members):
    """members: {name: path}. kind 'val' 또는 'OOF'. 분할 파일의 validation/train 순서(pos)로 정렬해 돌려준다.
    row 집합·y 일치를 검사한다 (불일치면 오류)."""
    idx, _ = split(how)
    ref = idx['validate' if kind == 'val' else 'train']
    y = y_all()[ref]
    out = {}
    for m, p in members.items():
        d = pd.read_csv(p)
        assert {'row', 'y', 'pred'} <= set(d.columns), (m, d.columns)
        assert set(d['row']) == set(ref) and len(d) == len(ref), (m, how, kind, 'row 집합 불일치')
        d = d.set_index('row').loc[ref]
        assert np.abs(d['y'].to_numpy(float) - y).max() < 1e-6, (m, 'y 불일치')
        assert not d['pred'].isna().any(), (m, 'NaN')
        out[m] = d['pred'].to_numpy(float)
    return out, y
