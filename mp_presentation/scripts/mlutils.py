"""mlutils.py — 발표 보조 스크립트 공용 함수 [추가: 연구실 코드가 아님]

- 열 이름 자동 인식 (smiles / y)
- canonical SMILES, Murcko 골격 (12, 14쪽)
- random / scaffold 분할 (14쪽) — scaffold는 DeepChem 방식(큰 골격부터 train)
- RDKit 기술자 행렬 (16쪽: Descriptors.CalcMolDescriptors)
- MAE / RMSE (23쪽)

연구실 datamanager.py 의 분할·표준화와는 별개이며, 결과표(과제 1·2단계)에는 쓰지 않는다.
후분석·SHAP(발표 5절)용.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")  # 무효 SMILES 경고 숨김 (개수는 따로 센다)

SMILES_CANDIDATES = ["smiles", "SMILES", "Smiles", "canonical_smiles", "smi", "mol", "molecule"]
Y_CANDIDATES = ["mp", "MP", "melting_point", "mpK", "mp_K", "y", "target", "value", "label", "mpc", "Tm", "tm"]


# ---------------------------------------------------------------- 열 찾기
def find_col(df: pd.DataFrame, given: str | None, candidates: Sequence[str], kind: str) -> str:
    """given이 있으면 그대로, 없으면 후보 이름에서 찾고, 그래도 없으면 kind별 규칙으로 추정."""
    if given:
        if given not in df.columns:
            raise SystemExit(f"[error] column '{given}' not in CSV. columns = {list(df.columns)}")
        return given
    for c in candidates:
        if c in df.columns:
            return c
    if kind == "smiles":  # 문자열 열 중 '(' 'C' 등이 많은 열
        for c in df.columns:
            if df[c].dtype == object:
                s = df[c].dropna().astype(str).head(50)
                if len(s) and (s.str.contains(r"[A-Za-z]").mean() > 0.9) and (s.str.len().mean() > 3):
                    return c
    if kind == "y":  # 첫 번째 숫자 열
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
    raise SystemExit(f"[error] cannot find {kind} column. columns = {list(df.columns)}  → use --{kind}-col")


def read_dataset(path: str, smiles_col: str | None = None, y_col: str | None = None) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(path)
    sc = find_col(df, smiles_col, SMILES_CANDIDATES, "smiles")
    yc = find_col(df, y_col, Y_CANDIDATES, "y")
    return df, sc, yc


# ---------------------------------------------------------------- SMILES 처리 (12, 14쪽)
def canonicalize(smi: str) -> str | None:
    """MolToSmiles(MolFromSmiles(s)) — 무효면 None (12쪽)."""
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol is not None else None


def murcko_scaffold(smi: str) -> str:
    """Murcko 골격 SMILES (14쪽). 비고리 분자 → '' (빈 골격 하나로 묶임)."""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
    except Exception:  # noqa: BLE001
        return ""


def has_salt(smi: str) -> bool:
    return isinstance(smi, str) and "." in smi


def has_stereo(smi: str) -> bool:
    return isinstance(smi, str) and any(ch in smi for ch in ("@", "/", "\\"))


# ---------------------------------------------------------------- 분할 (14쪽)
def random_split(n: int, frac_train=0.8, frac_valid=0.1, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(round(frac_train * n))
    n_va = int(round(frac_valid * n))
    return idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:]


def scaffold_split(smiles: Iterable[str], frac_train=0.8, frac_valid=0.1, seed=42):
    """DeepChem 방식: 골격별로 묶고 큰 골격 집합부터 train에 채움 → test는 작은(드문) 골격.
    seed는 같은 크기 골격 사이의 순서에만 영향."""
    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[murcko_scaffold(s)].append(i)
    rng = np.random.default_rng(seed)
    sets = list(groups.values())
    rng.shuffle(sets)
    sets.sort(key=len, reverse=True)
    n = sum(len(s) for s in sets)
    n_tr, n_va = frac_train * n, (frac_train + frac_valid) * n
    tr, va, te = [], [], []
    for s in sets:
        if len(tr) + len(s) <= n_tr:
            tr += s
        elif len(tr) + len(va) + len(s) <= n_va:
            va += s
        else:
            te += s
    return np.array(tr), np.array(va), np.array(te)


# ---------------------------------------------------------------- 기술자 (16쪽)
def descriptor_matrix(smiles: Sequence[str]) -> pd.DataFrame:
    """RDKit Descriptors.CalcMolDescriptors → (N, ~217) DataFrame. NaN/inf 열은 호출 쪽에서 정리."""
    rows = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        rows.append(Descriptors.CalcMolDescriptors(mol) if mol is not None else {})
    X = pd.DataFrame(rows).astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    return X


def clean_descriptors(X: pd.DataFrame) -> pd.DataFrame:
    """NaN이 있는 열 제거, 상수 열 제거."""
    X = X.dropna(axis=1, how="any")
    nunique = X.nunique()
    return X.loc[:, nunique > 1]


# ---------------------------------------------------------------- 표 출력 (tabulate 없이)
def df_to_md(df: pd.DataFrame, index: bool = True, floatfmt: str = ".2f") -> str:
    """DataFrame → 마크다운 표 (pandas.to_markdown은 tabulate가 필요해서 직접 구현)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]

    def fmt(v):
        if isinstance(v, float):
            return "nan" if math.isnan(v) else format(v, floatfmt)
        return str(v).replace("|", "\\|")

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------- 지표 (23쪽)
def mae(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(np.mean(np.abs(yhat - y)))


def rmse(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(math.sqrt(np.mean((yhat - y) ** 2)))


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def mol_weights(smiles: Sequence[str]) -> np.ndarray:
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        out.append(Descriptors.MolWt(mol) if mol is not None else np.nan)
    return np.array(out, dtype=float)


def heavy_atoms(smiles: Sequence[str]) -> np.ndarray:
    out = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        out.append(mol.GetNumHeavyAtoms() if mol is not None else np.nan)
    return np.array(out, dtype=float)
