"""error_analysis.py — 후분석: 어디서 틀리나 (발표 5절) [추가: 연구실 코드가 아님]

입력: test 예측 CSV (열: smiles, y_true, y_pred — 이름이 다르면 --smiles-col/--true-col/--pred-col)
      선택: 'split' 또는 'source' 열이 있으면 그 그룹별 MAE도 계산

사용:
    python error_analysis.py pred_test_xgb_scaffold.csv --out ../05_analysis_xai --top 20 --train-mean 340.2

출력 (--out):
    summary.md             MAE, RMSE, R², 기준선(--train-mean 있을 때), 큰 오차 비율
    worst_molecules.csv/.png   |오차| 상위 k개 (표 + 구조 격자 그림)
    error_by_range.csv     목표값 구간별 MAE·개수 (끝 영역 = 데이터 적은 곳인지)
    error_by_scaffold.csv  Murcko 골격별 MAE·개수 (14쪽)
    error_by_molwt.csv, fig_error_by_molwt.png   분자량 구간별 MAE
    fig_residuals.png      잔차 vs 실제값 + 잔차 히스토그램
    error_by_<group>.csv   split/source 열이 있을 때 (13쪽 출처별 오차)

슬라이드 근거: 13쪽(출처별 오차), 14쪽(골격), 20쪽(학습 범위 밖), 23쪽(MAE·RMSE·기준선·parity), 24쪽(끝 영역), 49쪽(틀린 분자 보기)
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlutils import (SMILES_CANDIDATES, canonicalize, df_to_md, find_col, mae,  # noqa: E402
                     mol_weights, murcko_scaffold, r2, rmse)

TRUE_CANDIDATES = ["y_true", "true", "actual", "y", "target", "mp", "label"]
PRED_CANDIDATES = ["y_pred", "pred", "prediction", "yhat", "y_hat", "predicted"]


def group_table(d: pd.DataFrame, key: str, min_n: int = 1) -> pd.DataFrame:
    t = d.groupby(key).agg(n=("abs_err", "size"), MAE=("abs_err", "mean"), RMSE=("err", lambda e: float(np.sqrt(np.mean(e ** 2)))),
                           mean_err=("err", "mean"))
    t = t[t["n"] >= min_n].sort_values(["n", "MAE"], ascending=[False, False])
    return t.round(2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--smiles-col")
    ap.add_argument("--true-col")
    ap.add_argument("--pred-col")
    ap.add_argument("--name", default="mp")
    ap.add_argument("--unit", default="K")
    ap.add_argument("--out", default=".")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--train-mean", type=float, default=None,
                    help="train 목표값 평균 → 기준선 MAE (23쪽). 없으면 CSV의 train_mean 열, 그것도 없으면 test 평균으로 근사(낙관적)")
    ap.add_argument("--bin-width", type=float, default=50.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.csv)
    sc = find_col(df, a.smiles_col, SMILES_CANDIDATES, "smiles")
    tc = find_col(df, a.true_col, TRUE_CANDIDATES, "y")
    pc = find_col(df, a.pred_col, PRED_CANDIDATES, "y")
    if tc == pc:
        raise SystemExit("[error] y_true / y_pred 열을 구분하지 못했습니다 → --true-col, --pred-col 지정")

    d = pd.DataFrame({"smiles": df[sc], "y_true": pd.to_numeric(df[tc], errors="coerce"),
                      "y_pred": pd.to_numeric(df[pc], errors="coerce")})
    for extra in ("split", "source", "model"):
        if extra in df.columns:
            d[extra] = df[extra]
    d = d.dropna(subset=["y_true", "y_pred"]).reset_index(drop=True)
    d["err"] = d["y_pred"] - d["y_true"]          # 잔차 부호: + = 과대 예측
    d["abs_err"] = d["err"].abs()
    d["canon"] = d["smiles"].map(canonicalize)
    d["scaffold"] = d["canon"].fillna("").map(murcko_scaffold).replace("", "(acyclic)")
    d["molwt"] = mol_weights(d["canon"].fillna(""))

    # ---- 전체 지표 (23쪽)
    y, yhat = d["y_true"].to_numpy(), d["y_pred"].to_numpy()
    m, r, rr = mae(y, yhat), rmse(y, yhat), r2(y, yhat)
    if a.train_mean is not None:
        base_mean, base_src = a.train_mean, "train 평균 (--train-mean)"
    elif "train_mean" in df.columns and pd.notna(df["train_mean"].iloc[0]):
        base_mean, base_src = float(df["train_mean"].iloc[0]), "train 평균 (CSV train_mean 열)"
    else:
        base_mean, base_src = float(y.mean()), "⚠ test 평균으로 근사 — --train-mean 권장"
    base_mae = mae(y, np.full_like(y, base_mean))
    big = float((d["abs_err"] > 2 * m).mean())

    # ---- 구간별 (24쪽 끝 영역)
    lo = np.floor(y.min() / a.bin_width) * a.bin_width
    hi = np.ceil(y.max() / a.bin_width) * a.bin_width
    edges = np.arange(lo, hi + a.bin_width, a.bin_width)
    d["range"] = pd.cut(d["y_true"], bins=edges, include_lowest=True)
    by_range = group_table(d, "range")
    by_range.index = by_range.index.astype(str)
    by_range.to_csv(os.path.join(a.out, "error_by_range.csv"))

    # ---- 골격별 (14쪽)
    by_scaf = group_table(d, "scaffold", min_n=1)
    by_scaf.to_csv(os.path.join(a.out, "error_by_scaffold.csv"))
    n_scaf = d["scaffold"].nunique()
    singleton = int((d["scaffold"].value_counts() == 1).sum())

    # ---- 분자량 구간별 (20쪽 학습 범위 밖)
    mw_edges = [0, 100, 150, 200, 250, 300, 400, 500, 10_000]
    d["mw_bin"] = pd.cut(d["molwt"], bins=mw_edges)
    by_mw = group_table(d, "mw_bin")
    by_mw.index = by_mw.index.astype(str)
    by_mw.to_csv(os.path.join(a.out, "error_by_molwt.csv"))

    extra_tables = {}
    for extra in ("split", "source", "model"):
        if extra in d.columns:
            t = group_table(d, extra)
            t.to_csv(os.path.join(a.out, f"error_by_{extra}.csv"))
            extra_tables[extra] = t

    # ---- 틀린 분자 (49쪽)
    worst = d.sort_values("abs_err", ascending=False).head(a.top)
    worst[["smiles", "y_true", "y_pred", "err", "abs_err", "molwt", "scaffold"]].round(2).to_csv(
        os.path.join(a.out, "worst_molecules.csv"), index=False)
    mols, legends = [], []
    for _, row in worst.head(12).iterrows():
        mol = Chem.MolFromSmiles(row["smiles"]) if isinstance(row["smiles"], str) else None
        if mol is not None:
            mols.append(mol)
            legends.append(f"true {row['y_true']:.0f} / pred {row['y_pred']:.0f} ({row['err']:+.0f} {a.unit})")
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=4, subImgSize=(260, 220), legends=legends, returnPNG=False)
        img.save(os.path.join(a.out, "worst_molecules.png"))

    # ---- 그림
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].scatter(y, d["err"], s=5, alpha=0.3, color="#4C72B0")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_xlabel(f"true {a.name} [{a.unit}]")
    axes[0].set_ylabel(f"residual (pred - true) [{a.unit}]")
    axes[0].set_title("residual vs true")
    axes[1].hist(d["err"], bins=60, color="#4C72B0", edgecolor="white")
    axes[1].set_xlabel(f"residual [{a.unit}]")
    axes[1].set_title(f"MAE {m:.1f}  RMSE {r:.1f}")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_residuals.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(range(len(by_mw)), by_mw["MAE"], color="#4C72B0")
    ax.set_xticks(range(len(by_mw)))
    ax.set_xticklabels([f"{i}\n(n={n})" for i, n in zip(by_mw.index, by_mw["n"])], fontsize=7)
    ax.set_ylabel(f"MAE [{a.unit}]")
    ax.set_xlabel("MolWt bin")
    ax.set_title("error by molecular weight")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "fig_error_by_molwt.png"), dpi=150)
    plt.close(fig)

    # ---- 요약
    L = [f"# 후분석 요약 — `{os.path.basename(a.csv)}`\n",
         "| 항목 | 값 | 근거 |", "|---|---|---|",
         f"| N (test) | {len(d):,} | |",
         f"| MAE [{a.unit}] | {m:.2f} | 23쪽 |",
         f"| RMSE [{a.unit}] | {r:.2f} (RMSE−MAE = {r-m:.2f} → 차이 큼 = 큰 오차 몇 개) | 23쪽 |",
         f"| R² | {rr:.3f} | [추가] |",
         f"| 기준선 MAE (평균 {base_mean:.1f}로 예측) | {base_mae:.2f} → 모델/기준선 = {m/base_mae:.2f} | 23쪽, {base_src} |",
         f"| \\|오차\\| > 2·MAE 비율 | {big:.1%} | |",
         f"| 평균 잔차 (pred−true) | {d['err'].mean():+.2f} → 전체적으로 {'과대' if d['err'].mean() > 0 else '과소'} 예측 | |",
         f"| 골격 수 / 1개짜리 골격 | {n_scaf:,} / {singleton:,}"
         + (" → test 골격이 전부 1개짜리 = scaffold split (골격별 표는 분자 1개 = 오차 1개)" if singleton == n_scaf else "")
         + " | 14쪽 |",
         "", f"## 구간별 (폭 {a.bin_width:.0f} {a.unit})\n", df_to_md(by_range),
         "", "## 분자량 구간별\n", df_to_md(by_mw),
         "", f"## 골격별 상위 15 (분자 수 순)\n", df_to_md(by_scaf.head(15))]
    for k, t in extra_tables.items():
        L += ["", f"## {k}별\n", df_to_md(t)]
    L += ["", f"## 가장 많이 틀린 {a.top}개 → `worst_molecules.csv`, 구조 그림 `worst_molecules.png`\n",
          df_to_md(worst[["smiles", "y_true", "y_pred", "err", "molwt"]].round(1).head(10), index=False, floatfmt=".1f")]
    text = "\n".join(L) + "\n"
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"[saved] {a.out}/summary.md, worst_molecules.csv/.png, error_by_*.csv, fig_*.png")


if __name__ == "__main__":
    main()
