"""mp_data_card.py — 데이터 카드 만들기 (발표 1절, 2절 정제 로그의 숫자) [추가: 연구실 코드가 아님]

사용:
    python mp_data_card.py <CSV> [--smiles-col smiles] [--y-col mp] [--name mp] [--unit K] [--out ../01_data]

출력 (--out 폴더):
    data_card.md            데이터 카드 (출처·측정 조건 줄은 손으로 채움)
    fig_<name>_hist.png     목표값 분포
    fig_molwt_vs_<name>.png 분자량 vs 목표값
    duplicates.csv          같은 canonical SMILES에 여러 값 (12쪽: 값 차이 큰 행부터 확인)

슬라이드 근거: 12쪽(canonical, 무효, 중복, 염, 입체), 13쪽(단위, 이상치, 출처), 14쪽(골격), 48쪽(mp 19,945, K)
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlutils import (canonicalize, has_salt, has_stereo, heavy_atoms, mol_weights,  # noqa: E402
                     murcko_scaffold, read_dataset)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--smiles-col")
    ap.add_argument("--y-col")
    ap.add_argument("--name", default="mp", help="물성 이름 (파일명·제목용)")
    ap.add_argument("--unit", default="K")
    ap.add_argument("--out", default=".", help="출력 폴더")
    ap.add_argument("--bin-width", type=float, default=50.0, help="히스토그램·구간표 구간 폭 (K)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df, sc, yc = read_dataset(a.csv, a.smiles_col, a.y_col)
    n_raw = len(df)
    y_raw = pd.to_numeric(df[yc], errors="coerce")
    n_y_missing = int(y_raw.isna().sum())

    # ---- SMILES 정리 (12쪽)
    canon = df[sc].map(canonicalize)
    n_invalid = int(canon.isna().sum())
    salt = df[sc].map(has_salt)
    stereo = df[sc].map(has_stereo)
    ok = canon.notna() & y_raw.notna()
    d = pd.DataFrame({"smiles_raw": df[sc], "smiles": canon, "y": y_raw})[ok].reset_index(drop=True)

    # ---- 중복 (12쪽)
    g = d.groupby("smiles")["y"]
    dup = g.agg(["count", "median", "min", "max"])
    dup = dup[dup["count"] > 1]
    dup["range"] = dup["max"] - dup["min"]
    dup = dup.sort_values("range", ascending=False)
    n_dup_mols = len(dup)
    n_dup_rows = int(dup["count"].sum() - n_dup_mols)  # 중앙값으로 합치면 사라지는 행 수
    dup.to_csv(os.path.join(a.out, "duplicates.csv"))

    # ---- 분자 통계
    d["molwt"] = mol_weights(d["smiles"])
    d["heavy"] = heavy_atoms(d["smiles"])
    d["scaffold"] = d["smiles"].map(murcko_scaffold)
    n_scaf = d["scaffold"].nunique()
    n_acyclic = int((d["scaffold"] == "").sum())
    top_scaf = d["scaffold"].replace("", "(acyclic)").value_counts().head(10)

    # ---- 목표값 통계 (13쪽)
    y = d["y"].to_numpy()
    q = np.percentile(y, [1, 5, 25, 50, 75, 95, 99])
    looks_celsius = (np.median(y) < 200) or ((y < 0).mean() > 0.01)
    lo = np.floor(y.min() / a.bin_width) * a.bin_width
    hi = np.ceil(y.max() / a.bin_width) * a.bin_width
    edges = np.arange(lo, hi + a.bin_width, a.bin_width)
    counts, _ = np.histogram(y, bins=edges)
    range_rows = [(f"{edges[i]:.0f}–{edges[i+1]:.0f}", int(counts[i]), f"{100*counts[i]/len(y):.1f}%")
                  for i in range(len(counts)) if counts[i] > 0]

    # ---- 출처 열 (13쪽)
    source_tbl = None
    for c in df.columns:
        if c.lower() in ("source", "src", "ref", "reference", "dataset", "origin"):
            source_tbl = df.loc[ok, c].value_counts()
            break

    # ---- 그림 (영문 라벨: Windows matplotlib 기본 글꼴에 한글 없음)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(y, bins=edges, color="#4C72B0", edgecolor="white")
    ax.set_xlabel(f"{a.name} [{a.unit}]")
    ax.set_ylabel("count")
    ax.set_title(f"{a.name} distribution (N = {len(y):,})")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"fig_{a.name}_hist.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(d["molwt"], y, s=4, alpha=0.3, color="#4C72B0")
    ax.set_xlabel("MolWt [g/mol]")
    ax.set_ylabel(f"{a.name} [{a.unit}]")
    corr = float(np.corrcoef(d["molwt"].fillna(d["molwt"].median()), y)[0, 1])
    ax.set_title(f"MolWt vs {a.name}  (Pearson r = {corr:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, f"fig_molwt_vs_{a.name}.png"), dpi=150)
    plt.close(fig)

    # ---- 데이터 카드
    L = []
    L.append(f"# 데이터 카드 — {a.name} [{a.unit}]\n")
    L.append(f"- 파일: `{os.path.abspath(a.csv)}`  (열: {', '.join(map(str, df.columns))})")
    L.append(f"- SMILES 열 `{sc}`, 목표값 열 `{yc}`")
    L.append("- 출처 · 측정 조건: **(data/README_data.md를 보고 채우기)**  ← 13쪽: 측정법, 온도, 순도, 분해 여부")
    L.append("")
    L.append("## 정제 로그 (12쪽)\n")
    L.append("| 단계 | 개수 | 비고 |")
    L.append("|---|---|---|")
    L.append(f"| 원본 행 | {n_raw:,} | |")
    L.append(f"| 목표값 결측 | {n_y_missing:,} | 제외 |")
    L.append(f"| 무효 SMILES (`MolFromSmiles → None`) | {n_invalid:,} | 제외 후 개수 기록 |")
    L.append(f"| 염·혼합물 (`.` 포함) | {int(salt[ok].sum()):,} | 가장 큰 조각·중성화 여부 결정 (아직 처리 안 함) |")
    L.append(f"| 입체 표기 (`@`, `/`, `\\`) | {int(stereo[ok].sum()):,} | 그래프 특징에 입체 정보 없음 → 입체이성질체는 같은 입력 |")
    L.append(f"| 유효 행 | {len(d):,} | |")
    L.append(f"| 같은 canonical SMILES에 여러 값인 분자 | {n_dup_mols:,} 분자 ({n_dup_rows:,} 행 중복) | 중앙값으로 합치면 {len(d)-n_dup_rows:,} 분자; `duplicates.csv`의 range 큰 행부터 확인 |")
    if n_dup_mols:
        L.append(f"| 중복 값 차이 (max−min) | 중앙값 {dup['range'].median():.1f}, 최대 {dup['range'].max():.1f} {a.unit} | 차이 큼 → 출처·단위 착오 의심 (12쪽) |")
    L.append("")
    L.append(f"## 목표값 {a.name} [{a.unit}] (13쪽)\n")
    L.append("| N | min | 1% | 5% | 25% | median | 75% | 95% | 99% | max | mean | std |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    L.append(f"| {len(y):,} | {y.min():.1f} | {q[0]:.1f} | {q[1]:.1f} | {q[2]:.1f} | {q[3]:.1f} | {q[4]:.1f} | {q[5]:.1f} | {q[6]:.1f} | {y.max():.1f} | {y.mean():.1f} | {y.std(ddof=1):.1f} |")
    if looks_celsius:
        L.append(f"\n> **경고**: 값이 °C처럼 보입니다 (중앙값 {np.median(y):.0f}, 음수 비율 {(y<0).mean():.1%}). 48쪽: bp, mp = 켈빈. °C → K = +273.15 (12쪽 2024 전처리).")
    L.append("")
    L.append(f"### 구간별 개수 (폭 {a.bin_width:.0f} {a.unit}) — 끝 영역 데이터 수 = 4절 parity 끝 영역 해석의 근거\n")
    L.append(f"| {a.name} 구간 [{a.unit}] | 개수 | 비율 |")
    L.append("|---|---|---|")
    L += [f"| {r} | {c:,} | {p} |" for r, c, p in range_rows]
    L.append("")
    L.append("## 분자 (14, 16쪽)\n")
    L.append("| 항목 | 값 |")
    L.append("|---|---|")
    L.append(f"| MolWt [g/mol] min / median / max | {np.nanmin(d['molwt']):.1f} / {np.nanmedian(d['molwt']):.1f} / {np.nanmax(d['molwt']):.1f} |")
    L.append(f"| 무거운 원자 수 min / median / max | {np.nanmin(d['heavy']):.0f} / {np.nanmedian(d['heavy']):.0f} / {np.nanmax(d['heavy']):.0f} |")
    L.append(f"| MolWt–{a.name} Pearson r | {corr:.2f} |")
    L.append(f"| Murcko 골격 수 (고유) | {n_scaf:,} (비고리 분자 {n_acyclic:,}개는 빈 골격 하나로) |")
    L.append(f"| 골격당 평균 분자 수 | {len(d)/max(n_scaf,1):.1f} → scaffold split에서 test가 '처음 보는 골격'이 되는 정도 |")
    L.append("")
    L.append("### 상위 골격 10 (scaffold split 때 train에 먼저 들어가는 집합)\n")
    L.append("| 골격 SMILES | 분자 수 |")
    L.append("|---|---|")
    L += [f"| `{s}` | {c:,} |" for s, c in top_scaf.items()]
    if source_tbl is not None:
        L.append("")
        L.append("## 출처별 개수 (13쪽: 출처별 오차를 따로 계산)\n")
        L.append("| source | 개수 |")
        L.append("|---|---|")
        L += [f"| {s} | {c:,} |" for s, c in source_tbl.items()]
    L.append("")
    L.append(f"그림: `fig_{a.name}_hist.png`, `fig_molwt_vs_{a.name}.png`")
    card = "\n".join(L) + "\n"
    with open(os.path.join(a.out, "data_card.md"), "w", encoding="utf-8") as f:
        f.write(card)
    print(card)
    print(f"[saved] {os.path.join(a.out, 'data_card.md')}")


if __name__ == "__main__":
    main()
