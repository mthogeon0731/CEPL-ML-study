"""shap_descriptors.py — XGBoost + RDKit 기술자 모델의 SHAP 설명 (발표 5절 XAI) [추가: 4주차 내용을 미리]

데이터 CSV → canonical SMILES → RDKit 기술자(16쪽, ~217개) → random 또는 scaffold 80/10/10 분할(14쪽)
→ XGBRegressor(튜닝 없음) → test MAE/RMSE(23쪽) → SHAP TreeExplainer → 요약 그림 + 상위 기술자 표
→ test 예측 CSV (error_analysis.py 입력으로 바로 사용)

사용:
    python shap_descriptors.py ../../261105_Week1_Practice/data/mp.csv --out ../05_analysis_xai --split scaffold
    옵션: --drop-cols MolLogP (logP 과제처럼 정답을 담은 기술자를 뺄 때, 16·38쪽)

출력 (--out):
    fig_shap_summary.png     beeswarm: 상위 20 기술자, 점 색 = 기술자 값, x = SHAP 값 [K]
    fig_shap_bar.png         mean |SHAP| 막대
    shap_top_features.csv    기술자, mean |SHAP|, 방향(기술자 값과 SHAP 값의 상관 부호), 값 범위
    pred_test_xgb_<split>.csv   smiles, y_true, y_pred (test)
    xgb_metrics.md           분할, N, MAE, RMSE, 기준선

주의 [추가]
- SHAP는 "이 모델이 무엇을 보는가"이지 "자연이 그런가"가 아니다. 기술자끼리 상관이 크면(16쪽) 기여가 나뉘어 들어간다.
- 이 모델은 발표 결과표(과제 1단계 노트북)와 별개의 보조 모델이다. 결과표의 XGBoost와 설정이 다를 수 있다.
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
from mlutils import (canonicalize, clean_descriptors, descriptor_matrix, df_to_md, mae,  # noqa: E402
                     random_split, read_dataset, rmse, scaffold_split)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--smiles-col")
    ap.add_argument("--y-col")
    ap.add_argument("--name", default="mp")
    ap.add_argument("--unit", default="K")
    ap.add_argument("--out", default=".")
    ap.add_argument("--split", choices=["random", "scaffold"], default="random")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--max-shap", type=int, default=2000, help="SHAP 계산에 쓸 test 행 수 상한 (속도)")
    ap.add_argument("--drop-cols", default="", help="제외할 기술자 (쉼표 구분), 예: MolLogP")
    ap.add_argument("--max-rows", type=int, default=None, help="빠른 시험용: 앞 N행만")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import shap  # noqa: E402  (43쪽 pip install shap)
    from xgboost import XGBRegressor  # noqa: E402

    # ---- 데이터
    df, sc, yc = read_dataset(a.csv, a.smiles_col, a.y_col)
    if a.max_rows:
        df = df.head(a.max_rows)
    canon = df[sc].map(canonicalize)
    y_all = pd.to_numeric(df[yc], errors="coerce")
    ok = canon.notna() & y_all.notna()
    d = pd.DataFrame({"smiles": canon[ok].to_numpy(), "y": y_all[ok].to_numpy(dtype=float)})
    d = d.groupby("smiles", as_index=False)["y"].median()  # 중복 → 중앙값 (12쪽)
    print(f"[data] rows {len(df):,} → valid unique molecules {len(d):,}")

    # ---- 기술자 (16쪽)
    print("[descriptors] RDKit CalcMolDescriptors ...")
    X = descriptor_matrix(d["smiles"].tolist())
    X = clean_descriptors(X)
    drop = [c.strip() for c in a.drop_cols.split(",") if c.strip()]
    X = X.drop(columns=[c for c in drop if c in X.columns])
    print(f"[descriptors] X shape = {X.shape}  (dropped: {drop or 'none'})")

    # ---- 분할 (14쪽)
    if a.split == "scaffold":
        tr, va, te = scaffold_split(d["smiles"].tolist(), seed=a.seed)
    else:
        tr, va, te = random_split(len(d), seed=a.seed)
    y = d["y"].to_numpy()
    print(f"[split] {a.split}: train {len(tr):,} / valid {len(va):,} / test {len(te):,}")

    # ---- 모델 (19쪽: XGBoost, 튜닝 없음)
    model = XGBRegressor(n_estimators=a.n_estimators, max_depth=a.max_depth, learning_rate=a.learning_rate,
                         subsample=0.8, colsample_bytree=0.8, random_state=a.seed, n_jobs=-1)
    model.fit(X.iloc[tr], y[tr])
    pred_va, pred_te = model.predict(X.iloc[va]), model.predict(X.iloc[te])
    base = mae(y[te], np.full(len(te), y[tr].mean()))
    m_va, m_te, r_te = mae(y[va], pred_va), mae(y[te], pred_te), rmse(y[te], pred_te)
    print(f"[xgb] valid MAE {m_va:.2f}  test MAE {m_te:.2f}  RMSE {r_te:.2f}  baseline(train mean) MAE {base:.2f} [{a.unit}]")

    pred_path = os.path.join(a.out, f"pred_test_xgb_{a.split}.csv")
    train_mean = float(y[tr].mean())
    pd.DataFrame({"smiles": d["smiles"].iloc[te].to_numpy(), "y_true": y[te], "y_pred": pred_te,
                  "split": a.split, "model": "xgb_descriptors", "train_mean": train_mean}).to_csv(pred_path, index=False)

    # ---- SHAP (31쪽 → 4주차)
    print("[shap] TreeExplainer ...")
    rng = np.random.default_rng(a.seed)
    te_sub = te if len(te) <= a.max_shap else rng.choice(te, a.max_shap, replace=False)
    X_te = X.iloc[te_sub]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_te)                       # (n, n_features), 단위 = 목표값 단위
    mean_abs = np.abs(sv).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][: a.top]
    rows = []
    for j in order:
        col = X.columns[j]
        xj = X_te[col].to_numpy()
        corr = float(np.corrcoef(xj, sv[:, j])[0, 1]) if np.std(xj) > 0 else float("nan")
        rows.append({"feature": col, "mean_abs_shap": float(mean_abs[j]),
                     "direction": "+ (value up → prediction up)" if corr > 0.1 else ("- (value up → prediction down)" if corr < -0.1 else "mixed"),
                     "corr(value, shap)": corr, "value_min": float(xj.min()), "value_median": float(np.median(xj)), "value_max": float(xj.max())})
    top = pd.DataFrame(rows).round(3)
    top.to_csv(os.path.join(a.out, "shap_top_features.csv"), index=False)

    plt.figure()
    shap.summary_plot(sv, X_te, max_display=a.top, show=False)
    plt.title(f"SHAP summary — XGBoost + RDKit descriptors ({a.name}, {a.split} split, n={len(te_sub)})", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "fig_shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.summary_plot(sv, X_te, plot_type="bar", max_display=a.top, show=False)
    plt.xlabel(f"mean |SHAP| [{a.unit}]")
    plt.tight_layout()
    plt.savefig(os.path.join(a.out, "fig_shap_bar.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 기록
    md = [f"# XGBoost + RDKit 기술자 — {a.name} [{a.unit}]\n",
          "| 항목 | 값 |", "|---|---|",
          f"| 데이터 | `{os.path.abspath(a.csv)}` → 유효 분자 {len(d):,} (중복은 중앙값) |",
          f"| 기술자 | {X.shape[1]}개 (NaN·상수 열 제거{', ' + ','.join(drop) + ' 제외' if drop else ''}) |",
          f"| 분할 | {a.split} 80/10/10, seed {a.seed} → {len(tr):,} / {len(va):,} / {len(te):,} |",
          f"| 모델 | XGBRegressor(n_estimators={a.n_estimators}, max_depth={a.max_depth}, lr={a.learning_rate}, subsample 0.8, colsample 0.8) — 튜닝 없음 |",
          f"| valid MAE | {m_va:.2f} |", f"| **test MAE** | **{m_te:.2f}** (RMSE {r_te:.2f}) |",
          f"| 기준선 MAE (train 평균 {train_mean:.1f} {a.unit}로 예측) | {base:.2f} → 모델/기준선 {m_te/base:.2f} |",
          "", f"## SHAP 상위 {a.top} 기술자 (test {len(te_sub):,}개)\n", df_to_md(top, index=False, floatfmt=".3f"),
          "", "그림: `fig_shap_summary.png`, `fig_shap_bar.png` · 예측: `" + os.path.basename(pred_path) + "`",
          "", "> 읽는 법: mean |SHAP| = 그 기술자가 예측값을 평균 몇 K 움직이는가. direction은 값↑일 때 예측↑(+)/↓(−). "
              "화학 상식과 맞는지(수소결합 공여체 수, 방향족 고리 수, 분자량, 대칭성 대리변수 …) 대조 = 31쪽 '설명 검증'."]
    with open(os.path.join(a.out, "xgb_metrics.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"[saved] {a.out}/xgb_metrics.md, shap_top_features.csv, fig_shap_summary.png, fig_shap_bar.png, {os.path.basename(pred_path)}")


if __name__ == "__main__":
    main()
