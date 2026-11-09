# XGBoost + RDKit 기술자 — mp [K]

| 항목 | 값 |
|---|---|
| 데이터 | `261105_Week1_Practice/data/mp.csv` → 유효 분자 19,945 (중복은 중앙값) |
| 기술자 | 203개 (NaN·상수 열 제거) |
| 분할 | scaffold 80/10/10, seed 42 → 15,956 / 1,994 / 1,995 |
| 모델 | XGBRegressor(n_estimators=600, max_depth=6, lr=0.05, subsample 0.8, colsample 0.8) — 튜닝 없음 |
| valid MAE | 35.34 |
| **test MAE** | **36.76** (RMSE 49.21) |
| 기준선 MAE (train 평균 366.9 K로 예측) | 78.23 → 모델/기준선 0.47 |

## SHAP 상위 20 기술자 (test 1,995개)

| feature | mean_abs_shap | direction | corr(value, shap) | value_min | value_median | value_max |
|---|---|---|---|---|---|---|
| RingCount | 24.904 | + (value up → prediction up) | 0.358 | 1.000 | 3.000 | 135.000 |
| TPSA | 24.292 | + (value up → prediction up) | 0.723 | 0.000 | 48.140 | 777.980 |
| BertzCT | 17.156 | + (value up → prediction up) | 0.528 | 11.245 | 694.577 | 5967.703 |
| NumHDonors | 12.258 | + (value up → prediction up) | 0.678 | 0.000 | 0.000 | 25.000 |
| NumRotatableBonds | 7.940 | - (value up → prediction down) | -0.855 | 0.000 | 3.000 | 28.000 |
| SMR_VSA10 | 4.553 | + (value up → prediction up) | 0.903 | 0.000 | 16.941 | 170.673 |
| FpDensityMorgan3 | 4.527 | - (value up → prediction down) | -0.897 | 0.250 | 2.321 | 3.625 |
| SlogP_VSA8 | 3.764 | + (value up → prediction up) | 0.766 | 0.000 | 0.000 | 76.825 |
| FpDensityMorgan2 | 3.450 | - (value up → prediction down) | -0.879 | 0.188 | 1.682 | 2.750 |
| NHOHCount | 3.079 | + (value up → prediction up) | 0.723 | 0.000 | 0.000 | 25.000 |
| FractionCSP3 | 3.011 | - (value up → prediction down) | -0.746 | 0.000 | 0.273 | 1.000 |
| SlogP_VSA1 | 2.903 | + (value up → prediction up) | 0.712 | 0.000 | 4.737 | 87.153 |
| fr_bicyclic | 2.755 | + (value up → prediction up) | 0.921 | 0.000 | 1.000 | 14.000 |
| VSA_EState3 | 2.720 | + (value up → prediction up) | 0.536 | -1.197 | 3.593 | 261.202 |
| AvgIpc | 2.588 | - (value up → prediction down) | -0.829 | 0.722 | 2.724 | 3.949 |
| HallKierAlpha | 2.423 | - (value up → prediction down) | -0.719 | -18.140 | -2.110 | 1.860 |
| NumHeterocycles | 2.412 | + (value up → prediction up) | 0.216 | 0.000 | 1.000 | 135.000 |
| SlogP_VSA2 | 2.249 | + (value up → prediction up) | 0.362 | 0.000 | 19.183 | 368.426 |
| PEOE_VSA7 | 2.057 | + (value up → prediction up) | 0.750 | 0.000 | 30.183 | 160.520 |
| SMR_VSA5 | 2.004 | - (value up → prediction down) | -0.586 | 0.000 | 16.212 | 240.837 |

그림: `fig_shap_summary.png`, `fig_shap_bar.png` · 예측: `pred_test_xgb_scaffold.csv`

> 읽는 법: mean |SHAP| = 그 기술자가 예측값을 평균 몇 K 움직이는가. direction은 값↑일 때 예측↑(+)/↓(−). 화학 상식과 맞는지(수소결합 공여체 수, 방향족 고리 수, 분자량, 대칭성 대리변수 …) 대조 = 31쪽 '설명 검증'.
