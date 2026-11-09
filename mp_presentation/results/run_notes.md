# run_notes — 2026-11-09 실행 기록 (발표 수치의 출처)

실행 환경: Claude 작업 컨테이너 (Linux, CPU 2코어, Python 3.13) — **사용자 PC의 `mlstudy` 환경이 아님.** 패키지 판은 거의 같음: RDKit 2026.09.1, scikit-learn 1.9.1, XGBoost **3.4.1**(PC는 3.4.2), PyTorch 2.15.0(cu132 빌드, CPU 실행; PC는 +cpu), PyG 2.8.1, shap 0.54.1(PC는 0.51.0). seed 42가 같으므로 표본·분할은 동일하고, 모델 수치는 소수점 아래에서만 다를 수 있다. 학습 시간(sec)은 이 컨테이너 기준 → PC와 다름.

| run | 명령 / 파일 | 코드 수정 | 산출물 (이 폴더) | 핵심 수치 |
|---|---|---|---|---|
| 1 노트북 (과제 1) | `01_multi_model_comparison_mp.ipynb` = 교수님 노트북 사본, 첫 셀 `PROP = 'mp'`만 변경 + 끝에 [추가] 저장·SHAP 셀. `jupyter nbconvert --execute`로 전체 실행 (00:38–00:46) | 교수님 셀 **무수정** (PROP만). 추가 셀은 결과표에 영향 없음 (재학습 XGBoost 예측 = 결과표 예측, 최대 차이 0.0 K 확인) | `results_mp.csv`(24행: split, feature, model, MAE, RMSE, R2, sec), `mae_table_mp.csv`(피벗), `run_info.json`(기준선·분할·골격 수), `notebook_outputs_mp.txt`(셀 출력 전문), `pred_test_<model>_<feature>_<split>.csv`(24개), `notebook_figs/fig01–09`, `shap_top20_*.csv`, `rf_importance_*.csv`; 실행된 노트북 = `../../261105_Week1_Practice/01_multi_model_comparison_mp_executed.ipynb` | 기준선 72.5 / 76.0 K · 최저 XGBoost+기술자 30.0 (random) / 37.8 (scaffold) |
| 2 GIN (과제 2) | `python train.py --prop mp` (00:46–00:50, 240 s) | **없음** | `summary.csv` 1행, `mp_GNN-GIN_mean_random_{parity,curve}.png` (= `fig_parity_gin_test.png`, `fig_learning_curve_gin.png`), `mp_GNN-GIN_mean_random_test_pred.csv` (= `pred_test_GIN_graph_random_80-10-10.csv`), `train_log_gin_mp.txt`(터미널 전문) | test MAE **36.52** K, RMSE 50.92, R² 0.711, 기준선 75.15, 145 epochs (최선 114), 147,009 파라미터, batch 64 |
| 3 [추가] GIN scaffold | `python train.py --prop mp --split scaffold` (00:51–00:57, 345 s) — 2주차 옵션 미리 실행 | 없음 | `summary.csv` 2행, `mp_GNN-GIN_mean_scaffold_*`, `train_log_gin_mp_scaffold.txt` | test MAE **41.81** K, RMSE 54.50, R² 0.440, 기준선 79.74, **45 epochs**(최선 14 — validate가 희귀 골격이라 잡음 커서 조기 종료가 일찍 걸림) |
| 4 [추가] 같은 분할 비교 | `fair_compare.py practice results [random\|scaffold]` — train.py의 80/10/10 분할을 재현(assert로 test SMILES 일치 확인)하고 노트북 6모델을 같은 train 4,000으로 학습 | 노트북 `get_models()` 설정 그대로 | `fair_compare_gin_split.csv` (random), `fair_compare_gin_split_scaffold.csv`, `*_info.json`, `pred_test_XGBoost_Descriptor_ginsplit_*.csv` | random: XGB+기술자 32.7 vs GIN 36.5 · scaffold: XGB+기술자 36.4 vs GIN 41.8 (같은 test 500) |
| 5 [추가] 후분석 | `scripts/error_analysis.py` → `../05_analysis_xai/{xgb_descriptor_random, xgb_descriptor_scaffold, gin_random}/` (`--train-mean` = 각 분할의 train 평균: 378.1 / 365.9 / 377.8 K) | – | summary.md, worst_molecules.csv/.png, error_by_range/scaffold/molwt.csv, fig_residuals.png, fig_error_by_molwt.png | 구간별 오차·틀린 분자 → `05_analysis_xai/xai_notes.md` |
| 6 [추가] 슬라이드 그림 | `make_slide_figs.py` → `../slides/figs/` (원본 CSV만 읽음, 재계산 없음) | – | fig_S4_results_table, fig_S5_parity_xgb_gin, fig_S6_error_by_range, fig_S6_shap_top10, fig_S6_worst_molecules, fig_S7_same_split_ranking | |

## 분할·기준선 한눈에

| 분할 | train / (val) / test | train 평균 [K] | 기준선 MAE [K] | 비고 |
|---|---|---|---|---|
| 노트북 random 80/20 | 4,000 / – / 1,000 | 378.1 | 72.5 | test 골격 79 %가 train에 있음 |
| 노트북 scaffold 80/20 | 4,000 / – / 1,000 | 365.9 | 76.0 | test 골격 전부 1개짜리, 겹침 0 |
| train.py random 80/10/10 | 4,000 / 500 / 500 | 377.8 | 75.2 | GIN·같은 분할 비교 |
| train.py scaffold 80/10/10 | 4,000 / 500 / 500 | 365.9 | 79.7 | GIN scaffold ([추가]) |

## 재현

```
conda activate mlstudy
cd "C:\Users\mthog\OneDrive\바탕 화면\CEPL ML\261105_Week1_Practice"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 01_multi_model_comparison_mp.ipynb --output 01_multi_model_comparison_mp_executed.ipynb
python train.py --prop mp
python train.py --prop mp --split scaffold
cd ..\mp_presentation
python scripts\fair_compare.py ..\261105_Week1_Practice results random
python scripts\error_analysis.py results\pred_test_XGBoost_Descriptor_scaffold.csv --out 05_analysis_xai\xgb_descriptor_scaffold --top 20
```

같은 실험을 PC에서 다시 돌리면 이 폴더의 파일을 덮어쓰지 말고 `_v2`를 붙여 비교한다 (수치가 소수점에서 달라질 수 있음).
