# run_notes — 2026-11-09 실행 기록 (발표 수치의 출처)

실행 환경: Claude 작업 컨테이너 (Linux, CPU 2코어, Python 3.13) — **사용자 PC의 `mlstudy` 환경이 아님.** 패키지 판은 거의 같음: RDKit 2026.09.1, scikit-learn 1.9.1, XGBoost **3.4.1**(PC는 3.4.2), PyTorch 2.15.0(cu132 빌드, CPU 실행; PC는 +cpu), PyG 2.8.1, shap 0.54.1(PC는 0.51.0). seed 42가 같으므로 표본·분할·기준선은 동일하다. ~~모델 수치는 소수점 아래에서만 다를 수 있다~~ → **2026-11-09 PC 재현으로 정정**: 결정적 모델(Ridge, SVR, MLP, 기술자 kNN)은 같고, XGBoost·RF는 최대 0.9 K, **GIN은 0.5 K 차이에 epoch 수까지 다름**(random 145 → 79 epochs). 아래 "PC 재현 (v2)" 절. 학습 시간(sec)은 이 컨테이너 기준 → PC와 다름.

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

## PC 재현 (v2) — 2026-11-09, 사용자 PC의 `mlstudy` 환경 (`../results_pc_v2/`)

> **2026-11-09 밤: 발표 채택 수치 = 이 PC v2.** 덱·`04_performance/results_table.md` ★ 절·`slides/figs_pc/`·`05_analysis_xai_pc/`가 이 수치 기준. 컨테이너 수치(위 표, `results/`)는 참고용.

환경: `C:\Users\mthog\miniforge3\envs\mlstudy` — Python 3.11.18, RDKit 2026.09.1, scikit-learn 1.9.1, XGBoost 3.4.2, PyTorch 2.15.0+cpu, PyG 2.8.1, shap 0.51.0 (`env/check_env_log.txt`). 교수님 코드 **무수정** (노트북은 위 run 1과 같은 `PROP='mp'` 사본). 이 폴더의 원본 파일은 덮어쓰지 않았다.

| run | 컨테이너 (위 표) | **PC v2** | 차이 |
|---|---|---|---|
| 노트북 기준선 random / scaffold | 72.51 / 75.96 K | 72.51 / 75.96 K | 0 |
| 노트북 최저 XGBoost + 기술자 random / scaffold | 29.97 / 37.77 K | **29.47 / 37.39 K** | −0.49 / −0.38 |
| 노트북 24칸 중 최대 차이 | | | 0.49 K (XGBoost+기술자 random); Ridge·SVR·MLP·기술자 kNN = 0 |
| GIN `train.py --prop mp` (random) | 36.52 K, 145 epochs | **35.97 K, 79 epochs** (RMSE 49.21, R² 0.730) | −0.55 |
| GIN `--split scaffold` ([추가]) | 41.81 K, 45 epochs | **42.07 K, 45 epochs** (RMSE 55.24, R² 0.425) | +0.26 |
| 같은 test 500: XGB + 기술자 random / scaffold | 32.71 / 36.38 K | **33.58 / 36.46 K** | +0.87 / +0.09 |
| 같은 test 500 순위 | XGB+기술자 < RF+기술자 < **GIN 3위** | 같음 (두 분할) | – |
| GIN − XGB 부트스트랩 95 % CI, random | [1.26, 6.48] | **[0.01, 4.67]** | 하한이 0에 닿음 |
| GIN − XGB 부트스트랩 95 % CI, scaffold | [2.84, 7.90] | [2.94, 8.13] | 견고 |

- PC에서 GIN은 결정적: 두 번 실행(아래 OpenMP 우회 전·후) 모두 35.9733 / 42.0699 K, 79 / 45 epochs → **우회 설정이 수치를 바꾸지 않음**을 확인.
- 전체 표: `../results_pc_v2/compare_container_vs_pc.csv`, `bootstrap_gin_vs_xgb.csv`. 실행된 노트북: `../results_pc_v2/01_multi_model_comparison_mp_executed_pc.ipynb` (오류 0). 그림: `train_py_results/`, `notebook_results_nb_mp/`.

### ⚠ PC 환경 문제: `train.py`가 그림 저장 단계에서 죽음 (OpenMP Error #15)

- 증상: `conda activate mlstudy` 상태(= env의 `Library\bin`이 PATH에 있음)에서 `python train.py --prop mp`를 돌리면 **학습·`summary.csv`·`_test_pred.csv`까지는 저장되고**, parity/학습곡선 `savefig`에서 `OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized`로 종료 (종료 코드 3 또는 0xC06D007F). 그림 2장(과제 2단계 제출물)이 안 생긴다.
- 원인: PyTorch pip 휠이 싣는 Intel OpenMP(`torch\lib\libiomp5md.dll`)와 conda-forge의 LLVM OpenMP(`Library\bin\libomp.dll`, numpy/matplotlib 경로)가 한 프로세스에 함께 로드됨. 노트북은 torch를 import하지 않아 영향 없음.
- 확인된 우회: 실행 전에 `set KMP_DUPLICATE_LIB_OK=TRUE` (PowerShell: `$env:KMP_DUPLICATE_LIB_OK="TRUE"`). 이번 재현에서 수치가 우회 전과 소수점 넷째 자리까지 같았다. Intel이 "unsupported"로 표시한 설정이므로 **결과 수치를 우회 전 `summary.csv`와 대조하는 습관**을 유지한다.
- 근본 해결 후보(미시험): conda-forge에서 torch를 함께 설치해 OpenMP를 하나로 맞추기(`conda install -c conda-forge pytorch-cpu`) — 2주차 전에 팀원 환경에서도 같은 문제가 나는지 확인할 것.
- (참고) 이번 재현은 Claude 샌드박스 안에서 돌렸기 때문에 추가로 임시 폴더 경로, Jupyter 연결 파일 권한, joblib 스레드 풀(named pipe 금지 → `sitecustomize.py`로 순차 실행) 우회를 썼다(`../results_pc_v2/logs/`). **이 세 가지는 PC에서 직접 실행할 때는 필요 없다.** 모든 모델이 `random_state=SEED`라 순차/병렬 실행의 결과는 같다.

## 재현

```
conda activate mlstudy
cd "C:\Users\mthog\CEPL ML\261105_Week1_Practice"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 01_multi_model_comparison_mp.ipynb --output 01_multi_model_comparison_mp_executed.ipynb
set KMP_DUPLICATE_LIB_OK=TRUE          & REM PC mlstudy에서 train.py 그림 저장 충돌(OpenMP Error #15) 우회 — 위 PC 재현 절
python train.py --prop mp
python train.py --prop mp --split scaffold
cd ..\mp_presentation
python scripts\fair_compare.py ..\261105_Week1_Practice results random
python scripts\error_analysis.py results\pred_test_XGBoost_Descriptor_scaffold.csv --out 05_analysis_xai\xgb_descriptor_scaffold --top 20
```

같은 실험을 PC에서 다시 돌리면 이 폴더의 파일을 덮어쓰지 말고 `_v2`를 붙여 비교한다 (수치가 소수점에서 달라질 수 있음).
