# 랩미팅 그림 — mp(녹는점) MAE 개선 (2026-11-23)

원자료: `mp_mae_push/phase1–3` 의 저장된 예측·표 (재학습 없음). 단위 K, 분할 seed 42, test n 1,995.
random 기준선 = 원본 예측 파일(27.50 K), scaffold 기준선 = 재적합(35.85 K). CI = 부트스트랩 95 %.

| 파일 | 내용 |
|---|---|
| `F01_pipeline.png` | 전체 순서. 멤버 4개는 독립적으로 학습되고, validation에서 정한 가중치로 예측값만 평균한다 (random 분할 기준). |
| `F02_gin_hybrid_architecture.png` | GIN 하이브리드 구조. 앙상블이 아니라 입력이 두 종류인 모델 하나다. |
| `F03_gin_learning_curves.png` | GIN 학습 곡선 (Claude Science T2 트랙 원본 그림). 회색 점선 = 그래프만, 색 = 하이브리드 seed 0–2, 점선 = train 손실, 원 = 복원된 최적 epoch. 세로축 = train 내부 검증 손실(위)·MAE(아래). |
| `F04_gin_graphonly_vs_hybrid_parity.png` | 같은 GIN(seed 1)에 기술자 217개를 붙였을 때의 차이. validation 예측. |
| `F05_test_mae_bars.png` | 모델별 test MAE와 95 % CI. 진한 파랑 = 앙상블, 회색 = 기준선. |
| `F06_delta_mae_forest.png` | 기준선 대비 ΔMAE와 95 % CI. 같은 test 분자에서 짝지어 비교. |
| `F07_parity_baseline_vs_A.png` | test parity. 대각선 = 완벽한 예측. 고융점 쪽이 대각선 아래로 처지는 경향(평균 쪽 쏠림)이 남아 있다. |
| `F08_residual_by_mp_bin.png` | 측정 MP 구간별 평균 잔차와 95 % CI. 0 위 = 과대예측, 0 아래 = 과소예측. |
| `F09_error_correlation_heatmap.png` | 멤버 모델 간 오차(예측 − 측정) 상관. 상관이 낮을수록 섞었을 때 오차가 상쇄된다. 색 하한 0.85. |
| `F10_lgb_vs_gin_error_scatter.png` | 같은 test 분자에서 두 모델의 오차. 대각선 위에 모이면 같이 틀린 것, 흩어질수록 평균으로 상쇄될 여지가 크다. |
| `F11_ensemble_weights.png` | 앙상블 A의 가중치. 가중치는 음수 불가, 합 1, validation에서만 결정. |
| `F12_optuna_convergence.png` | Optuna 튜닝 경과. 세로축은 train 안 내부 분할 MAE라 validation 값과 직접 비교하지 않는다. |
| `F13_learning_curve.png` | 고정 test에서 학습 데이터 양만 바꾼 학습 곡선. 띠 = seed 3개의 최소–최대. |
| `F14_text_llm_representations.png` | 같은 XGBoost에서 입력 표현만 바꿨을 때 RDKit 기술자 대비 ΔMAE (validation). |
| `F15_risk_coverage.png` | 신뢰도 점수(오차 예측 메타모델) 순으로 분자를 고를 때 커버리지별 MAE. 2 % 미만 구간은 표본이 적어 생략. |
| `F16_cost_vs_mae.png` | 후보별 총비용(클라우드 환산 USD + LLM API, 상한)과 validation MAE. 30 K 위 후보는 잘림. |

생성 스크립트: `make_figs.py`
