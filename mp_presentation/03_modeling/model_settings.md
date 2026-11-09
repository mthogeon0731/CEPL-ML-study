# 모델 설정표 — 왜 이 모델들을 썼고, 각각 어떤 특징인가 (19·21쪽을 mp 관점으로)

설정값 = 노트북 5절 `get_models()`·`train.py` 코드의 **실제값** (튜닝 없음, 37쪽). "기대" 열은 결과를 보기 전에 가설로 적은 것, "결과" 열은 `../results/mae_table_mp.csv`·`summary.csv`에서 옮긴 것 (test MAE [K], random / scaffold).

## 왜 6종을 먼저 돌리나

19쪽 첫 줄: **단순한 모델부터 → 기준 점수 → 복잡한 모델이 그 점수를 넘는지 확인.** 기준선(train 평균으로 예측) MAE = **72.5 K (random) / 76.0 K (scaffold)**. 모델이 이 숫자보다 얼마나 낮은가가 "좋다/나쁘다"의 첫 기준 (23쪽).

"임베딩" = 분자를 숫자 벡터로 바꾸는 것. 기술자·지문은 **사람이 정한 특징**, GIN의 h_G는 **모델이 배우는 특징** (15쪽 왼쪽 → 오른쪽, 21쪽). 두 쪽을 같은 표에 놓고 비교하는 것이 이번 과제.

## 모델별 설정·가설·결과

| 모델 | 한 줄 직관 (19쪽) | 입력 | 주요 설정 (코드 실제값) | mp에서 기대 (가설, 결과 보기 전) | 결과: 기술자 r / s · 지문 r / s [K] | 가설과 맞았나 |
|---|---|---|---|---|---|---|
| Ridge | 특징의 가중합, L2로 계수 억제 | 기술자 / 지문 | `StandardScaler` + `Ridge(alpha=1.0)` | 가법적 기여만 → 기준선보다 조금 나은 정도 | 35.4 / 45.1 · 61.0 / **96.2** | 기술자는 기대보다 좋음(기준선의 절반). 지문 scaffold는 **기준선(76.0)보다 나쁨** — 24쪽 logP Ridge+Morgan 지는 케이스와 같은 모양 |
| kNN | 가장 비슷한 분자 k개의 (거리 가중) 평균 | 기술자 / 지문 | `StandardScaler` + `KNeighborsRegressor(n_neighbors=5, weights='distance')` — 유클리드 (17쪽 Tanimoto 아님) | 비슷한 분자 많은 영역 강함, 새 골격 약함 | 38.5 / 51.7 · **86.2 / 125.9** | 기술자는 기대대로(새 골격에서 +13 K). 지문은 **random에서도 기준선보다 나쁨** → 표준화된 희소 0/1 비트 위의 유클리드 거리가 "비슷함"을 못 잡음 ([추가] 해석) |
| SVR | 커널(RBF)로 비선형, ε 밖 오차만 벌점 | 기술자 / 지문 | `StandardScaler` + `SVR(C=10, epsilon=0.1)` | N=5,000이면 가능; 전체 19,945면 느림 (19쪽) | 35.4 / 44.0 · 54.8 / 55.5 | 기술자 중위권. 지문에서는 유일하게 random≈scaffold (+0.7 K) — 둘 다 평평하게 나쁨 |
| Random forest | 깊은 나무 여러 개의 평균 | 기술자 / 지문 | `RandomForestRegressor(n_estimators=300, n_jobs=-1)` | 튜닝 없이 안정 | 33.5 / 40.7 · 40.9 / 55.5 | 기대대로 안정, 2위. 지문에서도 트리 모델은 살아남음 |
| XGBoost | 앞 나무의 오차를 다음 나무가 보정 | 기술자 / 지문 | `XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.5)` | 표 형식 특징, 수천 개 이상 → 유력 | **30.0 / 37.8** · 39.5 / 50.3 | **4열 모두 1위.** 가설 적중 |
| MLP | 층을 쌓은 비선형 함수 | 기술자 / 지문 | `StandardScaler` + `MLPRegressor((256, 128), alpha=1e-3, early_stopping=True, max_iter=300)` | 데이터 5,000이면 과적합 주의 | 34.4 / 53.6 · 67.0 / **143.1** | 기술자 random은 Ridge급이지만 scaffold에서 +19 K. 지문 scaffold **143 K = 기준선의 1.9배** → 20쪽 "학습 범위 밖에서 임의" 그대로 |
| **GIN** (`train.py`) | 이웃 원자 정보를 층마다 합쳐 전달 (21쪽 식) | 그래프 (원자 147; 결합 16은 GIN이 안 씀) | 5층, hidden 128, 원자 벡터 64, `global_mean_pool`, Linear(64→1), **147,009 파라미터**; AdamW 1e-3, ReduceLROnPlateau(10, ×0.8), 기울기 ±10, dropout 0.3, **batch 64**, epochs ≤ 300, 조기 종료 = validate MAE 30 epoch 정체, seed 42, **80/10/10** | 데이터 5,000은 GNN엔 적은 편 (19쪽 "1만 개 이상") → 최고 고전 모델(XGBoost+기술자)을 못 넘을 수 있음 | `../results/summary.csv` → `../04_performance/results_table.md` GIN 행 | → 4절에서 판정 |

r = random, s = scaffold. 노트북은 80/20, GIN은 80/10/10이라 **test 집합이 다름** (같은 5,000개 표본, seed 42).

## 결과를 한 줄로 (3절 → 4절 연결)

- **표현이 모델보다 컸다**: 기술자(217)는 6모델 모두 기준선의 41–53 %(random), 지문(2048)은 트리 모델만 살고 Ridge·kNN·MLP는 scaffold에서 기준선 이하. mp처럼 "분자 전체의 극성·크기·강직성"이 좌우하는 물성은 부분구조 유무(지문)보다 전체 요약값(기술자)이 맞는다는 쪽.
- **트리 앙상블(XGBoost, RF)이 두 표현 모두에서 1·2위** — 19쪽 표 "표 형식 특징, 튜닝 없이 안정"과 일치.
- **거리·내적 기반 모델(kNN, Ridge, MLP)이 희소 지문에서 무너진다** — 표현과 모델의 궁합(15쪽: 지문 → kNN·RF·XGB·MLP가 "맞는 모델"이라고 되어 있지만, 노트북의 kNN은 Tanimoto가 아니라 표준화 유클리드라는 점이 다르다).

## 21쪽 연구실 설정과 코드의 차이

21쪽 연구실 설정(5층, hidden 128, 지문 64, AdamW 0.001, mean pool, dropout 0.3)과 `train.py` 코드는 같다. 다른 점은 **batch 64**(표본 5,000 < 1만, 21쪽 규칙 그대로)와 **Scaler를 train으로만 fit** (14쪽 누수 예를 코드가 고침). 2주차 옵션: `--conv GINE`(결합 특징 사용), `--pooling sum`(GIN 논문 readout), `--split scaffold`, `--bayes`(BGNN, 22쪽).

## 자가점검 답

- 15쪽 표현 ↔ 19쪽 모델: 기술자 → Ridge·RF·XGB·SVR / 지문 → kNN·RF·XGB·MLP / 그래프 → GCN·GIN·BGIN. 노트북은 두 표현에 6모델을 전부 돌려 "맞는 모델"이 실제로 맞는지 확인한 셈.
- GIN 식 ↔ 코드: `code_summary.md` 표.
- 학습 손실 MSE vs 보고 MAE (9쪽): MSE는 미분이 매끈하고 큰 오차를 더 벌해 학습용, MAE는 단위가 K라 사람이 읽는 보고용. `train.py`가 정확히 그렇게 나눠 쓴다.
- "튜닝하지 않았다" = 24칸을 같은 조건에서 비교하는 **공정한 기준선** (37쪽) + 49쪽 규칙(test 성능을 보고 설정을 고르지 않음).
