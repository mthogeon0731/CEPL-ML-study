# 결과표 — mp [K] (37쪽 양식), 실행 2026-11-09

**mp, 표본 5,000 / 전체 19,945** (`MAX_N = 5000`, seed 42), 노트북 분할 = random 80/20 · scaffold 80/20 (train 4,000 / test 1,000), 값 = **test MAE [K]** (낮을수록 좋음), **하이퍼파라미터 튜닝 없음**. 원본: `../results/results_mp.csv`, `mae_table_mp.csv`, `run_info.json`, `summary.csv`.
GIN 행: `train.py` 기본 = 같은 5,000개 표본이지만 **80/10/10** 분할(train 4,000 / validate 500 / test 500) → test 집합이 다름 (각주 †).

## 표 1 — 모델 6 × 표현 2 × 분할 2 (+ GIN, 기준선)

| 모델 | 기술자 random | 기술자 scaffold | Morgan random | Morgan scaffold | 학습 시간 [s]† |
|---|---|---|---|---|---|
| Ridge | 35.4 | 45.1 | 61.0 | **96.2 ✗** | 0.0 / 0.6 |
| kNN | 38.5 | 51.7 | **86.2 ✗** | **125.9 ✗** | 0.1 / 0.3 |
| SVR | 35.4 | 44.0 | 54.8 | 55.5 | 2.2 / 14.4 |
| RandomForest | 33.5 | 40.7 | 40.9 | 55.5 | 31.6 / 57.6 |
| **XGBoost** | **30.0** | **37.8** | **39.5** | **50.3** | 4.8 / 4.4 |
| MLP | 34.4 | 53.6 | 67.0 | **143.1 ✗** | 5.6 / 22.7 |
| **GIN** (`train.py`, 그래프, 80/10/10)‡ | random: **36.5** (RMSE 50.9, R² 0.71) | scaffold: **41.8** (RMSE 54.5, R² 0.44; [추가] `--split scaffold`, 45 epochs에 조기 종료) | – | – | 240 (145 epochs) / 345 (45) |
| 기준선 (train 평균으로 예측) | 72.5 | 76.0 | 72.5 | 76.0 | – |
| 기준선 (GIN 분할, test 500)‡ | 75.2 | 79.7 | | | |

굵게 = 열마다 최저. **✗ = 기준선보다 나쁨** (24쪽 "지는 케이스" — 숨기지 않음). † 시간은 random 분할 기준 (기술자 / Morgan), 컨테이너 CPU 2코어; 사용자 PC와 다를 수 있음. ‡ GIN의 test 500개는 노트북 test 1,000개와 다른 집합 — 표 2에서 같은 집합으로 비교.

RMSE, R²까지 (random, 기술자): XGBoost 41.4 / 0.80, RF 45.3 / 0.76, MLP 46.5 / 0.75, Ridge 46.9 / 0.74, SVR 50.2 / 0.70, kNN 52.9 / 0.67. scaffold 기술자: XGBoost 50.9 / 0.55, RF 54.2 / 0.49, SVR 59.9 / 0.38, Ridge 60.2 / 0.37, kNN 69.6 / 0.16, MLP 78.4 / −0.07.

## 표 2 — [추가] 같은 분할에서 GIN과 고전 모델 비교 (`../results/fair_compare_gin_split.csv`)

`train.py`의 80/10/10 random 분할(seed 42)을 재현해, 노트북과 같은 설정의 6모델을 **같은 train 4,000개**로 학습하고 **같은 test 500개**로 평가 (스크립트 `fair_compare.py`, GIN은 validate 500개를 조기 종료에만 사용).

| 순위 | 모델 + 표현 | test MAE [K] | RMSE | R² |
|---|---|---|---|---|
| 1 | XGBoost + 기술자 | **32.7** | 44.3 | 0.78 |
| 2 | RandomForest + 기술자 | 35.7 | 47.5 | 0.75 |
| **3** | **GIN (그래프)** | **36.5** | 50.9 | 0.71 |
| 4 | MLP + 기술자 | 37.1 | 51.7 | 0.70 |
| 5 | Ridge + 기술자 | 38.5 | 51.1 | 0.71 |
| 6 | SVR + 기술자 | 38.6 | 52.9 | 0.69 |
| 7 | XGBoost + Morgan | 41.0 | 53.6 | 0.68 |
| 8 | kNN + 기술자 | 42.1 | 55.1 | 0.66 |
| 9 | RandomForest + Morgan | 42.2 | 55.4 | 0.66 |
| 10 | SVR + Morgan | 55.8 | 73.1 | 0.40 |
| 11 | Ridge + Morgan | 60.2 | 78.7 | 0.31 |
| 12 | MLP + Morgan | 68.4 | 89.6 | 0.10 |
| 13 | kNN + Morgan | 91.8 ✗ | 116.1 | −0.50 |
| – | 기준선 (train 평균 377.8 K) | 75.2 | | |

scaffold 80/10/10도 같은 방법으로 (`fair_compare_gin_split_scaffold.csv`, train 평균 365.9 K, 기준선 79.7 K): **XGBoost + 기술자 36.4** < RF + 기술자 40.5 < **GIN 41.8** < SVR + 기술자 43.3 < Ridge + 기술자 44.2 < XGBoost + Morgan 48.1 < MLP + 기술자 48.3 < kNN + 기술자 51.4 < RF + Morgan 52.0 < SVR + Morgan 55.3 < Ridge + Morgan 99.1 ✗ < kNN + Morgan 132.3 ✗ < MLP + Morgan 149.0 ✗. GIN은 scaffold에서도 **3위**, 최고 고전 모델과의 차이 5.4 K.

→ **49쪽 4단계 답**: 같은 test 500개에서 **최고 고전 모델 XGBoost + 기술자 32.7 K vs GIN 36.5 K** — GIN이 **3.8 K(12 %) 뒤진다.** 다만 GIN은 Morgan 지문 기반 6모델은 모두 이겼고(최고 41.0 K), 기술자 기반에서도 RF 다음 3위. 19쪽 표 "GNN: 데이터 1만 개 이상에서 유리"와 맞는 결과 — 5,000개는 경계 아래.

## 읽는 법 네 가지 (38쪽) — mp 결과에서

1. **기준선 대비**: 기술자 모델은 random에서 기준선의 **41–53 %**, scaffold에서 **50–71 %**. Morgan 지문의 Ridge·kNN·MLP는 scaffold에서 **기준선 이하**(96–143 K; kNN은 random에서도 86 K). 평균만 못한 칸이 24칸 중 **4칸**.
2. **random vs scaffold**: 12쌍 **모두 scaffold > random**. 차이가 작은 쪽: XGBoost·RF + 기술자 (+7.2 ~ 7.8 K), SVR + Morgan (+0.7 K, 둘 다 평평하게 나쁨). 큰 쪽: MLP + Morgan (+76 K), kNN + Morgan (+40 K), Ridge + Morgan (+35 K) → 노트북 8절 문장대로 "**본 적 있는 뼈대를 외운**" 정도가 큰 모델들.
3. **표현 vs 모델**: 같은 모델에서 기술자 → 지문으로 바꾸면 MAE가 1.2–2.2배 (XGBoost 30.0 → 39.5). 같은 표현에서 모델을 바꾸는 차이(기술자 random 30.0 ~ 38.5)보다 **표현의 차이가 크다**. mp는 부분구조 유무보다 분자 전체의 극성·크기·강직성 요약값이 맞는 물성.
4. **GIN**: 기준선 75.2 K → 36.5 K (0.49). 고전 최고(0.44, 같은 test) 뒤. 학습 240 s(145 epochs) vs XGBoost 5 s. scaffold([추가])에서는 41.8 K (기준선 79.7의 0.52) — validate가 희귀 골격이라 조기 종료가 45 epoch에 일찍 걸렸다 (`mp_GNN-GIN_mean_scaffold_curve.png`: validate MAE가 44–52 K 사이에서 요동). 2주차에 patience를 늘리거나 validate를 random으로 두는 것이 공정.

## 발표 문장 (S4)

> "random 분할에서 최저는 **XGBoost + 기술자 MAE 30.0 K**, scaffold에서도 **XGBoost + 기술자 37.8 K**, 기준선은 72.5 / 76.0 K입니다.
> 12쌍 모두 scaffold가 random보다 나쁘고, Morgan 지문의 Ridge·kNN·MLP는 scaffold에서 기준선보다도 나쁩니다(96–143 K).
> GIN은 같은 test 500개에서 **36.5 K**로, 최고 고전 모델 32.7 K보다 **3.8 K 높습니다**. 지문 모델 6종은 모두 이겼습니다."
