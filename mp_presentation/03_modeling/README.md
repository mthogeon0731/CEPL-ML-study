# 03 임베딩·모델링 — 왜 이 모델들을 썼고, 각각 어떤 특징인가

11쪽 단계: **모델** (Ridge … XGBoost, `BGNN (GIN)`) · 발표 1:20–2:00 · 슬라이드 S3 (+ S7 코드 요약)

## ① 발표에서 답할 질문

1. "임베딩"이란 — 분자를 숫자 벡터로 바꾸는 것. **사람이 정한 특징**(기술자·지문)과 **모델이 배우는 특징**(GNN의 h_G, 21쪽) 둘 다 임베딩이고, 15쪽 그림의 왼쪽 → 오른쪽이 그 순서 (필요한 데이터 수 증가).
2. 왜 모델 6종을 먼저 돌리나 — 19쪽 첫 줄: **단순한 모델부터 → 기준 점수 → 복잡한 모델이 그 점수를 넘는지 확인**. 하이퍼파라미터 튜닝 없음 (37쪽).
3. 각 모델의 한 줄 직관과 "잘 되는 때 / 지는 때" (19쪽 표) → **mp에 적용하면?** (아래 표)
4. GIN은 무엇이 다른가 — 21쪽: 층 하나 = 이웃 정보를 모아 자기 벡터 갱신, sum 집계 → 이웃 개수 구별, WL test 수준 표현력; readout = mean pool → 분자 크기 정보는 원자 특징에서.
5. (한 줄만) BGIN — 가중치를 분포로 학습해 μ ± σ (22쪽), 3주차 주제. 이번 발표는 GIN까지.

## ② 슬라이드 근거

| 쪽 | 내용 |
|---|---|
| 15 | 표현별 맞는 모델: 기술자 → Ridge, RF, XGBoost, SVR / 지문 → kNN, RF, XGBoost, MLP / 그래프 → GCN, GIN, BGIN |
| 19 | 모델 8종 표 (한 줄 직관, 잘 되는 때, 지는 때) |
| 20 | 학습 범위 밖: Ridge 추세 유지, kNN·RF 평평(외삽 불가), SVR·MLP 임의 — 큰 분자·새 작용기에서 오차 |
| 21 | GIN 식 h_v^(k) = MLP((1+ε)h_v^(k−1) + Σ h_u^(k−1)), h_G = READOUT → 선형층; **연구실 설정: 5층, hidden 128, 지문 64, AdamW lr 0.001, global mean pool, dropout GNN 0.3, batch 256 (N < 10만)** |
| 22 | BGIN: 가중치 분포, μ·σ, warm start = GNN 가중치로 시작 |
| 9 | 학습 = MSE, 조기 종료·보고 = MAE (실습 GNN 코드); w ← w − η dL/dw |
| 27–30 | 발전사: 규칙 → 고전 ML → 딥러닝 → 그래프; 연구실 위치 = BGIN, 다음 후보 KA-GNN — 발표에서는 한 문장 |

## ③ 만들 산출물

| 파일 | 내용 |
|---|---|
| `model_settings.md` | 모델별 설정표 (아래 양식) — 노트북·`train.py`에서 **실제 값**을 읽어 채움 |
| `code_summary.md` | 과제 3단계: `feature.py`, `datamanager.py`, `models.py`, `train.py` 각 한 줄 (S7) |

### `model_settings.md` 양식 — 19쪽 표를 mp 관점으로 (설정값은 노트북 5절 `get_models()`·`train.py`에서 읽은 **실제값**)

| 모델 | 한 줄 직관 (19쪽) | 입력 표현 | 주요 설정 (코드 실제값) | mp에서 기대 (가설) | 결과와 맞았나 (4절 후 채움) |
|---|---|---|---|---|---|
| Ridge | 특징의 가중합, L2로 계수 억제 | 기술자 / 지문 | `StandardScaler` + `Ridge(alpha=1.0)` | 가법적 기여만 → 기준선보다 조금 나은 정도 | |
| kNN | 가장 비슷한 분자 k개의 평균 | 기술자 / 지문 | `StandardScaler` + `KNeighborsRegressor(n_neighbors=5, weights='distance')` (유클리드; 17쪽 Tanimoto는 아님) | 비슷한 분자 많은 영역 강함, 새 골격 약함 | |
| SVR | 커널로 비선형, ε 밖 오차만 벌점 | 기술자 / 지문 | `StandardScaler` + `SVR(C=10, epsilon=0.1)` (RBF 기본) | N=5,000이면 가능; 전체 19,945면 느림 (19쪽) | |
| Random forest | 깊은 나무 여러 개의 평균 | 기술자 / 지문 | `RandomForestRegressor(n_estimators=300)` | 튜닝 없이 안정 | |
| XGBoost | 앞 나무의 오차를 다음 나무가 보정 | 기술자 / 지문 | `XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.5)` | 표 형식 특징, 수천 개 이상 → 유력 | |
| MLP | 층을 쌓은 비선형 함수 | 기술자 / 지문 | `StandardScaler` + `MLPRegressor((256, 128), alpha=1e-3, early_stopping=True, max_iter=300)` | 데이터 5,000이면 과적합 주의 (early_stopping이 validation 10 %로 막음) | |
| GIN (`train.py`) | 이웃 원자 정보를 층마다 전달 (21쪽 식) | 그래프 (원자 147; 결합 16은 **GIN이 안 씀**) | 5층, hidden 128, 원자 벡터 64, `global_mean_pool`, Linear(64→1); AdamW lr 1e-3, ReduceLROnPlateau(10, ×0.8), 기울기 ±10, dropout 0.3, **batch 64** (N=5,000 < 1만; `--max_n 0`이면 256), epochs ≤ 300, 조기 종료 = validate MAE 30 epoch 정체 → 최선 가중치 복원, seed 42 | 데이터 5,000은 GNN엔 적은 편 (19쪽 "1만 개 이상") → 고전 모델을 못 넘을 수 있음; 전체로 돌리면 달라질지 | |

> 21쪽의 연구실 설정(5층, hidden 128, 지문 64, AdamW 0.001, mean pool, dropout 0.3)과 코드가 같다. 다른 점은 **batch 64**(표본 5,000 기준)와 **Scaler를 train으로만 fit**하는 것 (14쪽 누수 예를 코드가 고침). 2주차 옵션: `--conv GINE`(결합 특징 사용), `--pooling sum`(GIN 논문 readout), `--split scaffold`, `--bayes`(BGNN, 22쪽).

> "기대" 열은 **발표 전에 가설로 적어 두고**, 4절 결과가 나온 뒤 "맞았나" 열을 채운다. 가설과 결과가 다른 칸이 발표에서 가장 할 말이 많은 칸이다.

### `code_summary.md` 양식 (49쪽 3단계, 47쪽 표를 자기 말로)

| 파일 | 하는 일 한 줄 | 11쪽 단계 | 발표 S7 |
|---|---|---|---|
| `feature.py` | RDKit Mol → (x 원자×147, edge_index 2×2·결합, edge_attr 2·결합×16); 원자 147 = 원소·전하·혼성·H수·방향족·원자가·연결수·고리 + 작용기 SMARTS ~110 | 표현 | |
| `datamanager.py` | `data/<prop>.csv` → (표본 max_n) → PyG Data 목록 → 80:10:10 random/scaffold 분할 → y 표준화 `Scaler` | 정제·분할 | |
| `models.py` | `GraphRegressor`: GIN 5층(hidden 128, dropout) → 원자 벡터 64 → mean/sum pool → Linear(64→1); `bayes=True`면 마지막 층을 확률 가중치로 (BGNN) | 모델 | |
| `train.py` | 학습 루프 (scaled MSE, AdamW) + validate MAE 조기 종료 → `results/summary.csv` 한 줄 + parity·학습곡선 + `_test_pred.csv` | 학습·평가 | |

## ④ 작업 순서

1. 위 표의 설정값은 코드에서 이미 옮겨 적었다 → `model_settings.md`로 복사하고 "기대" 열을 **발표 전에** 자기 말로 다시 쓴다.
2. `train.py` 실행 후 터미널의 `[model] … parameters` 줄(파라미터 수)과 마지막 epoch 수를 GIN 행에 추가한다 (7쪽: 직선 θ 2개 vs GNN θ 10만 개 이상 — 실제 수로 말하면 좋다).
3. `code_summary.md` 4줄은 위 양식을 **자기 말로** 줄인다. 각 줄에 입력 → 출력이 들어가면 한 줄로 충분하다.
4. [추가] 발표에서 "왜 GIN인가"를 묻는 질문에 대비: 21쪽 그림 (이웃 {A, B} vs {A, A, B, B}: mean·max 같고 sum만 다름) 한 장.

## ⑤ 자가점검

- [ ] 15쪽 그림의 세 표현과 19쪽 표의 모델을 짝지을 수 있다
- [ ] GIN 식의 각 항(ε, Σ, MLP, READOUT)이 코드의 어느 줄인지 말할 수 있다 (`models.py`)
- [ ] 학습 손실(MSE)과 보고 지표(MAE)가 왜 다른지 말할 수 있다 (9쪽)
- [ ] "튜닝하지 않았다"를 약점이 아니라 **공정한 기준선**으로 설명할 수 있다 (37, 49쪽 규칙)
