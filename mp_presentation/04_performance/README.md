# 04 성능 — 좋은가 나쁜가, 과소·과대적합이 있는가

11쪽 단계: **평가** (`parity`, `MAE`) · 발표 2:00–3:20 · 슬라이드 S4 (표 1장) + S5 (그림 2장)

## ① 발표에서 답할 질문

1. **숫자** — test MAE [K]를 모델 6 × 표현 2 × 분할 2 = 24칸으로 (37쪽 양식). 열마다 최저 굵게.
2. **기준선 대비** — train 평균으로 예측한 MAE (23쪽). "모델이 기준선보다 얼마나 낮은가"가 "좋다/나쁘다"의 첫 기준. 기준선보다 나쁜 칸이 있으면 그대로 보여준다 (24쪽: Ridge + Morgan scaffold 1.55 > 기준선 1.41).
3. **random vs scaffold** — 24칸 모두 scaffold > random인지 (38쪽). 차이 크기 = "새 골격에서 얼마나 못 하는가".
4. **GNN vs 최고 고전 모델** — 49쪽 4단계: 1의 최고 모델 MAE vs 2의 GIN MAE. 어느 분할 기준인지 명시.
5. **진단** — parity plot(끝 영역 퍼짐, train vs test), 학습곡선(train/val loss vs epoch)으로 과소·과적합·불안정 중 무엇인지.

## ② 슬라이드 근거

| 쪽 | 내용 |
|---|---|
| 9 | MSE(학습) vs MAE(보고): MAE 단위 = 물성 단위 → mp는 **K** |
| 10 | 과소적합 = train·validate 둘 다 큼 / 과적합 = train 작고 validate 큼 / 조기 종료 = validate 오차가 다시 커지기 전 멈춤 |
| 14 | 점수만 보고 판단 금지 → 어떤 분할의 점수인지 함께 |
| 23 | MAE, RMSE 식; RMSE ≥ MAE, 차이 큼 = 큰 오차 몇 개; 기준선 = train 평균 예측; parity 대각선·끝 영역; learning curve: 데이터 2배 → 오차 계속 줄면 데이터 추가가 답; parity 색 = 분자 종류 |
| 24 | 지는 케이스: logPvap train 대각선 vs test 끝 퍼짐 → 데이터 적은 끝 영역 과적합 / pKa BGNN train·test loss 같이 커짐 → **과적합이 아니라 학습 불안정** / Ridge+Morgan scaffold > 기준선 |
| 25–26 | 2024 결과: 물성마다 퍼짐이 다름 → 측정 잡음·출처 섞임부터 의심; 데이터 수 ≠ 정확도 순서 |
| 37–38 | logP 결과표 양식과 읽는 법 네 가지 |

## ③ 만들 산출물

| 파일 | 내용 | 출처 |
|---|---|---|
| `results_table.md` | 37쪽 양식 표 (test MAE [K], 학습 시간 [s], 기준선 행) — S4 | 노트북 `results_mp.csv` (열: split, feature, model, MAE, RMSE, R2, sec) + 6절 결과표 셀(`table`, 기준선 출력) → `../results/` |
| `mp_GNN-GIN_mean_random_parity.png`, `mp_GNN-GIN_mean_random_curve.png` | `train.py` 출력 그림 2장 — S5 (curve = a: train loss(scaled MSE), b: validate MAE) | `261105_Week1_Practice/results/` 복사 |
| `diagnosis.md` | 진단표 (아래) + 발표 문장 | 그림 + `results/summary.csv` 읽고 작성 |

### `results_table.md` 양식 (37쪽)

mp, **표본 5,000 / 전체 19,945** (`MAX_N = 5000`, seed 42), 노트북 분할 = random 80/20 · scaffold 80/20, 값 = **test MAE [K]** (낮을수록 좋음), 하이퍼파라미터 튜닝 없음
(GIN 행: `train.py` 기본 = 같은 5,000개 표본이지만 **80/10/10** 분할 → test 집합이 다름. `results/summary.csv`의 MAE·baseline_MAE·epochs·sec를 적고 "분할이 다름"을 각주로)

| 모델 | Morgan random | Morgan scaffold | 기술자 random | 기술자 scaffold | 시간 [s] |
|---|---|---|---|---|---|
| Ridge | | | | | |
| kNN | | | | | |
| SVR | | | | | |
| RandomForest | | | | | |
| XGBoost | | | | | |
| MLP | | | | | |
| **GIN (train.py)** — 그래프, 80/10/10 | random: | (`--split scaffold`로 한 번 더 돌리면) | – | – | |
| 기준선 (train 평균) | 노트북 random: | 노트북 scaffold: | (같은 값) | (같은 값) | – |

### `diagnosis.md` 진단표 (10, 24쪽)

| 보는 것 | 관찰 (실제 그림에서) | 판정 | 근거 쪽 |
|---|---|---|---|
| 학습곡선 a (train loss, scaled MSE) vs b (validate MAE) | a 계속 내려가는데 b가 멈춤·오름 / 둘 다 높게 머무름 / 둘 다 커짐 | 과적합 / 과소적합 / 학습 불안정 (주의: a는 표준화된 MSE, b는 K 단위 MAE — 축이 다름) | 9, 10, 24 |
| 조기 종료 시점 | validate MAE 최소 epoch = (`summary.csv`의 epochs − 30 근처) | 그 뒤 30 epoch은 개선 없음 → 최선 가중치로 복원됨 | 10 |
| parity train vs test | test 끝 영역(낮은·높은 mp) 퍼짐 | 끝 영역 과적합 또는 데이터 부족 | 24 |
| RMSE − MAE | 차이 큼 → 큰 오차 몇 개 | 5절 후분석으로 | 23 |
| random − scaffold | 차이 = 새 골격 일반화 비용 | | 14, 38 |
| 기준선 대비 | 기준선보다 나쁜 칸 | 평균만 못함 → 표현·모델 부적합 | 24 |

### 발표 문장 틀 (수치는 모두 표에서)

> "random 분할에서 최저는 ( ) + ( ) MAE ( ) K, scaffold에서는 ( ) + ( ) MAE ( ) K, 기준선은 ( ) K입니다.
> 24칸 중 ( )칸이 scaffold > random이고, GIN은 random 기준 ( ) K로 최고 고전 모델보다 (높/낮)습니다.
> 학습곡선에서 val loss가 epoch ( )에서 최소였고 그 뒤 ( ) → (과적합/안정). parity에서는 ( ) K 이하 영역이 퍼집니다 — 1절에서 본 대로 그 영역은 데이터가 ( )개뿐입니다."

## ④ 작업 순서

1. 노트북 결과표를 `../results/`에 저장(csv/png) → `results_table.md`로 옮겨 적고 열마다 최저 굵게.
2. `train.py` 결과 그림 2장을 `../results/`에 복사하고 이름을 위 표의 파일명으로 맞춘다.
3. 진단표를 채운다. **판정은 그림의 특정 위치를 가리키며** ("epoch 40 이후 간격이 벌어짐") 말한다 — 그래프 해석 순서: x축 → y축 → 기울기 → 특정 영역 → 임계점(최솟값·벌어지는 시점) → 모델과 연결.
4. [추가] mp 문헌 오차 수준을 언급하고 싶으면 출처를 먼저 확인한다. 슬라이드에는 mp 참고 MAE가 없다. 측정 잡음 때문에 bp·logP보다 큰 것이 보통이지만, 수치를 말하려면 논문을 인용해야 한다.

## ⑤ 자가점검

- [ ] MAE 단위를 K로 말하고, 분할 종류를 항상 같이 말한다
- [ ] 기준선 MAE를 알고, 기준선보다 나쁜 칸을 숨기지 않는다 (24쪽)
- [ ] 과적합과 학습 불안정을 학습곡선 모양으로 구분한다 (24쪽 pKa 예)
- [ ] "데이터를 더 모으면 나아질까"에 learning curve 논리로 답한다 (23쪽)
