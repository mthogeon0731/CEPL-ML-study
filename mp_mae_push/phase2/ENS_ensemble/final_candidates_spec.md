# 최종 후보 test 단계 실행 명세 (앙상블 트랙 2-C, 2026-11-21)

모든 수치는 validation 기준이다(random n 1,995 / 중복 제거 1,922, scaffold n 1,994 / 1,986, 단위 K). test는 열지 않았다. 표 형식 명세는 [final_candidates_spec.csv]({{artifact:d703797c-eca0-4aee-baa8-25411395e38e}})에 있다(멤버별 한 줄).

## 공통 규칙

- **주 결과는 train 전용 학습이다.** random 15,955행, scaffold 15,956행으로, 원 기준선과 같은 train이다. 보조 결과로 train+validation 재학습을 낸다. 이때 나무 수는 `train_candidate.py` 규칙(best_iter × n_fit / n_inner)으로 늘리고, 앙상블 가중치는 validation 값으로 고정한다.
- 재보정은 하지 않는다(H11: 선형 n.s., isotonic 악화).
- 스레드: XGBoost n_jobs 8, LightGBM num_threads 6(T1b 기록값과 같게), GIN 1스레드.
- test 예측은 후보마다 한 번만 만든다. 앙상블은 각 멤버의 test 예측(seed 평균)을 만든 뒤 아래 가중치로 합친다. 가중치 합이 반올림 때문에 1에서 벗어나면 합으로 나눈다.
- **GIN 재학습 점검(필수).** GIN은 가중치가 저장되지 않아 같은 seed(0,1,2)로 다시 학습해야 한다. 재학습한 모델로 validation도 다시 예측하고 저장본({{artifact:e00f6fc0-adac-4a79-bd4b-d7e739709548}}, {{artifact:4482fc87-4c20-4bfe-b75d-c55687fae794}})과 대조한다. 기준은 seed 평균 예측의 validation MAE 차이 ≤ 0.1 K, 상관 ≥ 0.995다. 벗어나면 가중치는 바꾸지 않고 그 사실을 보고한다. 이 경우 GIN이 빠진 대체안 `lgb_r1_F0F3F5` 단일 결과도 함께 낸다.
- 목표 C: 점수는 `metaA_trainOOF_self`(LightGBM |오차| 예측; 모델 파일 {{artifact:e550392e-113a-4304-b4f0-fb2b3f9af85d}} random, {{artifact:9992456f-bda5-4654-91cc-45a56c4296e8}} scaffold)다. 입력은 ens_sd, knn_gap, nn1, n07, nn5mean, n_meas, pred, heavy이며 정의는 `s4_risk.py`에 있다.
  - test에서 ens_sd를 만들려면 OOF 멤버 6개(random: baseline, lgb_r1_F0, xgb_r1_F0, lgb_r1_F0F3F5, h06, M2 / scaffold: M2를 뺀 5개)의 test 예측이 있어야 한다. knn_gap에는 O2 kNN test 예측이 필요하다.
  - 추가 재적합 비용은 lgb_r1_F0 약 0.59, xgb_r1_F0 약 0.81, M2 약 0.08 core-h다(ledger 실측 합). 나머지 멤버는 후보 학습과 겹친다.
  - test 행의 nn1·n07은 **train만** 참조해 계산한다(validation 행은 참조하지 않음).
  - test에는 고정한 τ를 한 번만 적용한다(커버리지와 MAE 보고). scaffold는 validation에서 MAE ≤ 10 K를 만족하는 커버리지가 없어 τ를 고정하지 못했다. 사전 예측은 "목표 C scaffold 미달"이다.

## A. 가장 정확 — `wavg_mse|noO2` (단체 가중 평균, MSE 목적, O2 제외; T3 전 멤버 풀)

| 분할 | 멤버(가중치) | 특징 | seed |
|---|---|---|---|
| random | h06_yalk_F0F3F5 0.4696, gin_hybrid_seedavg 0.3877, lgb_r1_F0F5 0.1232, xgb600_F0F1O3 0.0196 | F0+F3+F5 / graph147+F0 / F0+F5 / F0+F1+O3 | 0,1,2 (모두) |
| scaffold | gin_hybrid_seedavg 0.3722, xgb600_F0_patent1x_scaf 0.3178, h06_yalk_F0F3F5 0.3100 | graph147+F0 / F0(+특허 15,956행) / F0+F3+F5 | GIN·h06 0,1,2; 특허 42,43,44 |

- 레시피: h06은 `s2_yalk.py --stage refit`이다(a random 65 / scaffold 40, c 2.435, σ3d_max_log, τ clip0, S_floor 10, 나무 random 4,588 / scaffold 3,304). **test 예측 경로를 먼저 추가해야 한다**(T1b README). lgb_F0F5는 `train_candidate.py --cand lgb_F0F5`, F0F1O3은 T1 `s2_features.py`(BASE_XGB, colsample 0.5), 특허 1배는 T5 `step5_curve.py`(add1x), GIN은 T2 `train_hybrid.py`다.
- validation(정직한 교차 적합, 5-fold × 10회 평균): random 24.889 K(중복 제거 25.212), scaffold 34.017 K(34.002).
- 목표 C τ: random 14.675 K. validation 커버리지는 7.22 %(144/1,995), 그 부분집합 MAE는 9.98 K [8.55, 11.63]이다. 교차 검증 임계값 추정은 커버리지 7.23 %, MAE 10.06 K이다.
- 예상 비용: 최종 재적합 약 6.2 core-h(두 분할). 이 중 GIN이 4.1 core-h다. 추론은 625.7 s/1,000분자이며 F3 배좌 생성이 약 588 s를 차지한다.

## B. 가장 경제적 — `fixed_mean|best+GIN` (LGB F0+F3+F5와 GIN을 0.5 : 0.5로 단순 평균, 적합 모수 없음)

| 분할 | 멤버(가중치) | 특징 | seed |
|---|---|---|---|
| random·scaffold | lgb_r1_F0F3F5 0.5, gin_hybrid_seedavg 0.5 | F0+F3+F5 / graph147+F0 | 0,1,2 |

- 레시피: `train_candidate.py --cand lgb_F0F3F5`(lgb_r1: huber60, lr 0.01306, leaves 68, depth 12, min_leaf 12, bagging 0.826, feature_fraction 0.600, λ2 0.0214, λ1 0.0301, 나무 random 3,688 / scaffold 2,978)와 T2 `train_hybrid.py`다.
- validation: random 24.917 K(중복 제거 25.193), scaffold 34.161 K(34.140). 가장 정확(A) 대비 random +0.03 [−0.09, +0.16], scaffold +0.14 [−0.10, +0.38]로, 둘 다 구분되지 않는다.
- 목표 C τ: random 14.614 K. 커버리지 7.12 %(142), MAE 9.99 K [8.56, 11.52]. 교차 검증 추정은 6.74 %, 10.14 K다.
- 예상 비용: 최종 재적합 약 5.0 core-h, 추론 621.9 s/1,000분자.

## C. 균형안 — `xgb600_F0F5_M2` (기준선 XGB에 LLM 규칙 특징 F5를 더한 단일 모델)

| 분할 | 멤버 | 특징 | seed |
|---|---|---|---|
| random·scaffold | xgb600_F0F5_M2 1.0 | F0+F5(F5_rules.npy 101열) | 42,0,1 |

- 레시피: mpcommon.BASE_XGB(600그루, lr 0.05, depth 6, subsample 0.8, colsample 0.5), n_jobs 8. LLM 호출은 필요 없다(규칙은 고정 RDKit 코드).
- validation: random 26.499 K(26.765), scaffold 35.477 K(35.484). 기준선 대비 random −1.11 [−1.50, −0.71], scaffold −0.87 [−1.36, −0.37]이다.
- 목표 C τ: random 13.337 K. 커버리지 5.46 %(109), MAE 9.88 K [8.27, 11.62]. 교차 검증 추정은 4.94 %, 10.65 K로, 반복의 10 %만 ≤ 10 K였다.
- 예상 비용: 최종 재적합 약 0.08 core-h, 추론 26.2 s/1,000분자. GIN과 3D 배좌가 필요 없다.

## 참고: 엄격 규칙 승자(채택하지 않음)

`fixed_wavg_mse|F0F5+GIN+cbMLM@T3`는 PREREG의 '가장 경제적' 정의를 글자 그대로 적용하면 뽑힌다. 총비용은 1.66 USD로 B보다 0.18 USD 싸다. 그러나 이 후보는 T3의 탐색적 관찰을 보고 이 트랙이 사후에 만든 부분집합이다. 또 scaffold에서 A 대비 CI 하한이 −0.00이고(부트스트랩 seed 20개 중 85 %에서만 통과), 채택하지 않았다. 자세한 이유는 README 5절에 있다.
