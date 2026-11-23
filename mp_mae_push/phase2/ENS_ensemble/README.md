# 앙상블 트랙 (2-C, ENS) — 보고

작성 2026-11-21. Bradley Open MP 녹는점(K)을 다룬다. 모든 수치는 **validation**에서 나왔다(random n 1,995 / 중복 제거 1,922, scaffold n 1,994 / 1,986). 일부는 train 5-fold OOF(random 15,955 / scaffold 15,956)를 쓴다. **test는 어떤 형태로도 열지 않았다**(라벨·예측·특징 통계 모두; ledger의 test_* 열은 읽을 때 버렸다).

- Δ는 MAE(후보) − MAE(비교 대상)다. 음수면 개선이다.
- CI는 짝지은 부트스트랩 2,000회(seed 0)의 95 % 구간이다.
- 앙상블 수치는 validation 안 5-fold 교차 적합을 10회 반복(fold seed 0–9)한 반복 평균 예측, 곧 '정직한 예측'에서 계산했다.
- 사용 CPU는 1.66 core-h다(측정 1.41 + 미측정 추정 0.25). 배정은 5 core-h(PREREG 표는 4)였고, API·GPU는 쓰지 않았다.

## 1. 결론

1. **앙상블은 두 분할 모두에서 최선 단일보다 유의하게 낫다. 이득의 주원천은 GIN 하이브리드다.** 가장 정확한 후보 A(`wavg_mse|noO2`, 단체 가중 평균·MSE 목적)의 validation MAE는 random 24.889 K, scaffold 34.017 K다.

   | 비교 대상 | random | scaffold |
   |---|---|---|
   | 최선 단일 lgb_r1_F0F3F5 | −0.47 [−0.79, −0.15] | −1.22 [−1.73, −0.71] |
   | 원 기준선 | −2.72 [−3.19, −2.25] | −2.33 [−2.89, −1.78] |

   중복 제거 기준에서도 같다(최선 단일 대비 −0.56 / −1.22, 기준선 대비 −2.72 / −2.35). GIN과 최선 단일의 오차 상관은 random 0.878, scaffold 0.891로 트리 모델끼리(0.93–0.99)보다 낮다. GIN 가중치는 단체 가중·전진 선택 앙상블에서 0.37–0.48, Ridge 스태킹에서 0.20–0.39였다.
2. **O2 Tanimoto kNN은 단독으로 약하고 앙상블에 기여하지 않는다.** 단독 MAE는 random 38.91 K, scaffold 56.13 K다. 같은 방법에서 O2를 넣고 뺀 차이는 random 전체 −0.01 [−0.07, +0.05], 중복 제거 −0.00, scaffold +0.01이다. 겹치는 분자(random n 73)에서만 −0.18 [−0.38, +0.01](n.s.)이다. 작은 이득조차 **근접 중복에서만** 나온다(표시: '근접 중복 활용').
3. **재보정(H11)은 이득이 없다.** 선형 재보정은 ±0.1 K 이내(n.s.)였다. isotonic은 random에서 유의하게 나빠졌다(최선 단일 +0.44, 앙상블 +0.28 K). 1단계 예상대로이며 재보정은 채택하지 않는다.
4. **목표 C(10 K 커버리지)**
   - random: 메타모델 A로 τ를 고정하면 커버리지는 약 7 %(A 144/1,995), 부분집합 MAE는 9.98 K [8.55, 11.63]이다. 교차 검증 임계값 추정 MAE가 10.06 K이고 반복의 50 %만 ≤ 10 K다. 따라서 **경계선**이다.
   - scaffold: n ≥ 30에서 MAE ≤ 10 K를 만족하는 커버리지가 없다(상위 5 %의 MAE가 23.9 K).
   - PREREG 7절 부분집합 중 validation MAE ≤ 10 K인 것은 없다. 가장 낮은 것은 random 정제판(DPG) 18.6–18.9 K, 최근접 Tanimoto ≥ 0.9인 분자(n 176) 14.5–14.9 K다.
5. **T3(사전학습 표현) 멤버 6개는 앙상블을 개선하지 못했다.** T3 포함 풀의 같은 방법은 random 24.918, scaffold 34.040 K로 T3 전 풀(24.889 / 34.017)보다 약간 나쁘다. 전진 선택은 random에서 T3를 한 번도 고르지 않았고 scaffold fwd4에서 cbMLM에 0.022만 주었다.
   - T3가 보고한 탐색적 관찰(R217과 R217+ChemBERTa-MLM의 50:50)을 이 트랙의 교차 적합 틀에서 다시 재면 같은 head R217(seed 0–2) 대비 scaffold −0.58 [−0.87, −0.29], random +0.05 [−0.21, +0.31]이다. 약한 XGB 둘의 다양성 이득으로는 재현된다.
   - 그러나 강한 멤버(GIN·h06·특허)가 있으면 사라진다(wavg_mse 가중치 ≤ 0.07).
6. **최종 후보 3개.** 실행 명세는 [final_candidates_spec.md]({{artifact:d9f4790d-69a0-4017-af69-c16c8e42bef9}}) / [.csv]({{artifact:d703797c-eca0-4aee-baa8-25411395e38e}})에 있다. 순위는 전체 179개 후보 중이다.

| 역할 | 후보 | random MAE (중복 제거) | scaffold MAE (중복 제거) | 순위 random 전체/중복 제거 | 순위 scaffold | 총비용 core-h / USD | 추론 s/1k |
|---|---|---|---|---|---|---|---|
| 가장 정확 | A `wavg_mse\|noO2` | 24.889 (25.212) | 34.017 (34.002) | 42 / 42 | 10 / 10 | 39.99 / 2.36 | 625.7 |
| 가장 경제적 | B `fixed_mean\|best+GIN` (LGB F0+F3+F5 + GIN 50:50) | 24.917 (25.193) | 34.161 (34.140) | 50 / **29** | 56 / **52** | 29.55 / 1.83 | 621.9 |
| 균형안 | C `xgb600_F0F5_M2` (XGB F0+F5 단일) | 26.499 (26.765) | 35.477 (35.484) | 142 / 142 | 163 / 163 | 1.17 / 0.40 | 26.2 |

순위는 상위 후보들이 0.1 K 안에 몰려 있어 크게 흔들린다(전체·중복 제거 순위가 다른 후보는 random 126/179, scaffold 54/179개).

## 2. O2 Tanimoto 이웃 모델 (1a)

- 지문은 Morgan 반경 2, 2,048비트다(train·validation 행만 계산). 이웃은 train 행만 참조한다.
- k ∈ {1, 3, 5, 10, 20}와 가중 방식 ∈ {균등, sim, sim², sim⁴}를 train 안 5-fold CV로 골랐다. fold 규칙은 T1·T1b와 같다(random KFold shuffle 42 / scaffold GroupKFold Murcko).
  - random: k 10, sim⁴. CV MAE 40.02, validation 38.91, 중복 제거 39.62 K.
  - scaffold: k 20, sim⁴. CV 47.54, validation 56.13, 중복 제거 56.13 K. **k 20은 격자 끝이다.**
  - 표: [O2_knn_cv_grid.csv]({{artifact:f3d8c433-df47-4556-82ad-e7bcb59dcc97}}), [O2_knn_chosen.csv]({{artifact:ad78ba16-f40d-427b-a9c7-dd82e1f07044}})
- 예측: validation은 [random]({{artifact:b8f77e8a-84ce-4b5c-bdd2-f4c53a41eeca}}) / [scaffold]({{artifact:f6a54901-d57c-4f48-8d60-6e15a4b9f0ab}}), train OOF(fold 밖 이웃)는 [random]({{artifact:3f5b5e93-4f0d-4fbb-85a2-8fe234ca05e1}}) / [scaffold]({{artifact:4dac3b2c-0280-4926-877c-2229234c2ac4}})다.
- 유사도 특징: nn1, n07, nn5mean이다. train 행은 자기 제외(_self)와 같은 fold 제외(_fold) 두 가지를 저장했다. 파일은 [random]({{artifact:e44cb875-5c47-4494-bb7c-1649ea7c1a17}}), [scaffold]({{artifact:340ad5d4-b35f-43c4-8c93-47e86ece3bb1}})다. nn1 중앙값은 다음과 같다.

  | 분할 | validation (≥ 0.7 비율) | train 자기 제외 | train fold 제외 |
  |---|---|---|---|
  | random | 0.629 (31.6 %) | 0.636 | 0.619 |
  | scaffold | 0.456 (7.8 %) | 0.667 | 0.500 |

  scaffold에서는 자기 제외 정의가 validation보다 이웃을 훨씬 가깝게 본다.
- 유사도 층별 결과: [O2_by_similarity_stratum.csv]({{artifact:8e604a25-53e4-4c47-a5f5-07a9708571f2}}).
  - random 겹침 분자(n 73)에서는 앙상블(O2 없음, 16.38 K)이 최선 단일(14.59 K)보다 나쁘다. GIN이 근접 중복에서 트리보다 못하기 때문이다.
  - 앙상블 이득은 비겹침 분자에서 나온다.

## 3. 앙상블 (1b–1d)

방법은 모두 비음수 가중이다.

- 단체 가중 평균: MAE 목적은 선형계획(HiGHS), 멤버 2개는 가중 중앙값 정확해(LP와 같은 해 확인), MSE 목적은 NNLS + 합 1 제약.
- 상위 k개 단순 평균: k는 2–5 또는 학습 fold 안에서 자동 선택.
- Ridge(positive, 절편, alpha는 학습 fold 안 5-fold CV).
- 전진 선택 축소 앙상블(2–4개): 선택까지 교차 적합 안에서 했다.
- 최종 가중치는 validation 전체로 한 번 적합했다: [ensemble_final_weights.csv]({{artifact:8ddffcc3-110d-4455-abb8-8eec0942d602}}).

전체 결과는 [ensemble_results.csv]({{artifact:68444def-e018-4780-9527-847de2b2bec6}})(T3 전 풀)와 [ensemble_results_T3.csv]({{artifact:136d0422-38c0-407b-9a0a-2e2da8753913}})(T3 포함 풀)에 있다. 주요 행은 다음과 같다(T3 전 풀; 반복 10회 MAE SD 0.01–0.16 K).

| 방법 | random MAE | Δ vs 최선 단일 (중복 제거) | scaffold MAE | Δ vs 최선 단일 (중복 제거) |
|---|---|---|---|---|
| wavg_mse \| noO2 (A) | 24.889 | −0.47 [−0.79, −0.15] (−0.56) | 34.017 | −1.22 [−1.73, −0.71] (−1.22) |
| wavg_mae \| all | 24.891 | −0.47 [−0.77, −0.16] (−0.55) | 34.057 | −1.18 [−1.66, −0.70] (−1.18) |
| fwd2_mae (random: LGB F0F3F5 0.61 + GIN 0.39; scaffold: h06 0.52 + GIN 0.48) | 24.874 | −0.49 [−0.78, −0.19] | 34.191 | −1.05 [−1.53, −0.58] |
| ridge_pos \| all | 25.137 | −0.22 [−0.55, +0.11] | 33.989 | −1.25 [−1.79, −0.74] |
| mean_all (12 / 14개) | 25.793 | +0.43 [+0.12, +0.74] | 34.904 | −0.33 [−0.77, +0.08] |
| train OOF 가중(OOF 멤버만) wavg_mae | 25.330 | −0.03 [−0.13, +0.08] | 35.062 | −0.18 [−0.37, +0.02] |

- train OOF 가중의 이득이 작은 이유는 GIN·특허 멤버에 OOF가 없기 때문이다. 같은 멤버 집합으로 validation 교차 적합을 해도 비슷했다(random 25.316, scaffold 34.998).
- 오차 상관 행렬: [random]({{artifact:1296013f-7171-4107-95af-37cb272d8397}}), [scaffold]({{artifact:01dbf8ed-444b-4bf1-af6a-da75593856ec}}), T3 포함 [random]({{artifact:1103ba44-0a9c-44bd-9eab-fdd0160957d8}}) / [scaffold]({{artifact:f921efab-cde9-4e7b-b89e-af943f5c4fa2}}).
- 축소 앙상블(사후 지정 고정 부분집합 16개 + T3 5개): [ensemble_fixed_subsets.csv]({{artifact:b7ee17e7-a0a1-4eb1-bf2f-33e8d7c1595e}}), [ensemble_fixed_subsets_T3.csv]({{artifact:8fef1767-f2dc-422a-8753-c1c38c9a4df7}}). 멤버 2–3개(GIN + LGB 계열)로 전체 가중 앙상블과 0.05–0.15 K 이내가 된다.
- test 재현 가능성(멤버별 레시피·특징·seed·재적합 core-h·LLM 필요 여부): [member_test_reproducibility.csv]({{artifact:1729317f-b07b-49fc-a1fd-010cf0e60cbc}}).
  - GIN은 가중치가 저장되지 않아 재학습이 필요하다(실측 4.1 core-h, 1스레드, 동일성 미확인).
  - T8 F5는 고정 RDKit 규칙 코드로, test 단계 LLM 호출이 필요 없다.
  - h06은 `s2_yalk.py`에 test 경로를 추가해야 한다.

## 4. 재보정 (H11)

[recal_results.csv]({{artifact:3280c826-ed45-4945-ab57-501bbe1e92c6}}), MP 구간별 [recal_by_mpbin.csv]({{artifact:31118091-3090-4923-8d3c-71f5f32057f2}}). validation 교차 적합(5-fold × 10) 기준 ΔMAE는 다음과 같다.

| 대상 | 분할 | 선형 | isotonic |
|---|---|---|---|
| 최선 단일 | random | −0.03 [−0.14, +0.07] | +0.44 [+0.18, +0.69] |
| 최선 단일 | scaffold | +0.08 [−0.01, +0.17] | +0.36 [+0.08, +0.64] |
| 앙상블 A | random | +0.01 [−0.12, +0.13] | +0.28 [+0.04, +0.52] |
| 앙상블 A | scaffold | −0.01 [−0.19, +0.18] | +0.15 [−0.18, +0.49] |

train OOF로 적합한 경우도 같은 결론이다. 선형 재보정은 끝 구간을 개선한다(random > 550 K −4.6 K, < 250 K −0.5 K). 그 대신 350–450 K(n 845) +0.9 K로 나빠져 전체는 상쇄된다.

## 5. 위험–커버리지 (목표 C, O5)

![위험–커버리지 곡선: 최선 앙상블, 두 분할]({{artifact:2db00b62-3822-465b-9b81-a7bc92cb9e9d}})

**메타모델 구성**

- 학습기는 LightGBM(l2, |오차| 대상, num_threads 8, seed 0)이다.
- 입력은 ens_sd(OOF 멤버 6/5개 예측의 SD), knn_gap(|최선 단일 − O2|), nn1·n07·nn5mean, 반복측정 수 n_meas, 예측값, 무거운 원자 수다.
- 방식 A(주): train OOF에서 최선 단일 OOF의 |오차|를 학습한 뒤 validation에 적용한다. 앙상블 대상에도 같은 점수를 대리로 쓴다.
- 방식 B: validation 교차 적합이다.

**결과**

- AURC(낮을수록 좋음)는 다음과 같다.

  | 분할 | 메타 A | 메타 B | ens_sd만 | nn1만 | 이상적 순서 |
  |---|---|---|---|---|---|
  | random | 18.18 | 19.54 | 20.51 | 20.98 | 9.35 |
  | scaffold | 29.46 | 28.61 | 31.38 | 31.00 | 13.32 |

  scaffold에서는 방식 B가 약간 낫다.
- 이득 기여(gain)가 큰 입력은 random에서 ens_sd, nn5mean, pred, nn1 순, scaffold에서 pred, nn5mean, ens_sd, nn1 순이다.
- 표: [risk_summary.csv]({{artifact:32451d13-e0d5-453b-b536-ad524e99cff1}}), 곡선 [risk_curve.csv]({{artifact:f8dd2719-4756-49b6-95fe-9ff80b4b01da}}), 중요도 [risk_importance.csv]({{artifact:16e734fb-075d-4b16-b83e-5c6a4ff78fc5}}).

**고정 임계값**

정의는 '누적 MAE ≤ 10 K인 최대 커버리지, 최소 30분자'이고 점수는 metaA_trainOOF_self다. 출처는 [risk_thresholds.csv]({{artifact:66b80427-86c7-436d-8ef0-71afd96fcf44}}), [_cands]({{artifact:91d1e4d0-c568-4258-8f20-d1fb28c57ae0}}), [_M2]({{artifact:021a8819-af5e-4f61-8a90-9a671ec9550c}})다.

| 점예측 | 분할 | τ [K] | 커버리지 (n) | 부분집합 MAE [행 부트스트랩 CI] | τ 고정 전체 부트스트랩: 커버리지 / MAE | 교차 검증 임계값: 커버리지 / MAE (≤ 10 K 반복 비율) |
|---|---|---|---|---|---|---|
| A 가장 정확 | random | 14.675 | 7.22 % (144) | 9.98 [8.55, 11.63] | [6.0, 8.3] % / [8.50, 11.58] | 7.23 % / 10.06 (50 %) |
| B 가장 경제적 | random | 14.614 | 7.12 % (142) | 9.99 [8.56, 11.52] | [6.0, 8.2] % / [8.55, 11.60] | 6.74 % / 10.14 (50 %) |
| C 균형안 | random | 13.337 | 5.46 % (109) | 9.88 [8.27, 11.62] | [4.4, 6.4] % / [8.35, 11.63] | 4.94 % / 10.65 (10 %) |
| 최선 단일 | random | 14.640 | 7.17 % (143) | 9.84 [8.32, 11.47] | [6.0, 8.3] % / [8.39, 11.45] | 7.10 % / 9.99 (50 %) |
| 모든 후보 | scaffold | 고정 불가 | 0 % | – | – | – |

PREREG 7절 부분집합(계열 표지, MP 구간[y와 예측 두 가지], 라벨 품질 1–5, nn1 층)의 validation MAE는 [risk_subsets.csv]({{artifact:cd7a8370-943a-444a-9f93-8089f5f08914}}), [_cands]({{artifact:965fffe6-cb82-4643-ba02-a47dab4b101f}}), [_M2]({{artifact:43be7dd7-8e4f-455c-8631-76200bf1e4c2}})에 있다. ≤ 10 K인 부분집합은 없다. 라벨 품질 층의 n은 1단계 표와 일치한다(random DPG 285 / 측정 1개 1,547; scaffold 165 / 1,683).

분자별 점수·특징 파일:

- 점수: [random A]({{artifact:15889167-d041-4e78-9e6f-cb66bd529b12}}), [scaffold A]({{artifact:13c13be6-2fd2-4daf-a010-d0e6383756ec}}), [random B]({{artifact:902ca4b2-ca82-47f7-869b-dadfe9747132}}), [scaffold B]({{artifact:70fcbfec-e52a-493b-a4c0-5ad8104966ee}}), [random C]({{artifact:c79bd06b-bed4-4c51-8d77-697575f16e2b}}), [scaffold C]({{artifact:646b6d62-6d01-4be2-8b2e-5b803ed90077}})
- 메타모델 입력: [random]({{artifact:5b9c7a9e-91e2-418c-8c01-53402c798909}}), [scaffold]({{artifact:a28aad29-6e78-48c3-bced-7f82986b0fda}})
- 메타모델 파일: [random]({{artifact:e550392e-113a-4304-b4f0-fb2b3f9af85d}}), [scaffold]({{artifact:9992456f-bda5-4654-91cc-45a56c4296e8}}). 재적합해 저장 점수와 최대 차이 0.0을 확인했다.

## 6. 비용·파레토·후보 선정 (4단계)

![총비용 대 validation MAE 파레토]({{artifact:4b1f3313-8870-4a8e-a38b-229389646bf5}})

![추론 비용 대 validation MAE]({{artifact:d15fdbdc-4a82-4cbf-92b4-4a2310ee59a8}})

**비용 계산** (`s5_costs.py`; 비용 가정 문서: 0.0504 USD/core-h, 25 W / 12 스레드)

- 탐색 비용은 후보 '계보'에 든 ledger 블록의 합이다.
  - 예: LGB F0F3F5 = 기술자 0.12 + T1 seed·손실 0.98 + T1 튜닝 10.29 + F3 계산 3.26 + T8 F5 규칙·M2 0.91 core-h + API 0.343 USD + T1b 묶음 2.46 + T1b OOF 1.09.
  - 후보와 무관한 실험은 넣지 않았다(T8 N1–N3, T5 random 곡선·전체 추가, T2 그래프 전용 대조 등).
  - 앙상블은 멤버 계보의 합집합에 앙상블 트랙 비용 블록을 더했다.
- 최종 학습 비용은 train 전용 재적합(두 분할, 모든 seed)의 ledger 실측값이다.
- 추론 비용 = 특징 계산 s/1k + 모델 예측 × seed 수다. F3 배좌 계산은 3.257 core-h ÷ 19,945 → 588 s/1k, 기술자는 21.4, F5는 4.8, graph는 4.8, O2 지문+유사도 실측은 4.55, MoLFormer는 143.7, ChemBERTa는 10.8 s/1k다.
- 전체 표: [candidate_ranking_costs.csv]({{artifact:ca972924-4556-42e2-a125-fe4d7b1093dd}})(후보 179개). 열은 두 기준 MAE·순위·순위 차이 표시, 기준선·최선 단일·A 대비 CI, 비용, 파레토·무릎 표시다. 분할별 표는 [candidate_per_split.csv]({{artifact:3add507a-3484-4819-afd7-d4363a3d6d89}})다.

**선정 규칙** (`s6_select.py` 머리말에 실행 전 적은 문장)

- 가장 정확: 사전 지정 방법·단일 중 두 분할 전체 MAE 평균이 최소인 후보다.
  - 글자 그대로는 `wavg_mse|all`(29.4518)이고 `noO2`(29.4529)와 0.001 K 차이다.
  - O2가 든 후보는 PREREG 2절에 따라 중복 제거 기준을 주 기준으로 쓴다. 그 기준에서는 noO2가 낫다(29.607 대 29.609). 그래서 **A = `wavg_mse|noO2`**로 했다. 실질적으로 같은 후보다.
- 가장 경제적(PREREG): A 대비 짝지은 CI가 두 분할 모두 0을 포함하는 후보 중 총비용이 가장 싼 것이다.
  - **엄격 적용 시 승자는 `fixed_wavg_mse|F0F5+GIN+cbMLM@T3`(1.66 USD)다. 채택하지 않았다.** 이유는 다음과 같다.
    1. 이 부분집합은 이 트랙이 T3의 탐색적 관찰을 보고 사후에 만든 것이다. 리드 지시('특별 취급 금지')에 따라 T3 사후 부분집합을 경제안 규칙에서 뺐다.
    2. scaffold CI 하한이 −0.00이고 부트스트랩 seed 20개 중 85 %에서만 통과했다([econ_rule_bootstrap_seed_robustness.csv]({{artifact:3de83acd-3c34-449c-9bec-7c4ecede7d2a}})).
  - 그다음으로 싼 B(`fixed_mean|best+GIN`, 1.83 USD)는 같은 비용의 가중 변형과 동률이다. 적합 모수가 없는 단순 평균을 골랐고, 20개 seed 모두에서 통과했다.
  - B′는 B와 정확도가 비슷하면서 추론 비용이 1/14(44.9 s/1k)이다. 3D 배좌가 문제라면 3단계에서 대안으로 고려할 만하다.
- 균형안: 기준선보다 나쁘지 않은 후보로 만든 총비용(로그) 대 두 분할 평균 MAE 파레토 전선의 무릎(양 끝을 잇는 직선에서 정규화 거리 최대)이다.
  - 처음 규칙은 전체 후보로 전선을 만들었다. 그러면 O2(47.5 K)가 한쪽 끝이 되어 무릎이 기준선으로 퇴화했다. 그래서 '기준선 이하' 제한을 추가했다(실행 후 변경).
  - 결과는 **C = `xgb600_F0F5_M2`**다. 추론비용 축의 무릎은 `fixed_wavg_mse|F0F5+GIN`이며 표에 표시했다.

**두 기준 순위**

- A는 두 분할 모두 순위가 같다(random 42/42, scaffold 10/10).
- B는 중복 제거 기준에서 오른다(random 50 → 29, scaffold 56 → 52). 선정은 전체 기준과 CI로 했고, 중복 제거 기준으로 바꿔도 A 대비 차이가 n.s.라 선정이 같다(random −0.02 [−0.15, +0.11], scaffold +0.14 [−0.10, +0.38]).
- C는 두 기준 순위가 같다.

## 7. 편차와 단서

1. **T3 포함.** T3 멤버 6개를 [members_T3.csv]({{artifact:24c0572b-8e69-411a-a571-6b680294d793}})로 더해 1단계 전체를 다시 돌렸다. 결과는 위 5절에 있다. 최종 후보는 T3 전 풀에서 나왔다. 두 풀 중 더 좋은 쪽을 고른 것도 validation 기반 선택이며, 차이는 0.02–0.03 K다.
2. **scaffold train OOF 이질성.** 멤버마다 fold 규칙이 다르다.
   - P1 기준선은 자체 그룹 fold다. T1·T1b는 GroupKFold Murcko다. T8 M2는 무작위 KFold라 OOF MAE 25.28 K로 낙관적이다.
   - M2를 scaffold OOF 가중·메타모델 입력에서 뺐다.
   - scaffold OOF 기반 수치(가중 확인, 메타모델 A)는 이 이질성을 안고 있다.
3. O2 kNN의 train OOF는 P1 fold 열이 아니라 T1 fold 규칙으로 만들었다. 최선 단일 OOF와 맞추기 위해서다.
4. Ridge 스태킹은 alpha가 격자 끝(1e4)으로 간 경우가 있다(random all, scaffold all). scaffold kNN k 20도 격자 끝이다.
5. 멤버 2개 MAE 가중치 계산을 실행 도중 LP에서 가중 중앙값 정확해로 바꿨다. 한 사례에서 LP와 같은 가중치·MAE임을 확인했다. 초기 s2 결과는 LP로 계산한 것이다.
6. 고정 부분집합 앙상블(B 포함)은 s2 결과를 본 뒤 지정했다(사후). 가중치는 정직하게 교차 적합했지만, 부분집합 선택의 낙관은 fwd{m}_mae(선택까지 교차 적합)가 가늠한다. fwd2는 random 24.874, scaffold 34.191 K였다.
7. 메타모델 A에서 앙상블 대상은 최선 단일 OOF의 |오차|를 대리 목표로 썼다. 앙상블 멤버 GIN·특허에는 OOF가 없기 때문이다. 목표 C 임계값은 '최소 30분자'를 요구한다.
8. 라벨 품질 5(측정 1개)의 정의를 처음에 DPG와 겹치게 잘못 써서 고쳐 다시 실행했다(n 1,554 → 1,547).
9. 비용 근사. T3 XGB 예측 시간은 ledger에 없어 0.01 s/1k로 가정했다. 보조 물성 모델 추론은 0으로 잡았다. 앙상블에 분할 전용 멤버가 있으면 재적합 비용을 두 분할 합으로 셌다(작은 과대 계상). 계보 블록 귀속은 위 6절 규칙에 따른 판단이다.
10. 미측정 CPU 0.25 core-h를 ledger에 추정치로 적었다. 폐기한 첫 위험–커버리지 실행, 1차 후보 비교, 그림·검증 셀이 해당한다.
11. A의 random 가중치는 반올림 4자리 합이 1.0001이다. test에서는 합으로 나눈다.
12. GIN 재학습의 동일성은 확인되지 않았다(T2도 미확인). A·B 모두 GIN에 의존한다.

## 8. 실패하거나 이득이 없던 시도

- O2 Tanimoto kNN: 단독 MAE가 기준선보다 11–20 K 나쁘고 앙상블 기여 0이다.
- 재보정: 선형 n.s., isotonic 유의 악화.
- Ridge 스태킹, 전체 단순 평균: random에서 가중 평균보다 나쁘다(mean_all +0.43 K vs 최선 단일).
- train OOF 기반 가중 적합: OOF 없는 GIN을 못 써서 이득 −0.03 / −0.18 K(n.s.).
- T3 임베딩 멤버: 강한 앙상블에 추가 이득 없음.
- 목표 C: scaffold에서 10 K 커버리지 없음. ens_sd 단독과 nn1 단독 점수로는 random에서도 10 K에 닿지 못했다.
- 처음 균형안 무릎 규칙은 O2 끝점 때문에 퇴화했다(위 6절).

## 9. 파일

- 결과 표: [ensemble_results.csv]({{artifact:68444def-e018-4780-9527-847de2b2bec6}}), [ensemble_results_T3.csv]({{artifact:136d0422-38c0-407b-9a0a-2e2da8753913}}), 가중치 [T3 전]({{artifact:8ddffcc3-110d-4455-abb8-8eec0942d602}}) / [T3]({{artifact:12d8ce3c-82c4-489e-868f-6b12bc3dfd5e}}), 고정 부분집합 가중치 [T3 전]({{artifact:2cbf79c6-af8e-46e5-aa5a-7fba466d5b6e}}) / [T3]({{artifact:b27d3255-4a26-46e3-9c32-234fb8a51b7c}})
- 후보별 validation 예측(정직한 교차 적합 포함, 후보 열 × 행): [random]({{artifact:dd11de24-070b-4758-a321-0bba7311bc31}}), [scaffold]({{artifact:73026055-f6d9-44c0-9da4-4a0a7d4e6160}}). 반복별 열이 있는 A·B 파일: [A random]({{artifact:a1f417af-31dc-4c6b-88db-2b3cf198601e}}), [A scaffold]({{artifact:79e9a012-6e2b-45b8-a974-f39963d34e72}}), [B random]({{artifact:e1e6ff44-0a7d-4d90-95d7-7ce353e704e5}}), [B scaffold]({{artifact:250b7d76-393c-429e-8996-a6ceeadde5ea}})
- 실행 명세: [final_candidates_spec.md]({{artifact:d9f4790d-69a0-4017-af69-c16c8e42bef9}}), [final_candidates_spec.csv]({{artifact:d703797c-eca0-4aee-baa8-25411395e38e}})
- ledger: [ledger.csv]({{artifact:0be9ba09-102a-4052-bfce-7399d9f9cced}})(mpcommon.LEDGER_COLS, 20줄, 1.66 core-h)
- 스크립트: [enslib.py]({{artifact:c74f92d6-029b-4810-bd6b-d12b3d88e9a1}}), [s1_knn.py]({{artifact:f4fe0cdb-e7ef-472b-8246-d120c55f05f7}}), [s2_ensemble.py]({{artifact:60fe9384-b688-4f4a-9e41-3353904ad230}}), [s2b_subsets.py]({{artifact:75ff58b4-ca9b-4ffe-b746-f3875fec6b96}}), [s3_recal.py]({{artifact:b9cce6e2-e2fa-4e8c-9fab-f14cf159e6e1}}), [s4_risk.py]({{artifact:a5eb697c-04af-4feb-a8c3-6b64e449dc4f}}), [s5_costs.py]({{artifact:6d60e988-c1f0-432a-9226-7af748451db1}}), [s6_select.py]({{artifact:6def564a-0b9e-4c64-a129-e205d190f6ab}}), [s7_final_table.py]({{artifact:426f380a-5cec-4b29-9790-b53fae88a07d}}), 멤버 등록부 [members.csv]({{artifact:5d842dea-fd6f-4c90-906a-1f7891ea5be9}}) / [members_T3.csv]({{artifact:24c0572b-8e69-411a-a571-6b680294d793}})
- 다시 돌리는 법: 작업 공간 루트에서 다음 순서로 실행한다. 새 멤버는 members*.csv에 한 줄씩 더한다.
  1. `python ENS/s1_knn.py`
  2. `s2_ensemble.main(tag)`
  3. `s2b_subsets.main(tag, extra)`
  4. `s3_recal.main(...)`
  5. `s4_risk.main(targets, tag)`
  6. `s7_final_table.run()`
