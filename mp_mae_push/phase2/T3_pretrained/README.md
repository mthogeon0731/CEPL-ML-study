# T3 사전학습 표현 트랙 보고 (2026-11-21)

validation만 평가했다. test 예측·평가는 하지 않았다. 분할은 `splits/random.csv`, `splits/scaffold.csv`(validation n = 1,995 / 1,994), 중복 제거 validation은 `prereg/dedup_rows_*.csv`(n = 1,922 / 1,986).

## 0. 결론 (H14)

- **얼린 임베딩을 RDKit217에 붙여도 XGBoost(기준선 설정) 성능은 좋아지지 않았다.** 9개 XGB 결합 변형 중 RDKit217 단독보다 유의하게 나은 것은 없고(짝지은 95 % CI가 모두 0을 포함하거나 양수 쪽), random 분할에서는 MoLFormer 전체(+0.81 K), ChemBERTa-MLM(+0.97 K), MoLFormer PCA128(+0.60 K), scaffold에서는 ChemBERTa-MTR(+0.62 K)이 유의하게 나빴다. → 판정: **앙상블 멤버로만**.
- 임베딩 단독 XGB는 RDKit217보다 5.4–6.8 K 나쁘다(random 32.8–33.7 K, scaffold 42.1–43.5 K). 잔차 상관은 기준선과 0.79–0.87로 가장 다양하지만, 고정 50:50 평균으로는 R217보다 1.0–1.8 K 나빠진다.
- Ridge head에서는 임베딩 결합이 1.4–3.1 K 유의하게 개선하지만(선형 head는 기술자만으로 표현력이 부족), 가장 좋은 Ridge(R217 + MoLFormer: 33.41 / 40.21 K)도 XGB R217(27.41 / 36.72 K)보다 크게 나쁘다.
- 탐색적(사후 선택): R217 XGB 3 seed와 R217 + ChemBERTa-MLM XGB 3 seed의 50:50 평균은 scaffold에서 R217 6-seed 평균보다 −0.36 K [−0.65, −0.06] (중복 제거 −0.36 [−0.64, −0.07]). random에서는 +0.11 [−0.15, +0.36]. 8개 비교 중 하나이고 가중치·멤버를 validation을 보고 고른 것이므로 확정 결과가 아니다 — 앙상블 트랙에서 train 안 CV로 다시 판정할 후보로만 넘긴다.
- 문헌 수치(Dubonos 2026, ChemBERTa + MoLFormer 임베딩 + TabM, MP MAE 26.75 K, 11.8만 분자, 초록만 확인)와는 데이터 크기·head·분할이 달라 직접 비교할 수 없다.

## 1. 모델별 비용과 처리

| 항목 | ChemBERTa-77M-MTR | ChemBERTa-77M-MLM | MoLFormer-XL-both-10pct |
|---|---|---|---|
| 저장소 / revision | DeepChem/ChemBERTa-77M-MTR @66b895cab8 | DeepChem/ChemBERTa-77M-MLM @ed8a5374f2 | ibm-research/MoLFormer-XL-both-10pct @361063d0ad |
| 받은 파일 크기 | 14.07 MB (pytorch_model.bin 14.04 MB) | 13.75 MB | 187.36 MB (model.safetensors 187.25 MB) |
| 다운로드 시간 | 3.8 s | 3.5 s | 20.6 s |
| 가중치 형식 | pickle(.bin) — transformers 5.23 / torch 2.14 로드 | 같음 | safetensors |
| 매개변수(인코더) | 3,427,440 (3층, hidden 384; 무작위 초기화된 pooler 147,840 포함, 사용 안 함) | 같음 | 44,375,040 (12층, hidden 768; LM head 제외) |
| 라이선스(모델 카드) | 저장소에 README(모델 카드) 없음, 라이선스 미표기 | 같음 | apache-2.0 (모델 카드 메타데이터) |
| 사전학습 목표 | config에 회귀 출력 199개 + 정규화 평균·표준편차 (ChemBERTa-2 MTR은 RDKit 기술자 회귀로 알려짐 — 원문 미확인) | 마스크 언어모델 | 마스크 언어모델 (선형 attention, rotary) |
| 입력 SMILES | RDKit canonical (isomeric) | 같음 | RDKit canonical, 입체 정보 제거(모델 카드: 학습 시 isomeric 정보 제거·canonical) |
| 최대 길이 처리 | 512 토큰(max_position 515); 잘림 0분자 | 같음 | 202 토큰; **잘림 1분자**, `<unk>` 포함 5분자(그대로 사용) |
| 토큰 길이 (중앙값 / 99 % / 최대) | 25 / 73 / 242 | 같음 | 25 / 73 / 242 |
| 500분자 파일럿 CPU | 6.34 core-s (wall 2.05 s) | 6.48 core-s (2.02 s) | 83.16 core-s (25.02 s) |
| 파일럿 외삽 (19,945분자) | 0.070 core-h | 0.072 core-h | 0.921 core-h |
| **전체 임베딩 실측** | **0.0601 core-h** (wall 69.3 s, RSS 1.18 GB) | **0.0599 core-h** (68.6 s, 1.23 GB) | **0.7958 core-h** (864.3 s, 1.35 GB) |
| 1,000분자당 추론 | 10.84 core-s / 3.48 s wall | 10.81 core-s / 3.44 s wall | 143.65 core-s / 43.34 s wall |

조건: CPU(i5-1335U), torch 스레드 4, fp32, 배치 32(토큰 길이로 정렬), `torch.inference_mode`. mean pooling = attention mask 위 평균(특수 토큰 포함), CLS = 첫 토큰(`<s>` / `<bos>`) 은닉 상태. 두 가지를 한 번의 순전파에서 함께 저장했다. 다른 트랙과 동시 실행 중이라 시간은 그 부하를 포함한다.

임베딩 파일(float32, 19,945 × d, 행 = mp `row` 순서):
[cb_mtr_mean.npy]({{artifact:f70d2b08-feba-4514-8aed-0e4530faf323}}), [cb_mtr_cls.npy]({{artifact:027d23e1-b7b9-4a99-8f37-336ae93a19d5}}), [cb_mlm_mean.npy]({{artifact:a1d1ed38-2e11-4860-9c32-dba5adc5bb66}}), [cb_mlm_cls.npy]({{artifact:f432007d-f182-4294-b1f9-dea1c505cb56}}), [molformer_mean.npy]({{artifact:9b525fe0-abfc-4e37-a44d-492da60f0423}}), [molformer_cls.npy]({{artifact:b30072c5-3b5d-4e37-99e4-dfe383deb97f}}).

## 2. MoLFormer 사용자 코드 검토 (trust_remote_code)

- 대상: revision 361063d0ad의 `configuration_molformer.py`(143줄, sha256 b88ea8d4b7b5e54f…)와 `modeling_molformer.py`(873줄, sha256 6f1ef72022de2c69…). 토크나이저는 `PreTrainedTokenizerFast` + `tokenizer.json`(WordLevel, 정규식 분할)으로 사용자 코드가 없다. 변환 스크립트(`convert_…py`)는 내려받지 않았다.
- import: `math`, `typing`, `torch`, `torch.nn`, `transformers` 내부 모듈, 상대 import(`.configuration_molformer`)뿐이다.
- 위험 패턴 전수 검색(`os.`, `subprocess`, `socket`, `requests`, `urllib`, `http`, `shutil`, `eval(`, `exec(`, `compile(`, `__import__`, `importlib`, `pickle`, `marshal`, `open(`, `Popen`, `ctypes`, `base64`, `getattr(`/`setattr(`, `globals(`, `torch.load`, `download`, `sys.`, `pathlib`, `tempfile`, `environ`): 일치는 주석·docstring의 URL·설명문뿐이고 실행 코드에는 없다.
- 눈으로 읽은 부분: 임베딩, 무작위 특징 사상(feature map), 선형 attention, 인코더, `MolformerModel.forward`, `masked_avg_pool1d`. 네트워크·파일 시스템·셸 호출 없음 → **로드함**.
- 재현성 주의: config 기본값 `deterministic_eval: false`이면 무작위 특징 가중치를 **추론 때마다 다시 뽑는다**(feature map `forward`). 모델 카드 예시대로 `deterministic_eval=True`로 로드했고, 같은 입력 두 번 순전파의 최대 차이 0.0을 확인했다.
- 가중치는 safetensors라 역직렬화 코드 실행 위험이 없다. ChemBERTa는 사용자 코드가 없는 표준 RoBERTa지만 가중치가 pickle(`pytorch_model.bin`)이다. transformers 5.23이 torch 2.14로 읽었다(`training_args.bin`은 받지 않음).
- 검토한 사본: [molformer_modeling_molformer.py]({{artifact:87d55bb4-cf7e-4dd2-9de0-2b33911bb549}}), [molformer_configuration_molformer.py]({{artifact:19866235-91a4-4792-b69b-468e8cc48398}}).

## 3. head 비교 결과 (validation)

설정: XGBoost = `mpcommon.BASE_XGB`(600 trees, lr 0.05, depth 6, subsample 0.8, colsample 0.5), n_jobs 8, seed 0, 1, 2. Ridge = train 통계로 표준화(±10 클리핑, 결측은 train 중앙값) 후 `RidgeCV`(alpha 1e-2–1e4 13점, train 안 5-fold KFold seed 0, MAE 기준), 결정적이라 1회. PCA는 train 행만으로 fit. 기준(R217)은 **같은 head·같은 seed**의 RDKit217 실행이다(기준선 파일 seed 42가 아님. seed 42 재현은 mp-push에서 최대 차이 3.05e-5 K로 확인).

짝지은 차이 ΔMAE = 분자별 seed 평균 (|e_new,s| − |e_R217,s|)의 평균, 95 % CI = 분자 부트스트랩 2,000회. 음수 = 개선. 순위는 같은 head·같은 분할 안에서 seed별 MAE 평균 기준.

### 3.1 XGBoost (seed 3개, MAE = seed별 평균 ± SD)

| 표현 | 차원 | random MAE | random 중복 제거 | Δ random [95 % CI] (전체 / 중복 제거) | scaffold MAE | scaffold 중복 제거 | Δ scaffold [95 % CI] (전체 / 중복 제거) | 1 fit CPU (core-s) | 판정 |
|---|---|---|---|---|---|---|---|---|---|
| R217 (기준) | 217 | 27.408 ± 0.193 | 27.697 | — | 36.721 ± 0.299 | 36.730 | — | 54 | 기준 |
| R217 + MoLFormer PCA64 | 281 | 27.718 ± 0.158 | 28.093 | +0.310 [−0.041, +0.660] / +0.396 [+0.045, +0.772] | 36.606 ± 0.100 | 36.612 | −0.116 [−0.503, +0.252] / −0.118 [−0.517, +0.255] | 93 | 앙상블 멤버로만 |
| R217 + ChemBERTa-MTR | 601 | 27.762 ± 0.052 | 28.062 | +0.354 [−0.051, +0.736] / +0.365 [−0.053, +0.762] | 37.344 ± 0.050 | 37.357 | +0.623 [+0.147, +1.101] / +0.627 [+0.149, +1.107] | 302 | 앙상블 멤버로만 |
| R217 + MoLFormer PCA128 | 345 | 28.003 ± 0.196 | 28.373 | +0.595 [+0.198, +0.992] / +0.676 [+0.248, +1.079] | 36.931 ± 0.360 | 36.934 | +0.209 [−0.235, +0.628] / +0.204 [−0.225, +0.634] | 154 | 앙상블 멤버로만 |
| R217 + MoLFormer | 985 | 28.220 ± 0.270 | 28.663 | +0.812 [+0.321, +1.294] / +0.966 [+0.488, +1.471] | 36.744 ± 0.368 | 36.756 | +0.023 [−0.497, +0.556] / +0.026 [−0.505, +0.587] | 517 | 앙상블 멤버로만 |
| R217 + ChemBERTa-MLM | 601 | 28.374 ± 0.181 | 28.646 | +0.966 [+0.472, +1.453] / +0.949 [+0.459, +1.447] | **36.457 ± 0.049** | 36.473 | −0.264 [−0.818, +0.252] / −0.256 [−0.781, +0.265] | 266 | 앙상블 멤버로만 |
| ChemBERTa-MTR 단독 | 384 | 32.770 ± 0.284 | 33.143 | +5.363 [+4.499, +6.299] / +5.446 [+4.517, +6.334] | 42.736 ± 0.141 | 42.719 | +6.015 [+4.920, +7.069] / +5.989 [+4.987, +6.993] | 259 | 단독 열세 |
| MoLFormer 단독 | 768 | 33.570 ± 0.219 | 34.145 | +6.162 [+5.217, +7.091] / +6.448 [+5.425, +7.438] | 42.072 ± 0.487 | 42.076 | +5.351 [+4.288, +6.510] / +5.346 [+4.219, +6.479] | 477 | 단독 열세 |
| ChemBERTa-MLM 단독 | 384 | 33.731 ± 0.095 | 34.072 | +6.323 [+5.303, +7.322] / +6.375 [+5.377, +7.392] | 43.517 ± 0.113 | 43.600 | +6.796 [+5.594, +7.990] / +6.870 [+5.667, +8.045] | 222 | 단독 열세 |

두 기준 순위가 다른 후보(XGB): random에서 R217 + MoLFormer PCA64(전체 2위 → 중복 제거 3위), R217 + ChemBERTa-MTR(3 → 2), R217 + MoLFormer(5 → 6), R217 + ChemBERTa-MLM(6 → 5), MoLFormer 단독(8 → 9), ChemBERTa-MLM 단독(9 → 8). scaffold는 두 순위가 같다. 어느 기준으로도 결론(결합 이득 없음)은 같다.

### 3.2 오차 상관과 고정 50:50 평균 (XGB, 탐색적)

잔차 r = seed 평균 예측 잔차의 Pearson 상관. 50:50 = R217 3-seed 평균과 해당 표현 3-seed 평균의 단순 평균(가중치 튜닝 없음). 6개 모델 평균이라 배깅 효과가 섞이므로 대조군은 **R217 seed 0–5의 6-seed 평균**(random 27.066 K, scaffold 36.121 K; 중복 제거 27.348 / 36.127 K).

| 표현 | 잔차 r (같은 head R217 / 기준선 seed 42) random | 같은 r scaffold | 50:50 random MAE, Δ vs R217×6 [CI] | 50:50 scaffold MAE, Δ vs R217×6 [CI] |
|---|---|---|---|---|
| R217 + ChemBERTa-MLM | 0.946 / 0.938 | 0.962 / 0.953 | 27.173, +0.106 [−0.153, +0.358] | 35.761, **−0.359 [−0.647, −0.063]** (중복 제거 −0.356 [−0.641, −0.068]) |
| R217 + ChemBERTa-MTR | 0.963 / 0.953 | 0.971 / 0.961 | 27.001, −0.065 [−0.284, +0.148] | 36.297, +0.176 [−0.093, +0.435] |
| R217 + MoLFormer | 0.945 / 0.939 | 0.964 / 0.955 | 27.028, −0.038 [−0.321, +0.220] | 35.886, −0.235 [−0.514, +0.066] |
| R217 + MoLFormer PCA64 | 0.973 / 0.964 | 0.980 / 0.972 | 27.022, −0.044 [−0.242, +0.145] | 36.024, −0.097 [−0.306, +0.109] |
| R217 + MoLFormer PCA128 | 0.962 / 0.954 | 0.976 / 0.966 | 27.085, +0.019 [−0.204, +0.245] | 36.156, +0.036 [−0.200, +0.266] |
| ChemBERTa-MTR 단독 | 0.838 / 0.832 | 0.866 / 0.857 | 28.525, +1.459 [+0.993, +1.974] | 37.677, +1.557 [+0.917, +2.134] |
| MoLFormer 단독 | 0.817 / 0.814 | 0.854 / 0.848 | 28.772, +1.706 [+1.188, +2.212] | 37.339, +1.218 [+0.628, +1.860] |
| ChemBERTa-MLM 단독 | 0.799 / 0.794 | 0.831 / 0.822 | 28.824, +1.758 [+1.218, +2.273] | 37.823, +1.702 [+1.015, +2.374] |

배깅 대조 없이 R217 3-seed 평균과 비교하면 scaffold에서 R217 + MLM −0.58, R217 + MoLFormer −0.45, PCA64 −0.31 K처럼 커 보이지만, 그중 상당 부분은 모델 수가 늘어난 효과다(R217 자체가 3 → 6 seed로 36.337 → 36.121 K).

### 3.3 Ridge (결정적, 1회)

| 표현 | random MAE (중복 제거) | Δ vs Ridge R217 [CI] | scaffold MAE (중복 제거) | Δ vs Ridge R217 [CI] | 판정 |
|---|---|---|---|---|---|
| R217 (기준, alpha CV) | 35.713 (35.496) | — | 43.294 (43.292) | — | 기준 |
| R217 + MoLFormer | 33.410 (33.343) | −2.303 [−3.221, −1.428] | 40.210 (40.237) | −3.083 [−4.322, −1.931] | 결합 이득(두 기준) |
| R217 + ChemBERTa-MLM | 33.495 (33.419) | −2.217 [−2.970, −1.474] | 40.584 (40.582) | −2.710 [−3.938, −1.556] | 결합 이득(두 기준) |
| R217 + ChemBERTa-MTR | 33.931 (33.788) | −1.781 [−2.501, −1.042] | 41.941 (41.936) | −1.353 [−2.517, −0.289] | 결합 이득(두 기준) |
| ChemBERTa-MTR CLS | 35.359 (35.129) | −0.353 [−1.297, +0.583] | 43.430 (43.389) | +0.136 [−1.257, +1.538] | 차이 없음 |
| MoLFormer mean | 36.544 (36.566) | +0.832 [−0.288, +1.964] | 41.965 (42.004) | −1.328 [−2.958, +0.223] | 차이 없음 |
| ChemBERTa-MTR mean | 36.607 (36.501) | +0.894 [−0.068, +1.889] | 43.382 (43.334) | +0.088 [−1.472, +1.591] | 차이 없음 |
| MoLFormer CLS | 37.777 (37.758) | +2.064 [+0.847, +3.276] | 44.001 (44.019) | +0.707 [−0.926, +2.305] | 열세 |
| ChemBERTa-MLM mean | 38.348 (38.481) | +2.636 [+1.442, +3.829] | 44.474 (44.518) | +1.180 [−0.583, +2.792] | 열세 |
| ChemBERTa-MLM CLS | 42.700 (42.686) | +6.988 [+5.588, +8.375] | 51.901 (51.992) | +8.608 [+6.638, +10.461] | 열세 |

pooling: Ridge에서 mean이 CLS보다 대체로 낫다(MTR random만 예외). XGB는 mean pooling만 돌렸다.

전체 표: [s2_val_summary.csv]({{artifact:fd805a4d-574a-4239-8c72-7d338465cc77}}), 50:50 대조: [s2_blend5050_vs_R217_6seed.csv]({{artifact:f87c631b-4515-4f05-9fa3-79cfeed175cf}}).

## 4. 앙상블 트랙에 넘기는 validation 예측 (row, y, pred; seed 0–2 평균)

| 파일 | 용도 |
|---|---|
| [xgb_R217_cb_mlm_random_val.csv]({{artifact:8372307a-4aed-4f24-a035-a58db336e812}}), [xgb_R217_cb_mlm_scaffold_val.csv]({{artifact:67de1d8d-9862-486d-a908-bb4fabea39e0}}) | scaffold 50:50 신호가 있는 유일한 후보 |
| [xgb_R217_molformer_random_val.csv]({{artifact:efe9161a-7ad3-47ce-aeab-9d0ac9867cb1}}), [xgb_R217_molformer_scaffold_val.csv]({{artifact:04a9b0c0-72d5-41a4-a497-11f213eaacb7}}) | 결합 멤버 |
| [xgb_R217_molformer-pca64_random_val.csv]({{artifact:353af83e-78e3-456f-b0d2-d9c8eee20a3b}}), [xgb_R217_molformer-pca64_scaffold_val.csv]({{artifact:1ee4def1-29de-4f5f-bb48-580f864e852b}}) | 결합 멤버(가장 저렴: fit 93 core-s) |
| [xgb_molformer_random_val.csv]({{artifact:a508e428-dc7c-4053-81ad-dad3fd453ee6}}), [xgb_molformer_scaffold_val.csv]({{artifact:d130314b-5c04-4a46-9db7-7533d197c514}}) | 임베딩 단독(오차 상관 최저 계열) |
| [xgb_cb_mtr_random_val.csv]({{artifact:3a593f05-9a7d-46ab-8f7e-c406abca7568}}), [xgb_cb_mtr_scaffold_val.csv]({{artifact:fa143b8b-f1dd-4dff-a2b9-09e3b9e6c727}}) | 임베딩 단독(가장 저렴한 임베딩) |
| [xgb_R217_random_val.csv]({{artifact:6a3c3e1e-8403-4e9f-ab82-236e35e5ea09}}), [xgb_R217_scaffold_val.csv]({{artifact:c5e472f2-459a-41ec-aa62-eed1d9e4ba33}}) | 같은 seed의 R217 기준(내용이 T1의 F0 sq seed 0–2 평균 파일과 바이트 단위로 같아 저장소가 기존 파일로 연결함) |
| [ALL_val_preds_long.csv]({{artifact:9da2ff74-0473-4d5e-9a8b-9eab73c52624}}) | 38개 (head, 표현, 분할) 전부, 열 head, rep, split, row, y, pred |

test 예측은 만들지 않았다(최종 후보가 아님). 앙상블에 들어가 최종 후보가 되면 같은 스크립트로 test 예측을 만들어야 하고, 그때 MoLFormer 임베딩 추론 비용(1,000분자당 143.65 core-s)이 추론 비용에 더해진다.

## 5. CPU 비용 (ledger 합계)

| 구분 | core-h |
|---|---|
| 파일럿 500분자 × 3 모델 | 0.027 |
| 전체 임베딩 × 3 모델 | 0.916 |
| 기준선 재현 확인 (mp-push) | 0.018 |
| head 비교 (XGB 54 fit + R217 배깅 대조 XGB 6 fit + Ridge 20 fit + 요약) | 4.263 |
| **합계 (측정분)** | **5.224** (예산 6) |
| mp-text 커널 XGB 교착 (약 1.6 h, 리드가 중단) | 미측정 |

ledger: [ledger.csv]({{artifact:fae2d958-5af6-4f07-98a9-0dfdba240eb7}}) (88행; `pilot500_*`·`repro_*`는 커널 로그 값을 옮겨 적은 행, `hung_xgb_mptext`는 비용 미측정 실패 행).

## 6. 하지 않은 것과 이유

- **CheMeleon 미실시**: chemprop 설치 승인이 만료되어 리드가 이번 실행에서 제외했다(가중치도 받지 않음).
- CLS pooling은 Ridge에서만 평가했다(XGB는 mean pooling만; 예산).
- PCA 변형은 차원이 큰 MoLFormer(768)에만 적용했다. ChemBERTa(384)에는 적용하지 않았다.
- 미세조정은 하지 않았다(얼린 인코더만; CPU 예산 6 core-h와 규약 8절 '언어모델 미세조정은 단독 실행').

## 7. 실행 환경과 실패 기록

- 임베딩: env `mp-text`(torch 2.14.1 CPU, transformers 5.23.0). head 비교: env `mp-push`(xgboost 3.4.2, torch 미로드).
- 실패 1: `mp-text` 커널에서 torch를 로드한 뒤 XGBoost를 fit하자 약 1.6시간 멈췄다(정상 10 s). Windows에서 torch·xgboost OpenMP 런타임 충돌로 추정한다. 리드가 중단했고 커널이 초기화됐다. 이후 head 비교는 torch가 없는 `mp-push`에서 새 커널로 했다. 임베딩 파일은 디스크에 남아 있어 다시 계산하지 않았다.
- 실패 2(수정됨): 요약 스크립트 첫 실행에서 판정 문자열과 순위 계산에 버그가 있었다(파일명의 `+` → `_` 치환, `T.head` 메서드 충돌). 고친 뒤 다시 계산했고, 표의 수치는 고친 결과다.

## 8. 코드

[s2_heads.py]({{artifact:cedf9911-9bf8-44e7-b2f8-db08eab7efb3}}) (표현 구성·head 실행·ledger), [s2_summary.py]({{artifact:88eb6ec1-abdf-43c2-b0d7-a63d83866f1b}}) (짝지은 CI·순위·예측 CSV). 임베딩 코드는 대화 커널에서 실행했다: 배치 32, 길이 정렬, `AutoModel.from_pretrained(..., deterministic_eval=True, trust_remote_code=True)`(MoLFormer), mean·CLS를 한 번의 순전파에서 추출.
