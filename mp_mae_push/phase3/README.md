# 3단계 최종 평가 (P3) — 누설 감사와 최종 후보 3개의 test 1회 평가

작성 2026-11-22. Bradley Open MP 녹는점, 단위 K. 주 결과는 모두 **train 전용 학습**이다(random 15,955행 / scaffold 15,956행, 원 기준선과 같은 train).

- test는 random n 1,995 / scaffold n 1,995이다. 중복 제거 test는 random n 1,896 / scaffold n 1,984다.
- Δ는 MAE(후보) − MAE(비교 대상)이고, 음수가 개선이다.
- CI는 짝지은 부트스트랩 2,000회(seed 0)의 95 % 구간이다.
- 스레드는 XGBoost n_jobs 8, LightGBM num_threads 6, GIN torch 1이다.
- seed는 후보 A·B의 GBDT·GIN 멤버 0,1,2, 특허 멤버 42,43,44, 후보 C 42,0,1이다.

## 편차

1. **'python' 환경 커널 교착.**
   - 문서 발췌용 출력 셀이 약 50분 응답하지 않아 리드가 중단했다.
   - 이후 계산은 전부 `mp-push` 환경(GBDT·평가)과 mlstudy 인터프리터(GIN)에서 했다. 결과 영향은 없다.
   - 실수로 'python' 환경에 빈 셀 1개(`print('skip')`)를 한 번 보냈다. 아무것도 바꾸지 않았다.
2. **GIN 실행 방식.**
   - `run_mlstudy.ps1`을 `powershell -File`로 중첩 호출하면 샌드박스에서 출력 없이 실패했다.
   - 그래서 같은 환경 변수(PATH, KMP_DUPLICATE_LIB_OK, PYTHONPATH, TEMP, PYTHONDONTWRITEBYTECODE)를 한 호출 안에서 직접 설정하고 mlstudy `python.exe`를 실행했다.
   - GIN 스크립트 `gin_p3.py`는 T2 `train_hybrid.py`의 `run()`과 같다. 바꾼 점은 네 가지다: test 라벨 NaN 차단, 학습 뒤 test 예측(라벨 없이 저장), 가중치 저장, 프로세스별 ledger.
   - 6개 실행 모두 T2와 epoch·best epoch·validation 예측까지 **완전히 같았다**.
3. **특허 표본 추출 함수 재구성.**
   - T5 `step5_curve.py`에는 add1x 표본 함수(`pick`)가 저장되어 있지 않다.
   - `np.random.default_rng(seed).choice(194801, n_train, replace=False)`로 재구성했다.
   - seed 42/43/44의 validation MAE가 T5 ledger 값 35.3331 / 35.7513 / 36.2183 K와 소수 넷째 자리까지 같아 같은 함수로 판정했다.
4. **lgb_r1_F0(목표 C 멤버)은 T1의 8 스레드 대신 지시대로 6 스레드로 학습했다.** validation 예측은 저장본과 비트 단위로 같다.
5. **대체안 B_alt(lgb_r1_F0F3F5 단일)의 test 예측 파일도 만들어 sha256을 기록했다.**
   - GIN 재현 점검을 통과했으므로(아래) 평가에는 쓰지 않았다(지표 미산출).
   - 같은 예측이 멤버 파일 `lgb_r1_F0F3F5_*_test.csv`로도 있다.
6. **평가 스크립트 `s9_evaluate.py`는 한 번만 실행했다(버그 수정 재실행 없음).**
   - 그림 스크립트는 축·범례 표기 문제로 두 번 다시 그렸다(수치 변화 없음).
   - 명세에 없는 A−C 차이 CI도 함께 냈다.
7. **보조 학습(train+validation) 시작 시점.** 주 결과 표·예측 파일을 디스크에 모두 쓴 뒤, 아티팩트로 올리기 전에 시작했다. 순서 규칙(주 결과 저장 뒤)의 의도는 지켰다.
8. **seed 평균 ± SD의 정의.** 앙상블은 'i번째 seed끼리 조합'한 3개 예측의 MAE 평균 ± SD다. 특허 멤버는 42/43/44를 0/1/2와 짝지었다.
9. **ledger 표기 수정(검토 지적).** h06 12줄의 objective 열이 sq로 잘못 적혀 Tm-scale huber로 고쳤다. O2 kNN 줄의 n_train도 두 분할 값으로 고쳤다. 수치 영향은 없다.
10. **CPU 미측정분.** 커널 쪽 조립·점검·그림 스크립트(s3, s4, s5, s10; 각 수 초)는 ledger에 없다. 추정 < 0.02 core-h다.
11. **목표 C 메타모델 입력 n_meas.** test 분자의 반복측정 수(출처 메타데이터)를 쓴다. 라벨 값 자체는 아니다(누설 감사 2절 단서).

## 1. 결과 요약

1. **두 앙상블(A·B)은 두 분할 모두에서 기준선보다 유의하게 낫다.**
   - random test MAE: A 24.750 K, B 24.755 K (원본 예측 파일 27.500 K).
   - 원본 대비: A -2.75 [-3.29, -2.25], B -2.74 [-3.32, -2.22] K.
   - scaffold: A 33.649, B 33.987 K. sandbox 기준선(35.852) 대비 A -2.20 [-2.79, -1.61], B -1.86 [-2.52, -1.24].
2. **A와 B의 관계는 분할마다 다르다.**
   - random에서는 구분되지 않는다: A−B -0.01 [-0.13, +0.12], 중복 제거 +0.05 [-0.07, +0.19]. 두 기준 순위가 뒤바뀐다(전체 A 1위, 중복 제거 B 1위).
   - scaffold에서는 A가 유의하게 낫다: -0.34 [-0.62, -0.06]. A에만 있는 scaffold 멤버는 특허 1배 XGB와 h06이다(test에서 멤버별 기여 분해는 하지 않음).
3. **C(XGB F0+F5 단일, 추론 26 s/1k)도 기준선보다 유의하게 낫다.** 원본 대비 -0.99 [-1.42, -0.57], scaffold sandbox 대비 -0.93 [-1.46, -0.41]다. 다만 A·B보다 random 1.75–1.76 K, scaffold 0.94–1.27 K 나쁘다(B−C random -1.75 [-2.25, -1.25]).
4. **test 결과는 validation 예상과 잘 맞는다.**
   - validation(ENS 정직한 교차 적합): A 24.889 / 34.017, B 24.917 / 34.161, C 26.499 / 35.477 K.
   - test: A 24.75 / 33.65, B 24.76 / 33.99, C 26.51 / 34.92 K.
   - 차이는 random −0.16 ~ +0.01 K, scaffold −0.17 ~ −0.55 K다.
5. **목표 C(10 K 커버리지)는 test에서 달성하지 못했다.**
   - random에 validation 고정 τ를 적용하면 커버리지 4.5–6.8 %, 부분집합 MAE 12.6–12.9 K다(CI 하한 10.75 K 초과).
   - validation(≈ 10 K, 경계)보다 약 2.7–3.0 K 나쁘다.
   - scaffold는 사전 예측대로 미달이다(상위 5 % MAE 24.2–26.3 K).
6. **누설 감사 다섯 항목을 모두 통과했다.** 항목 i은 조건부(고정 분할의 알려진 겹침을 중복 제거 test로 처리), 항목 iii은 단서 1건(1단계 학습 곡선 진단 행)이다. → [leakage_audit.md]({{artifact:bd79f6f1-38dc-4663-b709-8cb3597dc5e3}})
7. **보조: train+validation 재학습은 test를 더 줄인다.**
   - A(GIN 포함 전부) random 24.482 / scaffold 32.503 K.
   - B(LGB만) random 24.437 K, B(전부) scaffold 32.582 K.
   - C random 26.191 / scaffold 33.930 K.
   - scaffold 이득(−0.8 ~ −1.4 K)이 random(−0.2 ~ −0.4 K)보다 크다.

### 핵심 표 (test, 단위 K)

| 후보 | random MAE (n 1,995) | random 중복 제거 (n 1,896) | random 원본 기준 대비 ΔMAE [95 % CI] | random sandbox 기준 대비 | scaffold MAE (n 1,995) | scaffold 중복 제거 (n 1,984) | scaffold sandbox 기준 대비 | 순위 random 전체/중복 제거 | 순위 scaffold |
|---|---|---|---|---|---|---|---|---|---|
| A `wavg_mse\|noO2` | 24.750 | 25.045 | -2.75 [-3.29, -2.25] | -2.88 [-3.42, -2.35] | 33.649 | 33.627 | -2.20 [-2.79, -1.61] | 1/2 **(다름)** | 1/1 |
| B `fixed_mean\|best+GIN` | 24.755 | 24.991 | -2.74 [-3.32, -2.22] | -2.88 [-3.45, -2.31] | 33.987 | 33.975 | -1.86 [-2.52, -1.24] | 2/1 **(다름)** | 2/2 |
| C `xgb600_F0F5_M2` | 26.509 | 26.841 | -0.99 [-1.42, -0.57] | -1.12 [-1.55, -0.69] | 34.924 | 34.896 | -0.93 [-1.46, -0.41] | 3/3 | 3/3 |
| 원본 예측 파일(수업 XGB+RDKit217) | 27.500 | 27.790 | 0 | – | – | – | – | 4/4 | – |
| sandbox 재적합 기준선(n_jobs 8, seed 42) | 27.633 | 27.940 | – | 0 | 35.852 | 35.826 | 0 | 5/5 | 4/4 |

- 순위는 위 5개 예측 사이에서 매겼다. '다름'은 전체와 중복 제거 순위가 갈린 경우다.
- seed 평균 ± SD(seed 대응 조합 3개, 전체 test):
  - random: A 24.96 ± 0.25, B 25.08 ± 0.27, C 26.77 ± 0.17.
  - scaffold: A 33.86 ± 0.15, B 34.34 ± 0.19, C 35.28 ± 0.33.
  - 단일 seed 조합은 seed 평균 예측보다 0.2–0.35 K 나쁘다.

![random test parity: 후보 A·C]({{artifact:b9e3a95d-3ded-453d-bc96-c751420ba086}})

## 2. 누설 감사 (test를 열기 전에 i–iv 완료)

| 항목 | 판정 | 근거 요약 |
|---|---|---|
| i InChIKey 겹침 | 조건부 통과 | SMILES에서 다시 계산한 키가 분할 파일과 불일치 0이다. test 첫 블록 겹침은 random 99 / scaffold 11행이고 중복 제거 목록과 행 단위로 같다. 특허 정제본 194,801분자는 어느 세트와도 전체 키·첫 블록 0, LLM에 보인 200분자는 두 분할 train에 있고 validation·test와 첫 블록 0이다. |
| ii 라벨 유래 특징 | 통과 | test 행 150개(F3는 25개)에서 F0·F5·F1·O3·F3를 SMILES로 다시 계산하니 최대 차이 0이다. 특징 코드에 라벨·외부 MP·LLM 참조가 없다. train 라벨과의 \|r\| 최대는 0.57이다. F3 39열에 보조 물성 열이 없고, 보조 물성 멤버는 후보에 없다. |
| iii test 없이 결정 | 통과(단서) | 2단계 9개 트랙 ledger의 test 열은 모두 비어 있다. 1단계 ledger만 기준선 2행과 학습 곡선 진단 24행(고정 기준선 설정, 선택에 미사용)에 test가 있다. 가중치·τ·메타모델은 validation 고정값이다. test 예측 29개 파일의 sha256을 평가 전에 기록했고, 평가 스크립트가 다시 확인했다. |
| iv T8 기억 누설 | 통과 | 분자별 LLM 호출이 없다. F5는 고정 RDKit 코드다. 규칙 생성 프롬프트의 200분자는 train 전용이다. |
| v 문헌 대비 | 통과 | 최저 test MAE 24.75 K로 20 K 안전 규칙이 발동하지 않았다. Bradley/Tetko 계열 25–33 K의 하단이다. 중복 제거 전후 차이(+0.24 ~ +0.33 K)가 기준선(+0.29 K)과 같다. |

상세: [leakage_audit.md]({{artifact:bd79f6f1-38dc-4663-b709-8cb3597dc5e3}}). 표: [audit_i_inchikey_overlap.csv]({{artifact:f2b70997-cb5e-4266-92bf-5f8dd2aa3476}}), [audit_ii_feature_recompute.csv]({{artifact:f74088bb-9eb2-4b5b-bc4d-f408a44b897d}}), [audit_ii_label_corr_train.csv]({{artifact:55aa44bf-32e2-4ce9-807a-bfde1426e0d0}}), [audit_ii_static_code.csv]({{artifact:92c7c1e1-c19f-484f-899a-e4d413766626}}), [audit_iii_ledgers.csv]({{artifact:c5ed80ea-0f7d-4259-9eb0-0ee071cf2d05}}), [audit_iii_script_test_refs.csv]({{artifact:bb8c1704-b986-44fb-acf1-bae285b775eb}}).

## 3. 재학습과 validation 재현 점검 (test 열기 전)

- **모든 멤버가 저장본과 같다.**
  - GBDT 15개(멤버 × 분할)는 seed 평균 validation 예측의 최대 |차이|가 0.0이다. 모든 seed MAE가 2단계 값과 같다.
  - GIN 하이브리드는 seed 3개 × 2분할 모두 seed별 예측이 동일하다(최대 차이 0). 명세 기준(seed 평균 MAE 차이 ≤ 0.1 K, 상관 ≥ 0.995)을 통과했다. 그래서 GIN 없는 대체안은 쓰지 않았다.
- **목표 C 입력도 같다.**
  - O2 kNN validation 예측은 같다(차이 0). 유사도 특징 차이는 ≤ 3e-8이다.
  - 메타모델 점수는 차이 0이다(ens_sd 차이 ≤ 1.1e-5 K).
- **후보 조립.** 고정 가중치로 조립한 validation MAE는 B·C에서 ENS 값과 같다. A는 in-sample 24.839 / 33.979 K로, ENS 정직한 교차 적합 24.889 / 34.017 K보다 0.04–0.05 K 낮다(같은 validation에 맞춘 가중치이기 때문).
- **기준선.** sandbox 재적합(seed 42, n_jobs 8)의 test 예측은 저장된 sandbox 기준선 파일과 최대 3.0e-5 K 차이다. 원본 예측 파일은 SMILES 순서가 random test 행과 정확히 같고 라벨도 같다. 다만 예측은 sandbox 재적합과 평균 5.9 K 다르다(1단계에서 알려진 미재현).

| 멤버 | 분할 | seed | validation MAE 재학습 / 저장본 | 최대 \|차이\| | 판정 |
|---|---|---|---|---|---|
| lgb_r1_F0F3F5 | random | 0,1,2 | 25.3590 / 25.3590 | 0 | 동일 |
| lgb_r1_F0F5 | random | 0,1,2 | 25.4677 / 25.4677 | 0 | 동일 |
| h06_yalk_F0F3F5 | random | 0,1,2 | 25.4212 / 25.4212 | 0 | 동일 |
| xgb600_F0F1O3 | random | 0,1,2 | 27.3993 / 27.3993 | 0 | 동일 |
| xgb600_F0F5_M2 | random | 42,0,1 | 26.4992 / 26.4992 | 0 | 동일 |
| baseline_xgb600_F0 | random | 42 | 27.6072 / 27.6072 | 3e-05 | 동일 |
| lgb_r1_F0 | random | 0,1,2 | 26.1505 / 26.1505 | 0 | 동일 |
| xgb_r1_F0 | random | 0,1,2 | 26.2537 / 26.2537 | 0 | 동일 |
| gin_hybrid_seedavg | random | 0,1,2 | 26.2554 / 26.2554 | 0 | 통과(명세 기준) |
| lgb_r1_F0F3F5 | scaffold | 0,1,2 | 35.2376 / 35.2376 | 0 | 동일 |
| h06_yalk_F0F3F5 | scaffold | 0,1,2 | 35.1936 / 35.1936 | 0 | 동일 |
| xgb600_F0_patent1x_scaf | scaffold | 42,43,44 | 35.3434 / 35.3434 | 0 | 동일 |
| xgb600_F0F5_M2 | scaffold | 42,0,1 | 35.4766 / 35.4766 | 0 | 동일 |
| baseline_xgb600_F0 | scaffold | 42 | 36.3484 / 36.3484 | 3e-05 | 동일 |
| lgb_r1_F0 | scaffold | 0,1,2 | 35.9708 / 35.9708 | 0 | 동일 |
| xgb_r1_F0 | scaffold | 0,1,2 | 36.2232 / 36.2232 | 0 | 동일 |
| gin_hybrid_seedavg | scaffold | 0,1,2 | 35.3744 / 35.3744 | 0 | 통과(명세 기준) |

표: [repro_val.csv]({{artifact:5a9a6583-a238-4afc-8c94-35fdad9445f1}}), [repro_candidates_val.csv]({{artifact:8bcb6de7-13f8-4e9a-b79a-0d49d5d0e5e0}}), [repro_knn_sim.csv]({{artifact:90dae79f-a25c-41c2-a61d-da0c9a7760ae}}), [repro_goalC_val.csv]({{artifact:873196a5-695b-4c7a-8bd4-53d0bcc996e1}}), 사용 가중치 [candidate_weights_used.csv]({{artifact:4d17f14b-9219-4174-a58b-a0352cdffda2}}).

## 4. test 표 (전체·중복 제거)

| 분할 | 기준 | 후보 | n | MAE | RMSE | R² | 평균 잔차(예측−관측) |
|---|---|---|---|---|---|---|---|
| random | 전체 | A `wavg_mse\|noO2` | 1995 | 24.750 | 33.786 | 0.867 | -0.25 |
| random | 중복 제거 | A `wavg_mse\|noO2` | 1896 | 25.045 | 34.072 | 0.865 | -0.42 |
| random | 전체 | B `fixed_mean\|best+GIN` | 1995 | 24.755 | 33.847 | 0.867 | -0.53 |
| random | 중복 제거 | B `fixed_mean\|best+GIN` | 1896 | 24.991 | 34.066 | 0.865 | -0.68 |
| random | 전체 | C `xgb600_F0F5_M2` | 1995 | 26.509 | 35.627 | 0.853 | +0.98 |
| random | 중복 제거 | C `xgb600_F0F5_M2` | 1896 | 26.841 | 36.050 | 0.848 | +0.86 |
| random | 전체 | sandbox 재적합 기준선(n_jobs 8, seed 42) | 1995 | 27.633 | 36.950 | 0.841 | +0.75 |
| random | 중복 제거 | sandbox 재적합 기준선(n_jobs 8, seed 42) | 1896 | 27.940 | 37.337 | 0.837 | +0.69 |
| random | 전체 | 원본 예측 파일(수업 XGB+RDKit217) | 1995 | 27.500 | 36.877 | 0.842 | +0.68 |
| random | 중복 제거 | 원본 예측 파일(수업 XGB+RDKit217) | 1896 | 27.790 | 37.251 | 0.838 | +0.61 |
| scaffold | 전체 | A `wavg_mse\|noO2` | 1995 | 33.649 | 45.007 | 0.636 | -0.13 |
| scaffold | 중복 제거 | A `wavg_mse\|noO2` | 1984 | 33.627 | 44.957 | 0.636 | -0.06 |
| scaffold | 전체 | B `fixed_mean\|best+GIN` | 1995 | 33.987 | 45.475 | 0.629 | +0.47 |
| scaffold | 중복 제거 | B `fixed_mean\|best+GIN` | 1984 | 33.975 | 45.428 | 0.628 | +0.53 |
| scaffold | 전체 | C `xgb600_F0F5_M2` | 1995 | 34.924 | 46.468 | 0.612 | +1.56 |
| scaffold | 중복 제거 | C `xgb600_F0F5_M2` | 1984 | 34.896 | 46.436 | 0.612 | +1.60 |
| scaffold | 전체 | sandbox 재적합 기준선(n_jobs 8, seed 42) | 1995 | 35.852 | 47.408 | 0.597 | -0.19 |
| scaffold | 중복 제거 | sandbox 재적합 기준선(n_jobs 8, seed 42) | 1984 | 35.826 | 47.383 | 0.596 | -0.15 |

### 짝지은 부트스트랩 (2,000회)

| 분할 | 기준 | 비교 | n | ΔMAE [95 % CI] | 유의 |
|---|---|---|---|---|---|
| random | 전체 | A−sandbox | 1995 | -2.883 [-3.421, -2.354] | 예 |
| random | 전체 | A−원본 | 1995 | -2.750 [-3.294, -2.251] | 예 |
| random | 전체 | B−sandbox | 1995 | -2.878 [-3.453, -2.311] | 예 |
| random | 전체 | B−원본 | 1995 | -2.744 [-3.323, -2.222] | 예 |
| random | 전체 | C−sandbox | 1995 | -1.123 [-1.546, -0.687] | 예 |
| random | 전체 | C−원본 | 1995 | -0.990 [-1.420, -0.566] | 예 |
| random | 전체 | A−B | 1995 | -0.005 [-0.131, +0.123] | 아니오 |
| random | 전체 | B−C | 1995 | -1.754 [-2.250, -1.254] | 예 |
| random | 전체 | A−C | 1995 | -1.759 [-2.211, -1.320] | 예 |
| random | 중복 제거 | A−sandbox | 1896 | -2.894 [-3.448, -2.379] | 예 |
| random | 중복 제거 | A−원본 | 1896 | -2.745 [-3.315, -2.220] | 예 |
| random | 중복 제거 | B−sandbox | 1896 | -2.949 [-3.548, -2.407] | 예 |
| random | 중복 제거 | B−원본 | 1896 | -2.799 [-3.400, -2.250] | 예 |
| random | 중복 제거 | C−sandbox | 1896 | -1.098 [-1.512, -0.696] | 예 |
| random | 중복 제거 | C−원본 | 1896 | -0.948 [-1.386, -0.549] | 예 |
| random | 중복 제거 | A−B | 1896 | +0.054 [-0.071, +0.189] | 아니오 |
| random | 중복 제거 | B−C | 1896 | -1.850 [-2.390, -1.364] | 예 |
| random | 중복 제거 | A−C | 1896 | -1.796 [-2.274, -1.358] | 예 |
| scaffold | 전체 | A−sandbox | 1995 | -2.202 [-2.791, -1.614] | 예 |
| scaffold | 전체 | B−sandbox | 1995 | -1.864 [-2.518, -1.240] | 예 |
| scaffold | 전체 | C−sandbox | 1995 | -0.928 [-1.462, -0.413] | 예 |
| scaffold | 전체 | A−B | 1995 | -0.338 [-0.615, -0.058] | 예 |
| scaffold | 전체 | B−C | 1995 | -0.936 [-1.469, -0.428] | 예 |
| scaffold | 전체 | A−C | 1995 | -1.274 [-1.796, -0.765] | 예 |
| scaffold | 중복 제거 | A−sandbox | 1984 | -2.200 [-2.785, -1.609] | 예 |
| scaffold | 중복 제거 | B−sandbox | 1984 | -1.851 [-2.489, -1.197] | 예 |
| scaffold | 중복 제거 | C−sandbox | 1984 | -0.930 [-1.462, -0.400] | 예 |
| scaffold | 중복 제거 | A−B | 1984 | -0.348 [-0.632, -0.076] | 예 |
| scaffold | 중복 제거 | B−C | 1984 | -0.921 [-1.445, -0.372] | 예 |
| scaffold | 중복 제거 | A−C | 1984 | -1.269 [-1.789, -0.771] | 예 |

### seed 대응 조합

| 후보 | 분할 | seed 대응 조합 MAE 평균 ± SD (n 3) | 조합별 MAE | 멤버 seed |
|---|---|---|---|---|
| A | random | 24.959 ± 0.251 | 24.7224 / 25.2218 / 24.9341 | h06_yalk_F0F3F5:0,1,2 | gin_hybrid_seedavg:0,1,2 | lgb_r1_F0F5:0,1,2 | xgb600_F0F1O3:0,1,2 |
| A | scaffold | 33.862 ± 0.152 | 34.0236 / 33.8396 / 33.7218 | gin_hybrid_seedavg:0,1,2 | xgb600_F0_patent1x_scaf:42,43,44 | h06_yalk_F0F3F5:0,1,2 |
| B | random | 25.077 ± 0.268 | 24.8185 / 25.3543 / 25.0580 | lgb_r1_F0F3F5:0,1,2 | gin_hybrid_seedavg:0,1,2 |
| B | scaffold | 34.337 ± 0.187 | 34.5262 / 34.3317 / 34.1517 | lgb_r1_F0F3F5:0,1,2 | gin_hybrid_seedavg:0,1,2 |
| C | random | 26.770 ± 0.166 | 26.8162 / 26.9080 / 26.5855 | xgb600_F0F5_M2:42,0,1 |
| C | scaffold | 35.278 ± 0.335 | 35.2966 / 34.9339 / 35.6027 | xgb600_F0F5_M2:42,0,1 |

### MP 구간별 MAE (평균 잔차), 관측 y 기준

| 분할 | 후보 | <250 K | 250-350 K | 350-450 K | 450-550 K | >550 K |
|---|---|---|---|---|---|---|
| random | n | 176 | 530 | 832 | 398 | 59 |
| random | A | 25.6 (+17.7) | 21.1 (+9.1) | 23.0 (+2.6) | 29.1 (-19.4) | 49.7 (-49.5) |
| random | B | 24.7 (+16.7) | 21.0 (+9.0) | 23.1 (+2.2) | 29.3 (-19.3) | 50.1 (-49.9) |
| random | C | 29.3 (+21.9) | 23.2 (+11.9) | 24.1 (+4.2) | 30.9 (-21.8) | 51.8 (-51.8) |
| random | 원본 예측 파일(수업 XGB+RDKit217) | 30.5 (+22.5) | 24.3 (+12.3) | 24.9 (+3.9) | 31.5 (-22.5) | 57.1 (-56.9) |
| random | sandbox 재적합 기준선(n_jobs 8, seed 42) | 29.9 (+22.2) | 24.2 (+12.6) | 25.1 (+3.9) | 32.0 (-22.7) | 56.7 (-56.6) |
| scaffold | n | 20 | 277 | 974 | 625 | 99 |
| scaffold | A | 56.9 (+50.3) | 37.7 (+33.4) | 27.0 (+11.9) | 35.0 (-23.9) | 74.6 (-72.5) |
| scaffold | B | 57.0 (+50.5) | 36.7 (+32.0) | 28.0 (+12.4) | 35.0 (-22.2) | 74.4 (-71.9) |
| scaffold | C | 57.3 (+48.4) | 40.8 (+35.9) | 28.4 (+14.8) | 35.0 (-23.4) | 77.9 (-76.9) |
| scaffold | sandbox 재적합 기준선(n_jobs 8, seed 42) | 55.7 (+47.0) | 40.8 (+35.2) | 28.8 (+13.7) | 36.9 (-26.4) | 80.9 (-79.7) |

모든 후보·기준선이 끝 구간에서 평균 쪽으로 끌린다. 낮은 MP는 과대 예측되고(random < 250 K +16.7 ~ +22.5, scaffold +47 ~ +51 K), 높은 MP는 과소 예측된다(> 550 K random −49.5 ~ −56.9, scaffold −71.9 ~ −79.7 K). 앙상블은 > 550 K 쏠림을 기준선보다 7 K(random) / 7–8 K(scaffold) 줄인다.

![MP 구간별 평균 잔차]({{artifact:38c144b6-85dd-41be-955b-356fecf243af}})

표: [test_results.csv]({{artifact:99ef2679-f85b-477c-8153-08e427f6cc16}}), [bootstrap_ci.csv]({{artifact:3ef8fb40-ee0f-4e5d-9403-7fe5c1c43935}}), [ranks.csv]({{artifact:741e0a38-1b35-46be-9fec-d263dd845785}}), [seed_metrics_test.csv]({{artifact:105b3165-056c-451c-a0c8-225362150b15}}), [mp_bin_residuals.csv]({{artifact:f4fd30a5-5a23-4e9f-bbf7-a12b2f9e94da}}).

## 5. 목표 C (10 K 커버리지)

점수는 `metaA_trainOOF_self`(ENS 저장 메타모델, 재적합 안 함)다. test 입력은 train만 참조한 유사도와 멤버 test 예측으로 만들었다.

**random: validation 고정 τ를 한 번 적용**

| 후보 | τ [K] (validation 고정) | validation 커버리지 (n) / MAE | test 커버리지 (n) | test 부분집합 MAE [행 CI] | τ 고정 전체 부트스트랩: 커버리지 / MAE | ≤ 10 K? |
|---|---|---|---|---|---|---|
| A | 14.675 | 7.22 % (144) / 9.98 [8.55, 11.63] | 6.82 % (136) | 12.64 [10.75, 14.77] | [5.7, 8.0] % / [10.82, 14.69] | 아니오 |
| B | 14.614 | 7.12 % (142) / 9.99 [8.56, 11.52] | 6.57 % (131) | 12.68 [10.76, 14.79] | [5.5, 7.7] % / [10.79, 14.82] | 아니오 |
| C | 13.337 | 5.46 % (109) / 9.88 [8.27, 11.62] | 4.51 % (90) | 12.91 [10.92, 15.09] | [3.7, 5.5] % / [10.82, 15.15] | 아니오 |

- 판정: **미달.** 세 후보 모두 test 부분집합 MAE가 12.6–12.9 K이고 CI 하한도 10.75 K를 넘는다.
  - ENS의 교차 검증 임계값 추정(A 10.06 K, 반복의 50 %만 ≤ 10 K)이 이미 경계라고 경고했다.
  - test는 그보다 더 나빴다.
  - 커버리지 자체는 validation과 비슷하다(A 6.8 % 대 7.2 %).
- 같은 점수 순위의 커버리지별 MAE(test)는 아래와 같다. scaffold는 τ를 고정하지 않았으므로 참고용이다.

| 분할 | 후보 | MAE@5 % | @10 % | @20 % | @50 % | 전체 | AURC |
|---|---|---|---|---|---|---|---|
| random | A | 11.70 | 13.16 | 15.01 | 19.44 | 24.75 | 18.48 |
| random | B | 11.70 | 13.00 | 15.07 | 19.56 | 24.76 | 18.55 |
| random | C | 12.89 | 14.86 | 16.61 | 20.71 | 26.51 | 19.95 |
| scaffold | A | 24.72 | 25.01 | 28.04 | 29.91 | 33.65 | 29.48 |
| scaffold | B | 24.18 | 24.49 | 27.95 | 30.01 | 33.99 | 29.47 |
| scaffold | C | 26.33 | 26.82 | 29.47 | 31.37 | 34.92 | 30.81 |

- PREREG 7절 부분집합(test): 세 후보·두 분할 어느 부분집합도 MAE ≤ 10 K가 아니다. 가장 낮은 것은 random nn1 ≥ 0.9(n 183) A 15.21 K, 정제판 DPG(n 322) A 18.70 K다.

| 분할 | 부분집합 | n | A MAE [CI] | B MAE [CI] | C MAE [CI] |
|---|---|---|---|---|---|
| random | 품질1 정제판(DPG) | 322 | 18.70 [17.03, 20.45] | 18.66 [17.03, 20.38] | 20.62 [18.82, 22.49] |
| random | 품질5 측정 1개 | 1488 | 25.52 [24.35, 26.65] | 25.53 [24.35, 26.68] | 27.34 [26.09, 28.63] |
| random | nn1≥0.9 | 183 | 15.21 [13.11, 17.56] | 15.64 [13.52, 17.95] | 17.42 [15.38, 19.57] |
| random | nn1≥0.7 | 666 | 21.60 [20.14, 23.10] | 21.68 [20.20, 23.21] | 23.39 [21.94, 24.93] |
| random | flag:sym2d_high | 221 | 30.27 [26.87, 33.76] | 30.33 [26.91, 33.79] | 32.88 [29.02, 36.78] |
| random | MP구간(예측) 250-350 | 526 | 21.98 [20.17, 23.91] | 21.83 [19.91, 23.74] | 23.28 [21.31, 25.44] |
| random | MP구간(y) >550 | 59 | 49.66 [42.14, 57.73] | 50.15 [42.70, 58.10] | 51.75 [43.87, 59.84] |
| scaffold | 품질1 정제판(DPG) | 122 | 32.54 [27.79, 37.41] | 33.16 [28.47, 38.15] | 33.58 [28.84, 38.89] |
| scaffold | 품질5 측정 1개 | 1750 | 33.96 [32.52, 35.37] | 34.25 [32.81, 35.67] | 35.26 [33.82, 36.73] |
| scaffold | nn1≥0.9 | 33 | 23.90 [17.39, 32.05] | 25.20 [17.84, 34.15] | 26.50 [18.91, 36.49] |
| scaffold | nn1≥0.7 | 151 | 30.62 [26.51, 34.96] | 30.76 [26.61, 35.23] | 32.55 [28.36, 37.19] |
| scaffold | flag:sym2d_high | 338 | 38.85 [35.16, 42.59] | 39.35 [35.63, 43.13] | 41.56 [37.66, 45.63] |
| scaffold | MP구간(예측) 250-350 | 168 | 31.75 [27.26, 36.39] | 30.42 [26.07, 35.02] | 35.13 [30.47, 40.20] |
| scaffold | MP구간(y) >550 | 99 | 74.58 [64.30, 85.24] | 74.37 [64.03, 84.96] | 77.93 [68.15, 88.66] |

![위험–커버리지: test(실선)와 validation(점선)]({{artifact:af0cb8a6-add1-441b-9356-ecb8b8d6cff4}})

표: [goalC_test.csv]({{artifact:ebf97c4c-5089-4b76-a541-e0ae1131b42c}}), [goalC_curve_test.csv]({{artifact:ad8d59a2-41c7-4c72-8650-a41592f3eec2}}), [goalC_subsets_test.csv]({{artifact:12eb20aa-a592-4690-a340-ba5717a10416}}). 분자별 점수: [goalC_score_random_test.csv]({{artifact:cf423875-ce9a-45f1-8af2-37c6b20d656a}}), [goalC_score_scaffold_test.csv]({{artifact:4e3170fe-26ce-48b4-8ba3-bfc44dc3c1a9}}).

## 6. 보조: train+validation 재학습 (주 결과 저장 뒤)

규칙(명세)은 다음과 같다.

- 나무 수 = best_iter × n_fit / n_inner로 정했다. lgb_r1은 random 4,150 / scaffold 3,350그루, h06은 random 5,162 / scaffold 3,717그루, BASE_XGB는 600그루 고정이다.
- 가중치는 validation 값으로 고정했다. GIN은 train+validation 안 10 % 내부 조기 종료를 썼다.
- 학습 행은 17,950개(특허 멤버는 + 15,956)다.
- 순서는 지시대로 C → B의 LGB → A의 GBDT → GIN이다. GIN 시작 시 남은 예산이 7.2 core-h(≥ 5)여서 GIN도 했다. 모두 실시했다.

| 후보 | 분할 | 변형 | test MAE 전체 (중복 제거) | train 전용 주 결과 | Δ vs 주 결과 [CI] | Δ vs 원본 기준 [CI] |
|---|---|---|---|---|---|---|
| C | random | GBDT만 trainval | 26.191 (26.510) | 26.509 | -0.32 [-0.57, -0.06] | -1.31 [-1.75, -0.85] |
| C | scaffold | GBDT만 trainval | 33.930 (33.963) | 34.924 | -0.99 [-1.59, -0.41] | – |
| B | random | GBDT만 trainval | 24.437 (24.671) | 24.755 | -0.32 [-0.43, -0.20] | -3.06 [-3.65, -2.52] |
| B | random | 전부 trainval | 24.529 (24.771) | 24.755 | -0.23 [-0.51, +0.08] | -2.97 [-3.55, -2.41] |
| B | scaffold | GBDT만 trainval | 33.105 (33.132) | 33.987 | -0.88 [-1.22, -0.55] | – |
| B | scaffold | 전부 trainval | 32.582 (32.587) | 33.987 | -1.41 [-1.94, -0.86] | – |
| A | random | GBDT만 trainval | 24.383 (24.694) | 24.750 | -0.37 [-0.50, -0.23] | -3.12 [-3.68, -2.60] |
| A | random | 전부 trainval | 24.482 (24.797) | 24.750 | -0.27 [-0.54, -0.00] | -3.02 [-3.58, -2.49] |
| A | scaffold | GBDT만 trainval | 32.873 (32.887) | 33.649 | -0.78 [-1.09, -0.47] | – |
| A | scaffold | 전부 trainval | 32.503 (32.505) | 33.649 | -1.15 [-1.62, -0.69] | – |

- 모든 후보가 train 전용 결과보다 좋아졌다. random은 −0.23 ~ −0.37 K다. GIN까지 바꾼 변형은 B의 CI가 0을 포함하고, A는 상한 −0.005로 겨우 0 아래다. scaffold는 −0.78 ~ −1.41 K로 유의하다.
- 1단계 학습 곡선이 예측한 train+val 이득은 random −0.29, scaffold −0.17 K였다(T1 O10). scaffold 이득은 예측보다 크다. validation 분자가 새 골격을 더하기 때문으로 보인다(해석, 미검증).
- random에서 GIN도 train+val로 바꾸면 GBDT만 바꾼 경우보다 오히려 0.1 K 나쁘다(A 24.482 대 24.383, B 24.529 대 24.437). 이 차이는 seed 잡음 범위다.
- 이 보조 결과는 같은 test를 두 번째로 쓴 것이다. 주 결과 순위·선정에는 쓰지 않는다.

표: [aux_trainval_test.csv]({{artifact:59a80189-0e94-4930-80c4-035944e202b0}}), sha256 [test_pred_sha256_trainval.csv]({{artifact:26e838f2-5155-4fd0-8738-4eb79ed97858}}).

## 7. 비용 (CostMeter 실측, ledger)

| 단계 | 실행 수 | CPU core-h | wall h | 에너지 kWh (25 W) | 최대 RAM GB |
|---|---|---|---|---|---|
| GBDT train 전용 (주) | 41 | 3.822 | 1.10 | 0.0080 | 0.72 |
| GBDT train+val (보조) | 27 | 2.821 | 0.75 | 0.0059 | 0.79 |
| GIN train 전용 (주) | 6 | 4.019 | 4.24 | 0.0084 | 1.22 |
| GIN train+val (보조) | 6 | 4.190 | 4.32 | 0.0087 | 1.22 |
| GIN 그래프 생성 | 1 | 0.026 | 0.03 | 0.0001 | 1.06 |
| O2 kNN·유사도 | 1 | 0.016 | 0.01 | 0.0000 | 0.71 |
| 누설 감사 | 1 | 0.101 | 0.10 | 0.0002 | 1.20 |
| 평가 | 2 | 0.008 | 0.01 | 0.0000 | 0.09 |
| **합계** | 85 | **15.001** | – | 0.0313 | – |

- 이 단계 합계는 **15.00 core-h**로 예산 18 core-h 안이다.
  - 주 결과(재학습 + 감사 + 평가)는 7.99 core-h, 보조 train+val은 7.01 core-h다.
  - 이전 단계 약 69 core-h를 더하면 프로젝트 누계는 약 84 / 90 core-h다.
  - vCPU 단가 0.0504 USD/core-h로 환산하면 이 단계는 0.76 USD다. 에너지는 0.031 kWh(25 W 가정)다.
  - API·GPU는 쓰지 않았다.
- 후보별 최종 재적합(train 전용, 두 분할, 모든 seed)의 실측 비용은 다음과 같다.
  - A: GIN 4.02 + h06 0.87 + lgb_F0F5 0.37 + F0F1O3 0.29 + 특허 0.06 = 5.60 core-h.
  - B: GIN 4.02 + lgb_F0F3F5 0.72 = 4.74 core-h.
  - C: 0.095 core-h.
  - 명세 예상(6.2 / 5.0 / 0.08)과 비슷하다.
- 원장: [ledger.csv]({{artifact:de582909-e69b-4f90-bc0c-14185f6d7e98}})(mpcommon.LEDGER_COLS, 85줄).

## 8. 파일

- 감사: [leakage_audit.md]({{artifact:bd79f6f1-38dc-4663-b709-8cb3597dc5e3}}).
- test 예측: 후보·멤버별 `row, y, pred, pred_s*`(seed 평균 + seed별)이다.
  - 후보: [A_wavg_mse_noO2_random_test.csv]({{artifact:69339f25-3086-4221-925b-cfff26391f62}}), [A_wavg_mse_noO2_scaffold_test.csv]({{artifact:edf0ba3c-2feb-40d3-a8e5-62f3fc78eac2}}), [B_fixed_mean_best_GIN_random_test.csv]({{artifact:2343268a-39ac-40a6-8335-169c4cb862cd}}), [B_fixed_mean_best_GIN_scaffold_test.csv]({{artifact:6c548f3d-a2d1-4d4d-a23d-aecef48489c8}}), [C_xgb600_F0F5_M2_random_test.csv]({{artifact:ebaebeb0-1f2f-4419-8492-c0743a529a3a}}), [C_xgb600_F0F5_M2_scaffold_test.csv]({{artifact:167bbf61-186b-48a0-8476-8158820e6138}}).
  - 멤버: [gin_hybrid_seedavg_random_test.csv]({{artifact:a10b6489-3b24-43af-9c65-1464232bc9c4}}), [gin_hybrid_seedavg_scaffold_test.csv]({{artifact:7f268a78-d06e-451b-a381-57a6827bef2d}}), [h06_yalk_F0F3F5_random_test.csv]({{artifact:2e6ec59b-f1c3-4da1-9705-279321e06115}}), [h06_yalk_F0F3F5_scaffold_test.csv]({{artifact:d9320414-eea4-4197-a544-1d112825173b}}), [lgb_r1_F0F3F5_random_test.csv]({{artifact:28dfa3cb-6045-4a4d-aae4-bb00e56b9b31}}), [lgb_r1_F0F3F5_scaffold_test.csv]({{artifact:b73ed1d3-bd2f-484d-8d21-988c11e5389b}}), [lgb_r1_F0F5_random_test.csv]({{artifact:d2c7ee95-a133-4b41-9f0c-15b6994cb04b}}), [xgb600_F0F1O3_random_test.csv]({{artifact:8d924ab4-f498-4469-9796-a8c2d1215938}}), [xgb600_F0_patent1x_scaf_scaffold_test.csv]({{artifact:a2970a3b-c796-4389-965e-34a74d6dae5c}}), [xgb600_F0F5_M2_random_test.csv]({{artifact:607d3e4e-b622-4e88-83e8-4f4667b07ed2}}), [xgb600_F0F5_M2_scaffold_test.csv]({{artifact:82ce63be-c336-4f30-9611-57b9ce2712f2}}), [baseline_xgb600_F0_random_test.csv]({{artifact:a90c85ec-2ac6-4df8-b4e3-103ab591d1f2}}), [baseline_xgb600_F0_scaffold_test.csv]({{artifact:f8e6e7e5-22fc-4406-b02d-e6f49ceeb835}}), [lgb_r1_F0_random_test.csv]({{artifact:e0106b0f-9c56-4347-b2c5-74202ba83c4c}}), [lgb_r1_F0_scaffold_test.csv]({{artifact:3cd784fe-8f18-4163-b18f-a0d0c5d5ef65}}), [xgb_r1_F0_random_test.csv]({{artifact:63461bcd-a98a-47b6-969d-d01178158dd7}}), [xgb_r1_F0_scaffold_test.csv]({{artifact:1e307ebe-ed7a-4609-90ac-263e7f983325}}), [O2_knn_random_test.csv]({{artifact:5ba2517d-670c-41d7-b606-c90c67979b7f}}), [O2_knn_scaffold_test.csv]({{artifact:7a81d242-f09c-465d-9028-8b39c956c270}}).
  - sha256 목록(평가 전 기록, y 열 없는 파일 기준): [test_pred_sha256_main.csv]({{artifact:4630ec14-650d-46b2-9583-6abd8aa7c432}}).
- validation 재예측: [A_wavg_mse_noO2_random_val.csv]({{artifact:966effea-cfd2-4c88-ab30-e89d801bed3c}}), [A_wavg_mse_noO2_scaffold_val.csv]({{artifact:effd8b37-20fc-4184-8c9c-7c829a20a896}}), [B_fixed_mean_best_GIN_random_val.csv]({{artifact:2c08502c-76b2-482d-b3a5-53c811729213}}), [B_fixed_mean_best_GIN_scaffold_val.csv]({{artifact:67cf45f7-9a51-4d36-9a57-ed334f2a4cc2}}), [C_xgb600_F0F5_M2_random_val.csv]({{artifact:544d9d4a-82ae-4caf-8ca3-87d60d6960b4}}), [C_xgb600_F0F5_M2_scaffold_val.csv]({{artifact:f61e21f9-0c43-4689-98ac-a1e97df20f13}}), [gin_hybrid_seedavg_random_val.csv]({{artifact:b8c369d4-79ab-4dab-b836-78c157f2f051}}), [gin_hybrid_seedavg_scaffold_val.csv]({{artifact:14121c82-b50d-4f8e-bcaf-005c4f9c8718}}) 외 멤버별 파일.
- 보조 예측: [A_wavg_mse_noO2__allTV_random_test.csv]({{artifact:3df48faf-d49b-48b6-bec7-1b7aaaa00f19}}), [A_wavg_mse_noO2__allTV_scaffold_test.csv]({{artifact:01afc7f1-6396-449d-9575-ad9e34a5a405}}), [A_wavg_mse_noO2__gbdtTV_random_test.csv]({{artifact:fe6dcc6e-a072-4e77-b498-5c0ce43abf5e}}), [A_wavg_mse_noO2__gbdtTV_scaffold_test.csv]({{artifact:549d9962-7778-4941-b635-b0be770ffe27}}), [B_fixed_mean_best_GIN__allTV_random_test.csv]({{artifact:f54f28d4-f7ab-4dbf-a272-633e2a9d75c8}}), [B_fixed_mean_best_GIN__allTV_scaffold_test.csv]({{artifact:e36749e1-2410-4e49-a4b0-dd396b5ed4ae}}), [B_fixed_mean_best_GIN__gbdtTV_random_test.csv]({{artifact:977cdd8f-dc93-4ab1-9d4b-2495810fcfe8}}), [B_fixed_mean_best_GIN__gbdtTV_scaffold_test.csv]({{artifact:0e26ac2e-911d-4d39-b0cf-0ea4f9d76f84}}), [C_xgb600_F0F5_M2__gbdtTV_random_test.csv]({{artifact:b7e2eda3-3d7a-4692-9a1e-c2d4769b3dd3}}), [C_xgb600_F0F5_M2__gbdtTV_scaffold_test.csv]({{artifact:bf70ebf4-bf4d-4296-b385-025ee878ae4f}}).
- 스크립트(작업 공간 루트에서 순서대로 실행):
  1. [s0_audit_pre.py]({{artifact:5d33ee8c-b987-49a2-be3a-20af33903fec}})
  2. [s1_gbdt.py]({{artifact:e8678c58-aa51-4dd6-ae65-bb68a28f12ac}})
  3. [gin_p3.py]({{artifact:73bd9187-d1d9-42f4-a4a8-e44419d54109}})
  4. [s2_knn.py]({{artifact:a4cc5f1e-2033-4a2c-9f9a-9ee5584b52dd}})
  5. [s3_repro.py]({{artifact:ea67bc89-c5ef-4ab7-8505-c15abe9f86a3}})
  6. [s4_goalc.py]({{artifact:841b285a-9244-45bb-9c9c-c1c90c98abea}})
  7. [s5_candidates.py]({{artifact:9fd49e31-29de-482f-a9cf-9d90440e7323}})
  8. [s9_evaluate.py]({{artifact:11903919-16be-46e4-9eed-5d6094e3bede}})
  9. [s10_figs.py]({{artifact:dacc3ded-5a5b-4e58-bedf-0045744c528c}})
  10. [s11_aux_eval.py]({{artifact:a2b120f3-bf50-4c48-a127-337d47301562}})
  - 공통 모듈은 [p3lib.py]({{artifact:957c634b-748b-4a00-a7de-5e264fc2d02d}})다.
