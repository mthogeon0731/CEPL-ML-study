# 누설 감사 (지시서 9절 5개 항목) — 3단계 P3

작성 순서: 항목 i–iv는 **test 라벨을 읽기 전에** 끝내고 이 문서에 적었다(스크립트 `s0_audit_pre.py`, 0.10 core-h). 항목 v는 test 1회 평가 뒤에 덧붙였다. 표 파일은 `P3/tables/audit_*.csv`, 수치 원본은 `P3/notes/audit_pre.json`.

## 판정 요약

| 항목 | 내용 | 판정 |
|---|---|---|
| i | train·validation·test(외부 특허 데이터 포함) InChIKey 전체·첫 블록 겹침 | **조건부 통과** — 겹침은 0이 아니지만(1단계부터 알려진 사실), 전부 중복 제거 목록에 정확히 반영되어 있고 중복 제거 test(n 1,896 / 1,984)를 함께 보고한다. 외부 특허·LLM 표본은 겹침 0 |
| ii | 특징에 라벨 유래 정보 없음 | **통과** |
| iii | 스케일러·특징 선택·하이퍼파라미터·앙상블 가중치·τ가 test 없이 정해짐 | **통과(단서 1건)** — 1단계 학습 곡선 진단 24행에 기준선 설정의 test MAE가 기록되어 있으나 어떤 선택에도 쓰이지 않음 |
| iv | T8 기억 누설 점검 결과 | **통과** — 최종 후보에 분자별 LLM 호출 없음, F5는 고정 RDKit 코드, 규칙 생성 프롬프트에 test 분자·라벨 없음 |
| v | 문헌 대비 이상치 | test 평가 뒤 아래 5절에 기록 |

세 후보(A·B·C) 모두 i–iv를 통과했다.

## 1. InChIKey 겹침 (항목 i)

- 19,945개 SMILES에서 InChIKey를 RDKit로 다시 계산했다. 분할 파일의 키와 불일치 0, 빈 키 0이다. 분할 파일 SMILES는 `mp.csv`와 행 순서까지 같다.
- 쌍별 겹침([audit_i_inchikey_overlap.csv]):

| 분할 | 쌍 | 전체 키 공유 (고유 키 수) | 첫 블록 공유 (고유 블록 수) | 뒤 세트에서 첫 블록이 겹치는 행 |
|---|---|---|---|---|
| random | train–validation | 6 | 71 | 73 |
| random | train–test | 8 | 89 | 92 |
| random | validation–test | 0 | 12 | 14 |
| scaffold | train–validation | 7 | 8 | 8 |
| scaffold | train–test | 5 | 6 | 6 |
| scaffold | validation–test | 6 | 6 | 6 |

- test 행 중 첫 블록이 train ∪ validation과 겹치는 행은 random 99개, scaffold 11개다. 다시 계산한 표지가 중복 제거 목록(`dedup_rows_*.csv`의 overlap=1)과 **행 단위로 일치**한다. 따라서 중복 제거 test는 random n 1,896, scaffold n 1,984다. validation도 같은 정의로 일치한다(73 / 8행, 중복 제거 1,922 / 1,986).
- scaffold 분할에도 전체 키가 같은 분자가 세트 사이에 있다(train–test 5개). SMILES 표기가 다르지만 InChIKey가 같은 분자(입체·호변이성 표기 차이)로 보인다. 모두 중복 제거 목록에 들어 있다.
- **외부 특허 정제본(194,801분자)**: SMILES에서 키를 다시 계산했다(파일 키와 불일치 0). 두 분할의 train·validation·test 어느 세트와도 **전체 키 0, 첫 블록 0**이다. 특허 멤버(후보 A scaffold의 `xgb600_F0_patent1x_scaf`)가 test 분자를 학습에 쓰지 않았음을 직접 확인했다.
- **T8 규칙 생성 때 LLM에 보인 200분자**: 200개 모두 두 분할의 train에 있다. SMILES가 mp.csv와 같고, 어느 분할의 validation·test와도 첫 블록 겹침 0이다.
- 판정 근거: 지시서는 '겹침 0'을 요구하지만 고정 분할 자체가 겹침을 갖고 있다. 1단계부터 이를 중복 제거 test라는 보조 지표로 다루기로 했다. 그래서 '조건부 통과'로 적고, 모든 test 결과를 전체·중복 제거 두 기준으로 낸다.

## 2. 특징에 라벨 유래 정보 없음 (항목 ii)

| 특징 | 쓰는 후보·멤버 | 확인 방법 | 결과 |
|---|---|---|---|
| F0 RDKit 기술자 217 | 전부 | test 행 150개를 SMILES에서 다시 계산해 캐시와 대조 | 최대 차이 0.0 |
| F5 LLM 규칙 101열 | A, B, C, h06 | 같은 150행을 `f5_rules.featurize`로 다시 계산 | 최대 차이 0.0 |
| F3 물리·결정 39열 | A(h06), B | 25행을 `phys_feats.featurize_smiles(n_conf 5)`로 다시 계산(2D·3D 열) | 최대 차이 0.0 (배좌 seed 고정이라 3D 열도 같음) |
| F1 Morgan count, O3 입체 | A random(xgb600_F0F1O3) | 150행 다시 계산 | 최대 차이 0 |
| GIN 그래프(원자 147) | A, B | 수업 `feature.py` 정적 점검 | import는 numpy·rdkit·torch뿐, 파일 읽기·라벨 참조 없음 |

- 정적 점검([audit_ii_static_code.csv]): 특징 코드(`phys_feats.py`, `f5_rules.py`, `feature.py`)는 파일을 읽지 않는다. `s2_features.py`(O3·F1)의 '라벨 참조' 4건은 docstring 1줄, 주석 1줄, mp.csv를 읽어 열 이름·행 수를 출력하는 2줄이다. O3·F1 값은 SMILES만으로 계산한다(`o3_row(smi)`, Morgan count). 라벨 사용은 아니다. 라벨 열·외부 MP 값·LLM 호출을 참조하지 않는다. `train_xgb.py`의 라벨 참조는 `load()`가 y를 돌려주는 부분뿐이고, `featurize()`는 분자만 받는다.
- train 행 라벨–특징 상관([audit_ii_label_corr_train.csv], random train n 15,955): 묶음별 최대 |Pearson r|는 F0 0.567, F3 0.519, F5 0.519, F1 0.528, O3 0.194다. |r| > 0.9인 열은 0개다.
- F3 39열은 모두 구조 기술자다(σ, τ, 수소결합, 고리, 3D 모양 등). 보조 물성 예측(H17: logS·logP·밀도·증기압·끓는점) 열이 없다. 보조 물성 특징을 쓴 `xgb600_F0aux4_scaf`는 어느 최종 후보에도, 목표 C 멤버에도 들어 있지 않다.
- 스케일러: GIN 기술자 표준화는 train 행으로, y 스케일러는 학습 90 %로 fit한다. GBDT는 스케일러가 없다.
- 목표 C 점수의 입력:
  - O2 kNN과 유사도는 train 행만 참조한다. test 유사도도 train 행만 이웃으로 썼다(validation 행 미참조).
  - 메타모델은 ENS가 train OOF로 학습한 파일을 그대로 썼다.
  - `n_meas`(반복측정 수)는 Bradley 원본의 측정 기록 수다. 라벨 값 자체는 아니지만 test 분자의 출처 메타데이터를 쓴다. 목표 C 선택적 예측에만 들어가고 점예측에는 들어가지 않는다(단서).
- 학습·예측 스크립트는 `y_masked(how)`(해당 분할 test 행 = NaN)만 받았다. test 라벨이 학습에 들어가면 NaN 때문에 학습이 실패하거나 assert에 걸린다.

## 3. test 없이 정해졌는가 (항목 iii)

- 13개 ledger 아티팩트와 호스트 폴더 ledger 11개를 모두 읽었다([audit_iii_ledgers.csv]).
- test_MAE·test_RMSE·test_R2 열이 채워진 행은 1단계 ledger에만 있다.
  - `p1_baseline_random`, `p1_baseline_scaffold`: 기준선 재현 2행.
  - `errdecomp_s2_lc_*` 24행: 1단계 학습 곡선 진단이다. 고정 기준선 설정을 train 10–100 %로 적합했다.
  - 2단계 트랙(T1, T1b, T2, T3, T4, T5, T8, T8 n3_representative, ENS)의 ledger는 test 열이 모두 비어 있다.
- 단서: 학습 곡선 진단 24행은 '1단계 기준선 행'보다 넓다.
  - 1단계 요약은 test 사용 범위를 두 곳으로 적었다: (a) 고정 test 학습 곡선, (b) 근접 중복이 기준선 test MAE에 주는 영향.
  - 이후 트랙이 쓴 학습 곡선 외삽(`learning_curve_extrap.csv`, T5 같은 n 비교, T1 O10)은 metric = `val_MAE` 행이다.
  - 학습 곡선은 하이퍼파라미터·특징·가중치·τ 선택에 쓰이지 않았다.
  - 판정: 통과(단서 기록).
- phase2 스크립트에서 test 행을 참조하는 곳은 4곳뿐이다([audit_iii_script_test_refs.csv]).
  - `prereg/05_preregister.py` 3곳: 중복 제거 목록을 만들 때 키 겹침만 세고 라벨은 쓰지 않는다.
  - `T4T5/step3_aux.py` 1곳: 보조 물성 학습에서 validation·test 분자를 **빼는** 누설 차단이다.
- 이 단계의 순서(지시대로):
  1. 모든 멤버를 재학습했고, validation 재현을 점검했다([repro_val.csv]).
  2. test 예측 29개 파일을 라벨 없이 저장하고 sha256을 기록했다([test_pred_sha256_main.csv]).
  3. 평가 스크립트 `s9_evaluate.py`가 sha256을 다시 확인한 뒤 test 라벨을 처음 읽는다.
  - 가중치는 명세 CSV 값, τ는 ENS validation 고정값(A 14.675, B 14.614, C 13.337 K)이다. 메타모델 파일은 ENS 저장본이다.

## 4. T8 기억 누설 (항목 iv)

- 최종 후보 A·B·C에는 분자별 LLM 호출이 없다. LLM 서술(N3), 검색 few-shot(M4), zero-shot은 어느 후보·멤버에도 없다.
- F5(101열)는 `f5_rules.py`의 고정 RDKit 규칙이다. import는 math·numpy·rdkit뿐이고 LLM 호출 코드가 없다. test 행에서 다시 계산한 값이 캐시와 같다(위 2절).
- 규칙 생성 때 LLM(claude-opus-5-5, 2회)에 들어간 데이터는 다음과 같다.
  - (i) 문헌 지식 요청은 분자 없이 했다.
  - (ii) 데이터 규칙 요청에는 train 200분자(SMILES + MP)를 보였다.
  - 200분자는 두 분할 모두 train이고, validation·test와 첫 블록 겹침이 0이다. 이 단계에서 직접 다시 계산해 확인했다(위 1절).
- T8 README의 단서(문헌 규칙 프롬프트의 예시가 1단계 train OOF 오차 분해를 참고함)는 train 정보다. test 정보는 아니다.
- T8 기억 누설 판정(README 12.4) 요약:
  - LLM은 유명 분자의 녹는점 일부를 기억한다(zero-shot 회상 신호 있음).
  - N3 서술 경로의 이득으로는 이어지지 않았다(N3는 N2보다 나쁨).
  - M4는 '기억 효과 증거 없음'이지만 검정력이 약하다. 그래서 test 후보에서 뺐다.
  - 기술자 계열 M2(F5)는 random 유명 층에서 I −0.86 [−1.72, −0.02] K로 조금 더 좋았다. 같은 조건의 scaffold 유명 층에서는 I +0.48 [−0.47, +1.48] K로 차이가 없었다.
  - F5가 분자별 LLM 출력이 아니라 고정 코드이므로, 이 층 차이를 기억 누설로 보지 않는다.

## 5. 문헌 대비 이상치 (항목 v) — test 1회 평가 뒤 작성

- 안전 규칙: 평가 스크립트는 후보 test MAE가 20 K 미만이면 결과 표를 쓰지 않고 멈추도록 짰다. 발동하지 않았다. 가장 낮은 값은 후보 A random 24.750 K다.
- test 결과와 문헌 범위의 비교는 아래와 같다. 문헌 범위는 1단계 문헌표 기준이다: Bradley/Tetko 계열 MAE 약 25–33 K, 같은 Bradley 약 2만 분자에서 RMSE 35.4 °C(Austermeier 2025), 23만 분자 모델 MAE 21.5–23 °C(Pu 2026, Huang 2025).

| 후보 | random MAE / RMSE (n 1,995) | 중복 제거 MAE (n 1,896) | 중복 제거 − 전체 | scaffold MAE (n 1,995) |
|---|---|---|---|---|
| A | 24.750 / 33.786 K | 25.045 | +0.30 | 33.649 |
| B | 24.755 / 33.847 | 24.991 | +0.24 | 33.987 |
| C | 26.509 / 35.627 | 26.841 | +0.33 | 34.924 |
| 원본 기준선 | 27.500 / 36.877 | 27.790 | +0.29 | – |

- 판정: **이상치 아님(통과).**
  - random A·B는 문헌 범위 하단(25 K) 바로 아래다. 같은 크기 데이터의 RMSE(35.4 °C)보다 1.6 K 낮다. 23만 분자 모델(21.5–23 °C)보다는 높다.
  - 중복 제거 전후 차이(+0.24 ~ +0.33 K)는 기준선(+0.29 K)과 같은 크기다. 근접 중복에서 후보만 유독 이득을 보는 현상이 없다.
  - scaffold(33.6–34.9 K)는 범위 상단에 있다.
- 결론: 다섯 항목 모두에서 A·B·C를 결과로 쓸 수 있다. 항목 i은 조건부, iii은 단서 1건을 붙인다.
