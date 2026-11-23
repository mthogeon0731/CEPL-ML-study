# mp(녹는점) 예측 오차 끌어내리기 — Claude Science 작업 지시서 (2026-11-20)

> 이 파일을 끝까지 읽고 따라 주세요.
> **1단계(진단·방법론 설계)를 마치면 실행 계획을 Plan으로 제출하고 승인을 기다립니다.** 승인 뒤 2단계는 Delegation(병렬 트랙)으로 진행합니다.
> 사용자가 추가한 요청 세 가지는 5–7절에 따로 정리했습니다: ① 데이터셋 바꾸기(T7) ② 자연어·LLM 데이터셋(T8) ③ CPU/GPU 비용 대비 효과.

---

## 0. 목표

사용자 목표는 **test MAE ≤ 10 K**입니다. 다만 2절의 문헌·기존 결과를 보면, 전체 random test에서 10 K는 현재 문헌 수준(RMSE 32–35 °C ≈ MAE 24–26 K)보다 2배 이상 좋은 값이라 누설 없이 도달할 가능성이 낮습니다. 그래서 목표를 네 층으로 둡니다.

| 층 | 정의 | 판정 |
|---|---|---|
| A. 주 목표 | 전체 random test(1,995개) MAE를 기준선 **27.50 K**보다 유의하게 낮춤 | 같은 test에서 짝지은 부트스트랩(≥ 2,000회) 95 % CI가 0을 넘지 않음. 참고 팀 최저 25.95 K(분할 인스턴스 다름)도 참고선 |
| B. scaffold | 같은 방법을 scaffold test에서도 보고 | 이기든 지든 보고 |
| C. 10 K 도전 | "어디까지 10 K가 되는가": ① 위험–커버리지 곡선에서 MAE ≤ 10 K를 만족하는 최대 커버리지(신뢰도 기준은 validation에서 정하고 test는 1회), ② 화학 계열·MP 구간별로 10 K 이하인 영역 | 커버리지·부분집합 정의는 test를 보기 전에 고정 |
| D. 비용 효율 | 정확도만이 아니라 CPU/GPU·API 비용까지 계산해 가장 경제적이면서 효과적인 방법 추천 | 7절의 파레토 그림과 결정 규칙 |

**전체 test MAE가 20 K 아래로 나오면 먼저 누설을 의심**하고, 9절 감사를 통과하기 전에는 결과로 쓰지 않습니다.
(예: Austermeier 2025 — Bradley 문헌 세트 2,886개에서 학습 데이터와 겹치는 분자를 지우기 전 RMSE 19.0 °C → 지운 뒤 32.4 °C.)

---

## 1. 작업 환경과 자산

루트: `C:\Users\mthog\CEPL ML\` (이 폴더에 Read & write 권한 필요).
**기존 파일은 수정하지 말고**, 새 결과는 모두 `mp_mae_push\` 아래에 둡니다.

| 경로 | 내용 |
|---|---|
| `261109_Week2_Practice\data\mp.csv` | 19,945행, 열 `smiles,value,unit,source`, 단위 K, 68–790 K. Bradley Open MP. 같은 canonical SMILES는 **중앙값**으로 병합됨 |
| `261109_Week2_Practice\data\README_data.md` | 출처·정제 기록. 반복측정 4,679종 중 410종은 값 차이 ≥ 10 K, 86종은 ≥ 50 K. 같은 폴더에 bp·logPvap·density·logSolubility 등 다른 물성 CSV도 있음(T7 다중 작업용) |
| `261109_Week2_Practice\train_xgb.py` | 기준선. `split()` = 수업 `datamanager.split_dataset`과 같은 로직(80:10:10, seed 42; scaffold는 작은 Murcko 뼈대 그룹부터 test로 → 사슬형·벤젠처럼 큰 그룹은 전부 train). `run('mp','random','desc',max_n=None)` → **MAE 27.50 K, RMSE 36.88 K, R² 0.842, 학습 6.2 s** (XGBoost 600그루, lr 0.05, depth 6, subsample 0.8, colsample 0.5, RDKit 기술자 217개, train만 학습) |
| `261109_Week2_Practice\datamanager.py`, `feature.py`, `models.py`, `train.py` | 수업 GIN 코드 (원자 특징 147, 결합 특징 16, GIN 5층, hidden 128). 5k 표본 1회 학습 242–289 s (CPU) |
| `261109_Week2_Practice\mp_xai_ext.py` | Joback Tm 원자 분해 (`joback_tm(smiles)`, thermo 패키지 필요) |
| `261109_Week2_Practice\results\summary.csv` | 기존 결과 (5k 표본·전체) |
| `mp_week2\` | 2주차 정리: GIN(5k) mean 36.0 K / scaffold 41.4 K, sum readout 이득 없음, σ 3종 비교, XAI |
| `mp_mae_push\01_context_참고덱_분석.md` | 같은 데이터로 발표한 다른 팀의 방법론 분석 — 먼저 읽기 |

환경: Windows PC. 기존 conda 환경 `mlstudy`(Python 3.11, XGBoost, torch CPU, RDKit, shap; PyG는 확인 필요)가 있을 수 있음. 필요하면 Claude Science 작업 환경을 새로 만들어도 됨.
PC에서 torch(pip)와 conda 패키지를 섞어 쓸 때 OpenMP Error #15가 난 적이 있음 → `KMP_DUPLICATE_LIB_OK=TRUE`로 우회(수치에는 영향 없었음).
GPU가 필요한 실험은 원격 컴퓨트·Modal이 연결돼 있을 때만, 비용 추정을 먼저 보여주고 승인받은 뒤 진행.

---

## 2. 이미 아는 것 (출발점)

**우리 결과 (random 80:10:10, seed 42)**
- XGBoost + RDKit 217 (튜닝 없음): 27.50 K.
- 5k 표본 GIN mean 36.0 K. sum readout은 더 나쁘고 큰 분자를 비물리적으로(음수 K) 예측할 수 있음.

**참고 팀 (같은 Bradley, 19,944행 — 분할 인스턴스가 우리와 다름)**
- XGB(기술자 15 + Morgan) 29.3 / scaffold 36.6 K → 나무 수를 늘린 튜닝 26.4 / 35.3 → + 결정 특징 13개 25.95 / 34.54 (튜닝 모델 위에서는 유의하지 않음).
- PubChem 근거로 정답지 교정: 시험지 교정 −0.06~−0.20 K, 학습 라벨 교정 0~+0.17 K.
- 큰 분자 가중치: 효과 없음.
- Yalkowsky Tm = ΔHm/ΔSm (ΔHm을 target으로 학습): 550 K 초과 쏠림 −82 → −72 K, 대신 scaffold 전체 38.3 K로 악화.
- 모든 모델이 평균 쪽으로 당김 (저융점 +25–60 K, 550 K 초과 −50–85 K).
- 약점(기준값 흔들림, 순위로 효과 주장, 학습 곡선 없음 등)은 `01_context_참고덱_분석.md` 참고.

**문헌 (정확도)**
- Tetko et al. 2014, *J. Chem. Inf. Model.* 54:3320 — 47k 화합물, consensus RMSE < 33 °C (50–250 °C 구간). 이상치 제거는 consensus 결과에 거의 영향이 없었고, distance-to-model 지표로 이상치를 표시하지 못함.
- Tetko et al. 2016, *J. Cheminform.* 8:2 — 특허에서 약 30만 MP. consensus 정확도가 추정 실험 정확도(32 °C)와 비슷.
- Austermeier et al. 2025, *ACS Omega* — 같은 Bradley 데이터, GNN + SMILES 변형·tautomer 증강 consensus RMSE 35.4 °C (증강 없이 37.8 °C). 실험 불확도 약 35 °C를 실질적 하한으로 봄.
- → 우리 RMSE/MAE 비(36.88/27.50 = 1.34)로 환산하면 문헌 수준은 MAE ≈ 24–26 K.

**문헌 (자연어·LLM 표현, 6절)**
- Jablonka et al. 2024, *Nat. Mach. Intell.* 6:161 — 화학 질문을 자연어로 바꿔 GPT-3를 미세조정. 기존 ML과 비슷하거나 나았고, 특히 데이터가 적을 때 강점.
- Balaji et al. 2023 (GPT-MolBERTa, arXiv 2310.03030) — ChatGPT로 분자 설명문을 만들어 RoBERTa 학습. 회귀에서 최고 수준에 근접. 설명문에 밀도 같은 물성이 들어 있었음 → 녹는점 문제에서는 누설 위험.
- Zheng et al. 2025 (LLM4SD, *Nat. Mach. Intell.* 7:437) — LLM이 문헌 지식·데이터에서 규칙을 뽑고, 그 규칙으로 해석 가능한 특징을 만들어 기존 모델에 입력.
- Jacobs et al. 2024 (arXiv 2409.06080) — SMILES로 미세조정한 LLaMA 3가 분자 물성 회귀에서 쓸 만한 결과.

---

## 3. 1단계 — 진단과 방법론 설계 (실행 전; 결과는 `mp_mae_push\phase1\`)

1. **하드웨어·환경·비용 측정 준비**: CPU 코어 수, RAM, GPU 유무, 설치 가능한 패키지 확인. 7절의 비용 측정 도구(psutil, 선택: CodeCarbon)를 준비하고 단가 가정을 `cost_assumptions.md`에 기록. 이를 바탕으로 총 시간·비용 예산과 트랙별 예산·스레드 배분 제안.
2. **고정 분할**: `train_xgb.split()`으로 19,945행 전체의 random·scaffold 분할을 만들어 `mp_mae_push\splits\random.csv`, `scaffold.csv` (row index, smiles, InChIKey, set)로 저장. 이후 모든 트랙은 이 파일만 사용.
3. **기준선 재현**: `train_xgb.run('mp','random','desc',max_n=None)` → 27.50 ± 0.1 K 확인, 비용도 측정. scaffold도 실행해 기준값 확보.
4. **오차 분해 (빠르게)**
   - 라벨 잡음 하한: README 통계 + 가능하면 Bradley 원본(figshare, 반복측정 포함)에서 화합물 내 측정 산포 → 라벨 잡음이 MAE에 주는 몫 추정.
   - 고정 test 학습 곡선: train의 10/25/50/100 %로 XGB 기준선 학습(seed 3) → 데이터 2배당 MAE 감소량, random vs scaffold 기울기 → 외부 데이터의 기대 가치.
   - 오차 구조: MP 구간별 평균 잔차(95 % CI), 무거운 원자 수 × MP 구간 교차표, 계열별(염·전하·입체 표기·사슬형·다환·할로젠 다량) MAE, 상위 오차 50개 목록.
   - 근접 중복 점검: train–test 사이 InChIKey 첫 블록(입체 무시) 겹침, Tanimoto 1.0 쌍, cis/trans 미표기 병합 의심.
5. **문헌 훑기** (문헌 커넥터 — Literature Graph, PubMed 등): 2018–2026 유기 분자 MP 예측의 최고 성능·방법 10편 이내를 표로 (데이터, 분할, 지표, 값, 학습–시험 중복 제거 여부, 계산 비용).
6. **가설 목록**: 4–6절 후보 + 스스로 찾은 아이디어 5개 이상. 각 가설마다 예상 효과(K)와 근거, **예상 비용(CPU·GPU 시간, API 비용)**, 검증 실험, 실패 판정 기준을 표로. 예상 효과 ÷ 비용 순으로 정렬.
7. **Plan 제출**: 트랙·순서·예산·중단 기준. **승인 전 2단계 시작 금지.**

---

## 4. 2단계 — 병렬 실행 (Delegation 켜고 트랙별로)

### 공통 규약 (모든 트랙)
- 분할은 `splits\`만 사용.
- 하이퍼파라미터·특징·가중치·보정 배율 선택은 **validation 또는 train 안 5-fold CV**로만. test는 트랙별 최종 후보 1–2개에만, 마지막에 한 번.
- 확률적 모델은 seed ≥ 3 → 평균 ± SD. seed를 test로 고르지 않음.
- 트랙마다 자기 폴더 `mp_mae_push\T<번호>_<이름>\`에만 쓰기. 결과는 `ledger.csv`에 실행 하나당 한 줄:
  run_id, track, model, features, data_variant, objective, split, seed, params(JSON), val_MAE, val_RMSE, test_MAE(최종만), test_RMSE, test_R2, n_train, **wall_s, cpu_core_h, gpu_h, peak_ram_gb, energy_kwh, api_tokens, api_cost, infer_s_per_1k, stage(특징/탐색/최종/추론)**, notes.
  val·test 예측값은 CSV로 저장 (앙상블용).
- 동시 실행 시 CPU 과다 할당 금지: 트랙별 n_jobs·스레드(OMP_NUM_THREADS 등) 상한은 1단계에서 정한 대로.
- 오래 걸리는 작업은 스크립트로 실행하고 중간 저장(Optuna는 SQLite storage) → 세션이 끊겨도 이어서 진행.
- **금지**: 외부 MP 값을 특징으로 쓰기 / test 라벨을 모델 예측과 비교해 골라 고치기 / test로 조기 종료·선택하기.

### T1 — 트리 부스팅 × 특징 × 손실
- 모델: XGBoost, LightGBM, CatBoost (대조군: ExtraTrees).
- 특징 묶음 (누적): F0 RDKit 217 → F1 + Morgan **count** (반경 2, 2,048) → F2 + Mordred 2D (`mordredcommunity`) → F3 + T4 물리·결정 특징 → F4 + Joback Tm 추정값 → F5 + T8-M2 LLM 규칙 특징.
- 손실: MSE / MAE(L1) / Huber / 분위수 0.5 — 평가 지표가 MAE이고 잔차 꼬리가 두꺼움.
- Optuna TPE, 설정당 ≥ 100 trial, validation 조기 종료, 나무 수 상한은 넉넉히(≥ 10,000, lr 0.01–0.1). 참고 팀은 500그루에서 덜 학습된 상태였음. trial 수는 7절 비용 곡선을 보고 줄여도 됨.

### T2 — 그래프 신경망
- 수업 GIN/GINE(mean readout), PyG AttentiveFP, chemprop v2 D-MPNN(+ RDKit 기술자를 분자 특징으로 추가).
- readout(mean / sum / attention) 비교 시 물리적으로 불가능한 예측(< 0 K, < 50 K) 개수도 보고.
- 탐색: 층 3–6, hidden 128–512, dropout, lr, batch. 입력 데이터 변형은 T7과 함께.
- CPU뿐이면 설정 수를 줄여 예산을 지킴.

### T3 — 사전학습 표현 (가능할 때만)
- 예: CheMeleon(chemprop 기반 사전학습 모델), MolFormer·ChemBERTa 임베딩 + GBDT/MLP head.
- 다운로드 크기·라이선스·CPU 시간을 확인한 뒤 진행. 불가하면 사유를 기록.

### T4 — 물리·결정 특징
- 특징 후보: 회전 대칭수 σ(여러 ETKDG conformer에서 대표값), Yalkowsky 계열 ΔSm(σ, 유연성 τ; 정확한 식은 문헌으로 확인), 수소결합 주개–받개 짝, 카복실산·아마이드·우레아·하이드록시, 평면성·NPR, 강직성(고리 원자 비율), π 계 크기, 할로젠.
- 넣는 방식 두 가지:
  - (a) 특징 열 (T1의 F3)
  - (b) 구조 — Tm_pred = ΔHm_pred / ΔSm, **손실은 Tm 척도**, ΔSm 계수 학습 허용. 참고 팀은 ΔHm 척도로 학습해서 강직한(ΔSm 작은) 분자의 오차가 Tm에서 커졌고 scaffold가 나빠졌음.
- 효과 판정: 한 열씩 빼보는 실험(ΔMAE + CI)과 550 K 초과·고대칭 부분집합의 짝지은 CI. SHAP 순위로 효과를 주장하지 않음.

### T5 — 정제와 외부 데이터
- 규칙 기반 정제 (규칙은 실행 전에 문서로 고정하고, 모델 예측은 보지 않음): 염·혼합물·전하, cis/trans 미표기 병합 의심, 반복측정 충돌, 분해(decomposition) 표지.
- **PubChem** (Featured 커넥터 또는 PUG-REST, 초당 5회 이하):
  - 이름↔구조 확인, train·val 라벨 감사 (예: PubChem 실험값 ±15 K 기준).
  - test 라벨은 원본 유지. 정제한 test는 따로 보고하고, 2×2(학습 라벨 원본/정제 × 시험 라벨 원본/정제)로 효과를 분리. 고친 행만의 MAE도 따로.
  - PubChem 실험 MP는 Bradley와 출처가 겹칠 수 있음 (순환 검증 주의).
- 외부 데이터: OCHEM(Tetko 2014·2016 세트) 등 공개 MP 데이터 → test·val 분자와 InChIKey(전체·첫 블록)로 겹치는 것을 지운 뒤 **학습에만** 추가. 1단계 학습 곡선이 예측한 효과와 비교.

### T6 — 앙상블·보정·선택적 예측
- 트랙들의 validation 예측으로 가중 평균/스태킹 (비음수 가중치, 단순 메타모델). 가중치는 validation에서만 정함. 앙상블의 비용은 구성 모델 비용의 합으로 계산.
- 평균 당김 보정: validation에서 isotonic/선형 재보정 → MAE 득실 보고.
- 불확실성: 앙상블 분산, 분위수 회귀, split conformal → 위험–커버리지 곡선과 MAE ≤ 10 K 최대 커버리지 (목표 C). Tetko 2014에서는 distance-to-model 지표가 잘 안 통했으니 결과가 약해도 그대로 보고.

---

## 5. [추가 요청 ①] T7 — 데이터셋 바꾸기 (GIN과 다른 모델 공통)

원칙: 바꾸는 건 train(+val)뿐이고, test의 분자와 라벨은 원본 그대로 둡니다. 같은 모델·같은 설정에서 **데이터만** 바꿔 ΔMAE(짝지은 CI)와 비용을 잽니다. 결과는 ledger의 `data_variant` 열로 구분합니다.

| 묶음 | 바꾸는 것 | 주로 듣는 모델 | 비용 |
|---|---|---|---|
| D1 라벨 품질 | 반복측정 산포로 표본 가중치(산포 큰 분자 가중↓); 규칙 기반 제외(염·혼합물·분해·cis/trans 병합 의심 — T5 규칙 공유); 극단 라벨 winsorize 또는 Huber 손실 | 전부 | 낮음 |
| D2 그래프 입력 보강 | 수업 그래프에는 일반 E/Z와 원자 키랄성이 없음 (결합 16 = 결합 차수 4·공액 1·회전/비고리/고리 밖 단일 3·양 끝 원소 8; 고리 안 cis/trans만 작용기 비트로 있음) → 결합 E/Z·원자 키랄성·Gasteiger 전하·고리 크기·수소결합 주개/받개 표시 추가, 가상 노드, 분자 기술자(RDKit 217 중 선택분, T4 결정 특징)를 readout 뒤에 이어 붙이는 하이브리드 | GNN | 낮음–중간 |
| D3 목표값 바꾸기 | Δ-학습: y − Joback Tm, 또는 y − XGB의 out-of-fold 예측 → GNN이 잔차만 학습; Yalkowsky 구조 target(T4-b) | GNN, GBDT | 낮음 |
| D4 데이터 늘리기 | 외부 MP(T5); tautomer 증강(그래프가 바뀜). SMILES 무작위화는 GIN 입력 그래프를 바꾸지 않음(원자 순서 불변) → 시퀀스·텍스트 모델(T3·T8)에만. Austermeier 2025의 증강 효과(37.8 → 35.4 °C)가 어느 쪽에서 왔는지 우리 데이터로 분리 | GNN, 시퀀스 | 중간 |
| D5 관련 물성 활용 | 다중 작업(mp + 같은 data 폴더의 bp·logPvap·density·logSolubility, 공유 인코더; 평가는 mp만), 또는 연쇄(구조로 예측한 bp를 mp 모델 특징으로 — 측정 bp는 쓰지 않음) | GNN(다중 작업), GBDT(연쇄) | 중간 |
| D6 사전학습 | 라벨 없는 대량 분자(예: ZINC 커넥터)에 RDKit/Mordred 기술자를 보조 target으로 GIN 사전학습 → mp 미세조정 | GNN | 높음 (파일럿 먼저) |
| D7 분포 조정 | 극단 MP 구간 oversampling·가중 (참고 팀: 큰 분자 가중은 효과 없음) — validation으로만 판단 | 전부 | 낮음 |

---

## 6. [추가 요청 ②] T8 — 자연어(LLM) 데이터셋

목표: 분자를 SMILES 대신 LLM이 바로 읽을 수 있는 자연어로 바꿨을 때의 정확도와 비용을 확인합니다.

**표현 만들기** — 구조에서 계산한 정보만 씁니다. 녹는점·끓는점·밀도 같은 물성값, 화합물의 알려진 용도나 상온 상태 서술은 넣지 않습니다.
- N1 이름: IUPAC 이름 (PubChem `IUPACName`; 없으면 빈칸으로 두고 표시).
- N2 템플릿 서술 (코드로 생성, LLM 불필요): 예) "분자식 C8H8O3, 무거운 원자 11개, 벤젠 고리 1개(1,4-치환), 카복실산 1개·하이드록시 1개, 수소결합 주개 2·받개 3, 회전 가능 결합 2개, 거의 평면, 회전 대칭수 1 …" — 가장 싸고 누설 위험이 없음.
- N3 LLM 생성 서술 (GPT-MolBERTa 방식): 구조 설명만 쓰도록 지시하고, 출력에 숫자+물성 단위(°C, K, g/cm³ 등) 패턴이 있으면 걸러냄. 분자 수천 개 이상 호출은 토큰 수·비용 추정치를 먼저 보여주고 승인받기.

**모델링** — 싼 것부터 하고, 각 단계에서 비용 대비 효과가 없으면 멈춥니다.
- M1 텍스트 임베딩(로컬 소형 임베딩 모델 또는 화학 언어 모델) + GBDT/Ridge head.
- M2 LLM 지식 → 특징 규칙 (LLM4SD 방식): LLM에게 녹는점을 좌우하는 구조 규칙(대칭, 수소결합 네트워크, 강직성, 패킹 등)을 제안받아 RDKit 코드로 구현 → T1 특징 묶음 F5로 추가. LLM 호출이 한 번뿐이라 싸고 추론 비용은 0. 규칙을 제안받을 때 test 데이터는 보여주지 않음.
- M3 소형 언어 모델 미세조정 (RoBERTa·ChemBERTa급 + 회귀 head) — GPU가 필요할 수 있음.
- M4 대형 LLM 미세조정·few-shot 회귀 (Jablonka 2024 방식) — 데이터가 적을 때 강점이 보고됐고, 우리 규모(학습 약 1.6만)에서는 이득이 불확실. 학습 1,000/4,000개 파일럿으로 학습 곡선과 비용을 먼저 보고 결정.

**기억 누설 점검 (필수)**: LLM은 유명 화합물의 녹는점을 학습 중에 봤을 수 있습니다.
- (a) scaffold 분할에서도 평가.
- (b) PubChem에 실험 MP 기록이 있는/없는 분자, 이름이 흔한/드문 분자로 나눠 오차 비교 — 텍스트 모델만 '유명한' 분자에서 유독 좋으면 기억 효과로 판정.
- (c) 생성 텍스트에 물성 숫자 금지.

**판정**: 같은 head에서 RDKit 217 특징 대비 ΔMAE(짝지은 CI)와 7절 비용.

---

## 7. [추가 요청 ③] 비용 대비 효과 — 가장 경제적이고 효과적인 방법 찾기

정확도만 보지 않고, 모든 run의 자원 사용량을 재서 비용으로 환산합니다.

**측정** (ledger 열)
- wall_s
- cpu_core_h: psutil로 프로세스와 자식 프로세스의 user+sys CPU 시간 합 ÷ 3600
- gpu_h, gpu_avg_W: GPU를 쓸 때 nvidia-smi 샘플링
- peak_ram_gb
- energy_kwh: CodeCarbon 또는 CPU 사용률 × TDP 추정 (어느 방법인지 같이 기록)
- api_tokens(in/out), api_cost: LLM API를 쓸 때
- infer_s_per_1k: 분자 1,000개 예측 시간
- 단계별로 나눠 기록: ① 특징·임베딩 생성(한 번 만들어 여러 모델이 쓰면 나눠 계산) ② 탐색(튜닝 trial 전체 합) ③ 최종 학습 ④ 추론.

**환산** — 가정은 날짜·출처와 함께 `mp_mae_push\cost_assumptions.md`에 기록
- 두 기준으로 모두 계산: (i) 로컬 PC 전기요금 기준 (kWh × 원/kWh; 한국전력 공시 요금 확인), (ii) 클라우드 환산 (공개 on-demand 단가: vCPU-시간, GPU-시간; 조회한 페이지와 날짜 기록).
- Claude 자체 사용량(에이전트 토큰)은 정확히 나누기 어려우면 트랙별 대략치로 따로 표기.

**결정 규칙**
- 파레토 그림: x = 총비용(탐색 + 최종 학습, 로그 축), y = validation MAE (최종 후보는 test MAE도 표시). 추론 비용 그림도 따로.
- **"가장 경제적인 방법" = 가장 정확한 후보와 통계적으로 구분되지 않는(짝지은 CI가 0을 포함) 후보 중 총비용이 가장 낮은 것.**
- 보조 지표: 기준선(27.50 K, 학습 6.2 s) 대비 1 K 개선당 비용, 100만 분자 스크리닝 추론 비용.
- 비싼 방법(대형 LLM, 3D, 대규모 사전학습)은 작은 파일럿 → 학습 곡선과 비용을 외삽 → 파레토 안쪽에 들 가능성이 있을 때만 본 실행.

---

## 8. 3단계 — 최종 평가와 보고 (`mp_mae_push\REPORT.md`, 한국어)

- 최종 후보 ≤ 3개만 test 1회: MAE, RMSE, R², 기준선 27.50 K와의 짝지은 부트스트랩 95 % CI, seed 평균 ± SD, MP 구간별 잔차, scaffold 결과, 비용(단계별).
- 순서: 결론 3줄 → 목표 A/B/C/D 달성 여부 → 추천 세 가지(**가장 정확한 방법 / 가장 경제적인 방법 / 균형안**)와 이유 → 무엇이 효과가 있었고 없었나 (유의/비유의 구분, 비용 열 포함 표) → 파레토 그림·위험–커버리지 그림 → 다음 단계.
- 모든 수치 옆에 분할·단위(K)·n·seed 표기.
- 제출 전 Claude Science reviewer로 주장과 실행 기록이 맞는지 점검.

---

## 9. 누설 감사 (최종 전 필수; test MAE < 20 K가 나오면 즉시)

1. train·val·test(외부 데이터 포함) 사이 InChIKey 전체·첫 블록 겹침 0.
2. 특징에 라벨 유래 정보 없음 (외부 MP 값, target encoding, 라벨로 만든 클러스터, LLM이 만든 텍스트 속 물성값 등).
3. 스케일러·특징 선택·하이퍼파라미터·앙상블 가중치가 test 없이 정해짐.
4. T8 기억 누설 점검 결과 첨부.
5. 문헌 대비 비정상적으로 좋으면 중복 제거 전후 비교.

---

## 10. 진행 보고

- 1단계 끝: 진단 요약 + 가설 표(효과·비용) + Plan.
- 각 트랙 끝: ledger 상위 5개(validation 기준, 비용 포함)와 실패한 시도 한 줄씩.
- 1 GB 넘는 다운로드, 외부 데이터 사용, 대량 LLM 호출, GPU·클라우드 사용, 예산 초과 전에는 먼저 물어보기.
