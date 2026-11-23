# mp_week2 — 2주차 (GIN · 불확실성 · XAI) mp 재실행 결과와 공부 자료

2주차 자료(`261109_2주차_GIN학습불확실성XAI_발표.pdf` 58쪽 + `261109_Week2_Practice/`)를 팀 물성 **mp(녹는점)** 로 다시 돌리고, XAI 를 mp 에 맞게 확장한 결과. 실행 2026-11-14 ~ 11-17 (Claude 컨테이너).
표기: **[PPT]** 덱·실습 꾸러미 내용, **[추가]** 이번에 덧붙인 것. 덱의 숫자는 logP, 여기 숫자는 전부 mp (K).

## 읽는 순서

| 순서 | 파일 | 내용 |
|---|---|---|
| 1 | [`00_2주차_내용정리.md`](00_2주차_내용정리.md) | 덱 전체 정리 — **1절 = 어떤 튜닝(학습 설정)을 썼나 총정리**, 블록별 개념·수식·shape, 암기/이해, 자주 하는 실수, 자가진단 |
| 2 | [`01_gin/gin_results_table.md`](01_gin/gin_results_table.md) | 과제 ① GIN 결과표 (mean/sum × random/scaffold), seed 3개, 층 수·GINE·GCN, 1주차 가설 확인 |
| 3 | [`02_uncertainty/uncertainty_mp.md`](02_uncertainty/uncertainty_mp.md) | 과제 ② σ 세 방법 표 (random/scaffold), 재보정, 적용범위, 능동학습 |
| 4 | [`03_xai/xai_mp_notes.md`](03_xai/xai_mp_notes.md) | 과제 ③ **mp XAI**: SHAP 상위 10 + 화학 해석 3줄, GIN 원자 기여를 Joback 으로 검증, sanity check 셋, 설명이 틀린 분자 2개 |

## 결과 한눈에 (test 500, seed 42, K)

| | random | scaffold |
|---|---|---|
| 기준선 (train 평균) | 75.2 | 79.7 |
| XGBoost + 기술자 (04 노트북 설정, 189개) | **33.1** | **36.7** |
| GIN mean (`train.py`) — seed 3개 평균 ± SD | 36.0 (36.5 ± 0.5) | 41.4 (42.3 ± 0.8) |
| GIN sum | 36.4 (36.9 ± 1.0) | 43.9 (44.4 ± 0.9) |
| BGNN-GIN (σ 포함) | 36.1, 2σ 적중 0.77 | 41.5, 2σ 적중 **0.42** |
| GP (Tanimoto) | 38.1, 2σ 적중 0.95 | 46.3, 2σ 적중 0.92 |

- **sum readout 은 mp 에서 이득 없음** (validate 기준 1.3–1.4 K 나쁨, seed 3개 모두). logP(덱 13쪽)와 반대.
- 층 수·GINE·GCN: validate 로는 기본 설정과 차이가 seed 흔들림 안 → 고를 것 없음 (test 최저 GCN 35.0 은 규칙상 고르지 않음).
- **정확한 모델 ≠ 쓸 만한 σ**: BGNN MAE 최저, σ 순위 Spearman 0.09. GP 는 적중률 거의 이상적이지만 순위는 0.04–0.21.
- **XAI**: SHAP 상위 10 이 분할·seed 에 안정(9/10, ρ 0.98), 화학적으로 일관(극성·수소결합·강직성 ↑, 유연성 ↓). GIN IG 원자 기여가 Joback 작용기 기여와 ρ 0.61 (학습 안 한 모델 0.25, 무작위 −0.03), seed 일치 0.82. 그러나 설명 일치도는 예측 정확도와 무관(ρ 0.14), 방법끼리 부호가 반대인 분자도 있음.
- 1주차 가설 "sum readout 이 작은 비고리 분자 오차를 고친다"는 **validate 에서 기각**.

## 무엇을 바꿨나 (교수님 파일은 고치지 않음)

`261109_Week2_Practice/` 에 **추가한** 파일:

| 파일 | 원본 대비 |
|---|---|
| `02_gin_walkthrough_mp.ipynb` (+ `_executed`) | 5절 `run(prop='mp')` 로만 |
| `03_uncertainty_active_learning_mp.ipynb` (+ `_executed`) | `PROP='mp'`, `MAX_N=5000`, 16번 셀 표 행 번호 (pool 크기 변화), 끝에 [추가] 3칸 |
| `04_xai_methods_mp.ipynb` (+ `_executed`) | `PROP='mp'`, `MAX_N=5000`, `torch.set_num_threads(1)`, 6번 위치에 mp 주의 문단, 끝에 [추가] X0–X7 22칸 |
| `mp_xai_ext.py` | [추가] Joback Tm 원자 분해 (04 노트북이 import) |
| `run_seed_init.py` | [추가] 분할은 seed 42 고정, 가중치 seed 만 바꿔 `train.run` |
| `parse_val.py` | [추가] `train.py` 로그에서 최저 validate MAE 뽑기 (`summary.csv` 에 validate 가 없어서) |
| `fig_wrong_expl.py` | [추가] 과제 그림 (설명이 틀린 분자 2개) |

`train.py`·`models.py`·`datamanager.py`·`feature.py`·`xai.py`·`uncertainty.py` 는 그대로 (1주차 파일과 바이트 동일한 것 확인: `train.py`, `models.py`, `datamanager.py`, `feature.py`, `data/mp.csv`).

## 다시 돌리기 (PC `mlstudy`)

```bat
conda activate mlstudy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric shap thermo          :: thermo = Joback (04 [추가] 절에만 필요)
set KMP_DUPLICATE_LIB_OK=TRUE                     :: 1주차 OpenMP Error #15 우회 (env/00_설치안내.md ⑥)
cd 261109_Week2_Practice
python train.py --prop mp
python train.py --prop mp --split scaffold
python train.py --prop mp --pooling sum
python train.py --prop mp --pooling sum --split scaffold
python train.py --prop mp --conv GINE
python train.py --prop mp --conv GCN
python -c "from train import run; run('mp', num_layers=3, outdir='results_layers/L3')"
python run_seed_init.py --seed 7 --pooling mean --split random      :: seed 7·123 × mean·sum × random·scaffold
python parse_val.py "logs/*.log"                                    :: 로그를 파일로 남긴 경우
jupyter nbconvert --to notebook --execute 04_xai_methods_mp.ipynb   :: 약 20분 (CPU 2개 기준)
```

실행 시간 (CPU 2개): `train.py` 1번 1–4분, 노트북 03 약 16분 (능동학습 9분), 04 약 20분.

## 실행 환경과 재현성

Python 3.13.18 · torch 2.15.0 · torch_geometric 2.8.1 · RDKit 2026.09.2 · shap 0.54.1 · xgboost 3.4.1 · scikit-learn 1.9.1 · numpy 2.5.3 · pandas 3.0.5 · thermo 0.6.1 (Claude 컨테이너, CPU 2개, `OMP_NUM_THREADS=1`).
1주차 PC(`mlstudy`)와 버전이 달라 GIN 은 같은 seed 라도 에폭 수·MAE 가 조금 다르다 (1주차 run_notes 와 같은 현상) — 그래서 **seed 3개로 흔들림 범위를 같이 적었다**. 결정적 모델(Ridge, 나무, Joback)은 같은 결과가 나와야 한다.
실행 도중 환경이 한 번 끊겨(11-14) 04 노트북과 여유 실험 일부를 11-17 에 처음부터 다시 돌렸다. 02·03 노트북과 `train.py` 핵심 4번은 11-14 실행분.

## 남은 것

- 3주차 발표 10분: 축 하나(초안: E 해석, `03_xai/xai_mp_notes.md` 9절) + **선행 3편 표**(비어 있음) + 가설 한 줄
- BGNN 재보정: `train.run` 이 validate 예측을 저장하지 않아 못 함 (`02_uncertainty` 2절)
- 적용범위에서 유사도 0.8 이상인데 오차가 큰 분자 30개 그려 보기 (`02_uncertainty` 4절)
- GCN 이 정말 나은지 seed 3개로 validate 재확인 (`01_gin` 3절)
