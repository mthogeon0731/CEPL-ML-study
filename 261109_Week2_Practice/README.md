# 2주차 실습 (과제). GIN · 불확실성 · XAI

분자 물성 강의는 2주차로 끝나고, 3–4주차는 팀 발표, 5–6주차는 공정 데이터 강의와 마무리다. 수업에서는 설명만 했고, 실습은 이 꾸러미의 노트북 세 개를 팀별로 돌려 오는 과제다.

## 1. 설치 (1주차 환경 `mlstudy` 에 더하기)

```bash
conda activate mlstudy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric shap
```

GPU 없이 CPU 로 충분하다. 확인: `python feature.py` 가 아래처럼 나오면 된다.

```
piperidine  x: (6, 147)  edge_index: (2, 12)  edge_attr: (12, 16)
```

## 2. 파일

| 파일 | 덱 블록 | 하는 일 |
|---|---|---|
| `02_gin_walkthrough.ipynb` | 1 GIN, 2 학습 개념 | 그래프 텐서 읽기 → GIN 한 층 손 계산 → sum vs mean → 배치 · readout → 학습 |
| `03_uncertainty_active_learning.ipynb` | 3 불확실성 → 결정 | σ 세 방법 (RF · BGNN-GIN · GP) → 2σ 적중률 → 적용범위 → 능동학습. CPU 약 4분 (BGNN 이 가장 김) |
| `04_xai_methods.ipynb` | 4 XAI | a 기술자 모델 전역 설명 → b Morgan 비트 → c GIN 원자 기여 → d 설명 검증 → e 해석 모델 → f 함정. CPU 약 8분 |
| `feature.py` | | 분자 → 그래프 (원자 특징 147, 결합 특징 16) |
| `datamanager.py` | | CSV → 그래프 → train / validate / test 80:10:10 → y 표준화 (train 으로만) |
| `models.py` | | GIN, GINE, GCN, GAT / mean 또는 sum readout / Bayesian 출력층 |
| `train.py` | | 학습 한 번 = 결과표 한 줄 (`results/summary.csv`). AdamW 1e-3, 조기 종료 30 에폭 |
| `uncertainty.py` | | RF σ, GP σ (Tanimoto), 보정 지표, 최대 유사도, 능동학습 루프 |
| `xai.py` | | permutation · SHAP, Morgan 비트 그림, saliency · grad×input · IG · GNNExplainer · readout 분해, Crippen 정답 |
| `data/` | | 물성 9종. 출처 · 단위는 `data/README_data.md` |

노트북마다 첫 셀의 `PROP` 를 팀 물성으로 바꿔 끝까지 실행한다.

```bash
python train.py --prop logP                    # 5,000개 기준 CPU 약 3분
python train.py --prop logP --split scaffold
python train.py --prop logP --pooling sum
python train.py --prop logP --conv GINE
python train.py --prop logP --bayes
```

`--prop` 은 팀 물성(`data/` 의 파일 이름), `--max_n 0` 이면 전체 데이터.

## 3. 과제 (3주차 팀 발표 10분)

| 노트북 | 낼 것 |
|---|---|
| 02 + `train.py` | 팀 물성 GIN 을 random · scaffold 로 돌려 **1주차 결과표에 GIN 줄 추가** (mean, sum readout). 학습 곡선 그림 |
| 03 | σ 세 방법 표 (MAE, 평균 σ, 2σ 적중률, Spearman) 를 random · scaffold 로 |
| 04 | SHAP 상위 10 표 + 화학 해석 3줄. 설명이 틀린 분자 2개 그림과 이유 (d 절 정답 비교는 logP 에서만) |

발표에 같이 가져올 것

1. 고른 축 하나 (A 표현 · B 물리 결합 · C 데이터 · D 불확실성 · E 해석 · F 결정 연결, 덱 43쪽)
2. 가장 가까운 선행 3편 표: 선행 | 데이터 | 모델 | 분할 | 오차 | 우리와 다른 점
3. 가설 한 줄: "X 를 바꾸면 Y 가 좋아질 것이다. 선행은 Z 를 안 했다"

여유가 있으면 (축별로 하나)

- A · C: 층 수 1, 3, 5 / `--conv GINE` vs `GCN` / `--pooling sum` 중 둘을 바꿔 MAE 표
- D: 2σ 적중률이 0.954 에서 가장 먼 방법을 validate 로 재보정하고 test 에서 다시 재기
- E: 설명 방법 하나를 seed 3개로 돌려 순위 일치도. 낮으면 그 설명으로 결론 내지 않기
- B: 04 노트북 e 절 원자 가산 모델을 팀 물성에. GIN 과의 차이를 물리 항(예: logS = f(logP, mp))으로 메울 수 있나
- F: '가장 큰 값' 대신 목표 구간(예: 점도 1–10 cP)에 드는 분자를 찾는 획득 함수

## 4. 규칙

- 하이퍼파라미터 · 재보정 배율 · 획득 함수 설정은 **validate 로 고른다**. test 는 마지막에 한 번만 본다
- 이긴 경우와 진 경우를 모두 보여 준다 (scaffold 분할에서 지는 것도 결과다)
- 설명 그림을 결과로 쓸 때는 무작위 기준선, 학습 안 한 모델, seed 2개 이상을 같이 보고한다
