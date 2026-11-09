# 05 모델의 분석 — 후분석(어디서 틀리나)과 XAI(무엇을 보고 예측하나)

슬라이드 근거: 31쪽 정확도 vs 설명 가능성, 23–24쪽 parity 읽기, 49쪽 선택 과제 "틀린 분자 보기" · 4주차 XAI 예고 · 발표 3:20–4:20 · 슬라이드 S6

> 이 절은 1주차 범위를 조금 넘는다 (SHAP·원자 기여는 4주차). 발표에서는 **후분석(오차 해부)을 주로**, XAI는 "이렇게 볼 수 있다" 한 장으로.
> 31쪽의 질문 "여러분 팀 물성에서는 정확도와 설명 중 무엇이 먼저입니까?"에 mp 관점으로 답을 준비한다.

## ① 발표에서 답할 질문

### 후분석 (post-hoc error analysis)
1. **어떤 분자에서 가장 틀리나** — |오차| 상위 20개 표: SMILES, 실제 mp, 예측, 오차. 공통점(큰 분자? 염? 이온성? 수소결합 많은 분자? 대칭 분자?)
2. **골격별 오차** — Murcko 골격별 MAE: 새 골격(scaffold test)에서 얼마나 나빠지나 (14쪽). 비고리 분자(빈 골격)의 오차.
3. **영역별 오차** — mp 구간별(예: < 250 K, 250–400, 400–550, > 550 K) MAE와 데이터 수 → 끝 영역 오차 = 데이터 부족 (24쪽).
4. **분자 종류별** — 23쪽처럼 parity 점 색을 종류별로(아민 1·2·3차/비아민, 또는 작용기) → 종류별 오차 차이.
5. **출처별** — source 열이 있으면 출처별 MAE (13쪽 "출처별 오차를 따로 계산"). 잡음의 출처를 분리.

### XAI (explainable AI)
6. **기술자 모델에서 무엇이 중요한가** — XGBoost + RDKit 기술자 → SHAP (43쪽에 설치한 `shap`, 4주차 도구). 상위 기술자가 화학 상식과 맞는가 (예: 수소결합 공여체 수·방향족 고리·분자량이 mp를 올리는 방향인가) → **31쪽 "설명 검증"**.
7. **GNN에서는** — 원자 기여(4주차), BGIN의 σ (22쪽: σ 큼 = 학습 데이터와 먼 분자). 이번 발표에서는 "다음 단계"로만.
8. **정확도 vs 설명** — 스크리닝(순위만 맞으면 됨) vs 실험 설계(왜 그 분자인지) 중 mp는 어느 쪽인가 (31쪽).

## ② 슬라이드 근거

| 쪽 | 내용 |
|---|---|
| 13 | 출처 섞기 → source 열 유지, 출처별 오차 따로 |
| 14 | scaffold split = 새 골격 예측 능력 |
| 20 | 학습 범위 밖(큰 분자, 새 작용기) → 어떤 모델이든 오차 큼 |
| 22 | BGIN σ: 지문 공간(PCA 2D)에서 데이터가 드문 곳일수록 σ 큼; σ가 실제 오차와 맞는지 따로 확인 |
| 23 | parity 색 = 분자 종류 → 종류별 오차 차이 |
| 24 | 지는 케이스 그대로 보여줌 |
| 31 | 정확도 vs 설명; 설명 = 다음 실험을 고르는 근거, 틀린 예측을 찾는 단서; 4주차 XAI: SHAP, 원자 기여, 설명 검증; 연구실 KAN-DoE 기술자 중요도 |
| 49 | 선택 과제: 노트북 마지막 절 "틀린 분자 보기" |

## ③ 만들 산출물

| 파일 | 내용 | 만드는 법 |
|---|---|---|
| `worst_molecules.csv` | \|오차\| 상위 k개 (SMILES, y_true, y_pred, error, MolWt, scaffold) | `python ../scripts/error_analysis.py <예측 CSV>` |
| `error_by_scaffold.csv`, `error_by_range.csv` | 골격별·구간별 MAE와 개수 | 같은 스크립트 |
| `fig_residuals.png` | 잔차 vs 실제값 + 잔차 히스토그램 | 같은 스크립트 |
| `fig_error_by_molwt.png` | 분자량 구간별 MAE | 같은 스크립트 |
| `fig_shap_summary.png`, `shap_top_features.csv` | XGBoost+기술자 SHAP 요약 (상위 20 기술자) | `python ../scripts/shap_descriptors.py <mp CSV>` — 또는 노트북 9절 선택 과제 **RF `feature_importances_` 상위 10** (추가 패키지 없이) |
| `xai_notes.md` | 상위 기술자 vs 화학 상식 대조표 + 31쪽 질문의 답 | 손으로 |

**예측 CSV는 이미 나온다**

- GIN: `python train.py --prop mp` 가 `results/mp_GNN-GIN_mean_random_test_pred.csv` (열 `smiles, y, pred, sd`)를 저장한다 → `error_analysis.py`가 열 이름을 자동 인식한다. 기준선용 train 평균은 `results/summary.csv`의 `baseline_MAE`로 갈음하거나 `--train-mean`으로 준다.
- 고전 모델: 노트북 6절의 `preds[(split, feature, model)] = (y_true, y_pred)`와 `splits[split][1]`(test 인덱스)로 한 셀:

```python
import pandas as pd
split, feat, model = 'scaffold', 'Descriptor', 'XGBoost'          # 결과표에서 고른 조합
yt, yp = preds[(split, feat, model)]; tr, te = splits[split]
pd.DataFrame({'smiles': df['smiles'].values[te], 'y_true': yt, 'y_pred': yp, 'train_mean': y[tr].mean()}
             ).to_csv(f'../mp_presentation/results/pred_test_{model}_{feat}_{split}.csv', index=False)
```

### 예시 출력 — `example_xgb_descriptors_scaffold/` (전체 19,945개, scaffold 80/10/10, 보조 모델; 2026-11-06 생성)

| 결과 | 값 | 발표에서 |
|---|---|---|
| XGBoost + 기술자 203개, scaffold test 1,995개 | **MAE 36.8 K**, RMSE 49.2 K, R² 0.60, 기준선(train 평균 366.9 K) MAE 78.2 K → 모델/기준선 0.47 | 결과표의 XGBoost·기술자·scaffold 칸과 비교 (표본 5,000 vs 전체, 80/20 vs 80/10/10 차이 명시) |
| 구간별 MAE | 350–500 K: 28–33 K / 500–550: 45 / 550–600: 66 / 600–650: 106 / 700 K 이상: 200 K 이상 (n ≤ 5) — **끝 영역일수록, 데이터가 적을수록 오차 큼**; 높은 mp는 과소, 낮은 mp는 과대 예측(평균 회귀) | 4절 parity 끝 영역 해석의 숫자 근거 |
| SHAP 상위 | RingCount ↑, TPSA ↑, BertzCT(복잡도) ↑, NumHDonors ↑ → mp ↑; **NumRotatableBonds ↑ → mp ↓**; 그다음 SMR_VSA10, FpDensityMorgan3 ↓, SlogP_VSA8, NHOHCount ↑ | 화학 상식과 대조: 고리·극성·수소결합 공여체는 격자를 단단히(mp ↑), 유연한 사슬은 느슨하게(mp ↓) → "설명 검증" 통과. 상관 큰 쌍(TPSA–NHOHCount–NumHDonors)은 기여가 나뉜 것 |
| 골격별 | scaffold test는 골격 1,995개가 모두 1개짜리 → 골격별 표 대신 구간별·분자량별 표 사용 | 14쪽 "새 골격"의 뜻을 숫자로 |
| 틀린 분자 상위 20 | `worst_molecules.csv/.png` — 20개 중 17개가 **과소 예측**(실제 mp가 더 높음), 9개는 mp ≥ 600 K; 1·2위(오차 −330, −310 K)는 비방향족 7원 고리 표기 등 **SMILES·값 오류 의심** → 12쪽 "값 차이 큼 → 원 출처 확인"의 실제 예 | S6 그림: 상위 3개 구조 + "오류 의심 vs 모델 한계" 구분 |

> 이 예시는 결과표(과제 1단계)와 **다른 모델·다른 분할**이다. 발표에서는 "보조 분석"으로 명시하고, 본 결과표의 XGBoost 예측으로 같은 스크립트를 다시 돌린 결과를 쓰는 것이 깔끔하다 (`--max_rows`로 5,000개 표본도 가능).

## ④ 작업 순서

```
cd "C:\Users\mthog\CEPL ML\mp_presentation"
python scripts\error_analysis.py "..\261105_Week1_Practice\results\mp_GNN-GIN_mean_random_test_pred.csv" --out 05_analysis_xai\gin_random --top 20
python scripts\shap_descriptors.py "..\261105_Week1_Practice\data\mp.csv" --out 05_analysis_xai\xgb_scaffold --split scaffold
python scripts\error_analysis.py 05_analysis_xai\xgb_scaffold\pred_test_xgb_scaffold.csv --out 05_analysis_xai\xgb_scaffold --top 20
```

1. 후분석 먼저. 상위 20개 분자를 **RDKit으로 그려서 눈으로 본다** (스크립트가 `worst_molecules.png` 격자 그림을 만든다). 공통점을 한 문장으로.
2. 골격별·구간별 표에서 "오차가 큰 곳 = 데이터가 적은 곳"인지 확인 → 4절의 parity 끝 영역 해석과 같은 결론인지.
3. SHAP 요약 그림에서 상위 5개 기술자를 `xai_notes.md` 표에 옮기고, **방향**(값이 크면 mp가 올라가나 내려가나)을 화학적으로 설명해 본다. 설명이 안 되는 기술자가 있으면 그것이 "설명 검증"의 결과다 (31쪽).
4. [추가] 주의: SHAP는 **모델이 무엇을 보는가**이지 **자연이 그런가**가 아니다. 기술자끼리 상관이 크므로(16쪽) 기여가 서로 나뉘어 들어갈 수 있다. "MolWt가 중요하다"는 "크기와 상관된 무언가가 중요하다"로 읽는다.

## ⑤ 자가점검

- [ ] 가장 많이 틀린 분자 3개를 구조와 함께 말하고, 왜 틀렸을지 가설이 있다
- [ ] 오차가 큰 영역이 데이터가 적은 영역인지 숫자로 확인했다
- [ ] SHAP 상위 기술자와 화학 상식이 맞는 것/안 맞는 것을 하나씩 말할 수 있다
- [ ] 31쪽 질문(정확도 vs 설명)에 mp 관점의 답이 있다
- [ ] "후분석 결과 → 2주차에 무엇을 바꿀지" 한 줄이 있다 (49쪽 2주차 예고: readout, 층 수, 결합 특징)
