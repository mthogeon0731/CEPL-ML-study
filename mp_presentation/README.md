# mp_presentation — 녹는점(mp) 예측 발표 (다음 주, 팀별 5분)

발표 구성 = 슬라이드 **11쪽 "전체 흐름"** 순서. 폴더 번호 = 발표 순서.

```
SMILES 수집 → 정제 → 표현 → 분할 → 모델 → 평가        (11쪽)
   01_data   └──── 02_preprocessing ────┘  03_modeling  04_performance  + 05_analysis_xai (31쪽, 4주차 예고)
```

| 절 | 폴더 | 11쪽 단계 | 2024 연구실 코드 함수 (11쪽 회색 글씨) | 발표에서 답할 질문 |
|---|---|---|---|---|
| 1 데이터 | `01_data/` | SMILES 수집 | `load_data` | mp 데이터는 무엇이고(출처), 몇 개, 단위, 분포, 왜 어려운가 |
| 2 전처리 | `02_preprocessing/` | 정제 → 표현 → 분할 | `gen_dataset`, `mol_to_features`, `split_dataset` | 어떻게 정제했나, 어떤 표현자(feature)를 썼나, 어떻게 나눴나 |
| 3 임베딩·모델링 | `03_modeling/` | 모델 | `BGNN (GIN)`, Ridge…XGBoost | 왜 그 모델들을 썼나, 각 모델의 특징은 |
| 4 성능 | `04_performance/` | 평가 | `parity`, `MAE` | 좋은가 나쁜가(기준선 대비), 과소·과대적합이 있는가 |
| 5 모델 분석 | `05_analysis_xai/` | (31쪽 정확도 vs 설명) | — (4주차 SHAP·원자 기여) | 어디서 틀리나(후분석), 무엇을 보고 예측하나(XAI) |
| 산출물 | `results/` | | 노트북·`train.py` 출력 | 표 1장 + 그림 2장 (49쪽) |
| 슬라이드 | `slides/` | | | 5분 발표 파일 |
| 도구 | `scripts/` | | [추가] 데이터 카드·오차 분석·SHAP 스크립트 | 1, 5절 재료 |

각 폴더의 `README.md`에 **① 답할 질문 → ② 슬라이드 근거(쪽) → ③ 만들 산출물 → ④ 작업 순서 → ⑤ 자가점검**이 있다.
슬라이드에 없는 내용은 **[추가]** 로 표시.

## 발표 형식 (49쪽 4단계)

5분 = **표 1장** (모델 6 × 표현 2 × 분할 2, test MAE [K]) + **그림 2장** (GNN parity, 학습곡선) + **코드 요약** (파일 4개 한 줄씩) + 1의 최고 모델 vs 2의 GNN MAE 비교.

| 시간 | 절 | 슬라이드 (`slides/README.md`) | 재료 |
|---|---|---|---|
| 0:00–0:30 | 1 데이터 | S1 문제·데이터 카드 | `01_data/data_card.md`, 분포 그림 |
| 0:30–1:20 | 2 전처리·표현 | S2 정제 로그 + 표현 3종 표 | `02_preprocessing/preprocessing_log.md` |
| 1:20–2:00 | 3 모델 | S3 모델 선택 이유 (19쪽 표를 mp에 맞춰) | `03_modeling/model_settings.md` |
| 2:00–3:20 | 4 성능 | S4 결과표 1장, S5 GNN 그림 2장 + 진단 | `results/`, `04_performance/diagnosis.md` |
| 3:20–4:20 | 5 후분석·XAI | S6 틀린 분자·골격별 오차 (+SHAP 1장) | `05_analysis_xai/` |
| 4:20–5:00 | 코드·결론 | S7 코드 4줄 요약, 최고 모델 vs GNN, 2주차에 바꿀 것 | `03_modeling/code_summary.md` |

## mp(녹는점)에 대해 슬라이드가 이미 말해 주는 것 — 발표 전체에서 일관되게

| 항목 | 내용 | 쪽 |
|---|---|---|
| 데이터 수 · 단위 · 난이도 | **19,945개, K (°C 아님), 어려움 (잡음 큼)** — 9종 중 가장 많지만 6번 추천. 노트북·`train.py`는 **5,000개 표본**(seed 42)이 기본 | 48, 노트북 1절 |
| 공정에서의 쓰임 | 결정화, 고체 취급 → 녹는점; 아민 후보 거르는 기준 | 2 |
| 2024 전처리 | 'do not use' 행 삭제, 열 이름 정리 (**mpc → mp**), **°C → K**, SMILES 정리 후 `dataset.pkl` → 이번 `data/mp.csv`는 그 결과(원본 27,976 → 중복 8,031 병합 → 19,945, Bradley Open MP) | 12, README_data |
| 목표값 변환 | bp, mp: 흔한 원 단위 °C·K → 모델 목표값 **K 그대로 (log 변환 없음)**; 같이 기록할 것: 분해 여부 | 13 |
| 2024 BGIN 덱 자료 수 | mp train 22,373 / test 5,593 — 데이터 수 ≠ 정확도 순서 (물성의 측정 잡음·구조 의존성이 다름) | 26 |
| GNN 배치 크기 | `train.py` 기본 표본 5,000 (< 1만) → **64**; `--max_n 0` 전체 19,945 (< 10만) → 256 | 21, train.py |
| 기준선 | train 평균으로 예측한 MAE — 모델이 이보다 얼마나 낮은지 | 23 |
| 규칙 | test 성능을 보고 설정을 고르지 않음; 지는 케이스도 그대로 보여줌 | 49, 24 |

**[추가] 왜 mp가 "잡음 큼"인가** — 발표 1절·5절의 해석 근거로 쓰되, 슬라이드에 없는 내용이므로 [추가]로 말한다:
녹는점은 분자 하나의 성질이 아니라 **결정(고체)의 성질**이다. 같은 분자라도 결정 다형(polymorph), 순도, 측정법(모세관 vs DSC), 분해(decomposition)에 따라 값이 수십 K씩 달라진다. 2D 그래프·지문·기술자는 결정 패킹·대칭성·분자간 수소결합을 직접 담지 못하므로, 입력이 같은데 y가 다른 "모델이 설명 못 하는 잡음"(12쪽 입체이성질체와 같은 논리)이 크다. → 어떤 모델도 MAE 하한이 있고, 과적합 진단 때 "train 오차가 작아도 test 오차가 큰 것 = 잡음 + 과적합"을 분리해서 말해야 한다.

## 체크리스트 (발표 전)

- [ ] `env/` 설치 완료, `python feature.py` 출력 확인 (44쪽)
- [x] 실습 zip을 `261105_Week1_Practice/`에 풀었음 (2026-11-06) — `data/README_data.md`의 mp 항목은 `261105_Week1_Practice/README.md`에 요약
- [x] 과제 1: 노트북 `PROP = 'mp'` → 결과표 → `results/` (2026-11-09, `results/results_mp.csv`·`mae_table_mp.csv`; 실행된 노트북 `261105_Week1_Practice/01_multi_model_comparison_mp_executed.ipynb`)
- [x] 과제 2: `python train.py --prop mp` → parity·학습곡선 → `results/` (2026-11-09, test MAE 36.5 K; `--split scaffold`도 [추가] 실행 41.8 K)
- [x] 과제 3: 코드 4파일 한 줄 요약 → `03_modeling/code_summary.md`
- [x] 1–5절 README의 질문에 답 채우기 → `02_preprocessing/preprocessing_log.md`·`representation_table.md`, `03_modeling/model_settings.md`, `04_performance/results_table.md`·`diagnosis.md`, `05_analysis_xai/xai_notes.md` (수치 출처 `results/run_notes.md`)
- [x] `slides/` — Claude 슬라이드 아티팩트 10장 (`slides/mp_week1_발표_슬라이드.md`에 링크·장별 구성) · [ ] 5분 리허설 (현재 ≈ 5:20, 3·4장 줄이기)
- [ ] 수치 옆에 분할 종류(random/scaffold)와 단위(K)를 항상 함께 (14쪽 "어떤 분할에서 나온 점수인지 함께 보고")
