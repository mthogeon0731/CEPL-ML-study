# CEPL ML study

KIST CEPL 학부연구생 ML 스터디(2026-11~) — **분자 물성 예측** 공부·과제 기록. 팀 물성은 **mp(녹는점)**.

| 바로 가기 | 내용 |
|---|---|
| [`mp_presentation/README.md`](mp_presentation/README.md) | 1주차 발표 구성(11쪽 흐름 = 폴더 순서), 체크리스트 |
| [`mp_presentation/04_performance/results_table.md`](mp_presentation/04_performance/results_table.md) | 결과표: 모델 6 × 표현 2 × 분할 2 + GIN, test MAE [K] |
| [`mp_presentation/03_modeling/mp_모델 7종_학습노트.pdf`](mp_presentation/03_modeling/mp_모델%207종_학습노트.pdf) | 모델 7종 학습 노트 (34쪽): 학습 원리·장단점·mp에서의 정합성 |
| [`mp_presentation/05_analysis_xai/xai_notes.md`](mp_presentation/05_analysis_xai/xai_notes.md) | 후분석(틀린 분자·구간별 오차) + SHAP |
| [`mp_presentation/slides/mp_week1_발표_슬라이드.md`](mp_presentation/slides/mp_week1_발표_슬라이드.md) | 발표 슬라이드 10장 구성·링크·예상 질문 |
| [`mp_presentation/results/run_notes.md`](mp_presentation/results/run_notes.md) | 모든 수치의 출처·실행 환경·재현 명령 |
| [`mp_week2/README.md`](mp_week2/README.md) | **2주차** (GIN · 불확실성 · XAI) mp 재실행 결과 + 덱 정리(학습 설정 총정리) + mp XAI 노트 |
| [`mp_mae_push/REPORT.md`](mp_mae_push/REPORT.md) | **MAE 개선 실험** (Claude Science): test MAE 기준선 27.50 / 35.85 K → 가중 앙상블 24.75 / 33.65 K (random / scaffold). 지시서 `00`·`01`·`02`, 랩미팅 그림 `figs_labmeeting/` |

**1주차 결과 한 줄** (표본 5,000, seed 42, test MAE K, **PC `mlstudy` 실행 = 발표 채택 수치**): 기준선 72.5 / 76.0 (random / scaffold) → 최저 **XGBoost + RDKit 기술자 29.5 / 37.4**; GIN(`train.py`, 80/10/10) 36.0 — 같은 test 500개에서 XGBoost+기술자 33.6 vs GIN 36.0 (3위; 차이 2.4 K는 부트스트랩 95 % CI [0.0, 4.7]로 경계선, scaffold에서는 36.5 vs 42.1로 5.6 K 뒤짐; 지문 모델 6종은 모두 이김).

**수치 두 벌**: `mp_presentation/results/`(Claude 컨테이너 실행, 11-09 오전)와 `mp_presentation/results_pc_v2/`(PC `mlstudy` 재현, 11-09 오후). 결정적 모델은 동일, XGBoost·RF ≤ 0.9 K, GIN은 epoch 수가 다름(145 → 79, MAE 0.5 K). **발표(덱·`slides/figs_pc/`·`05_analysis_xai_pc/`)는 PC판 기준** — `04_performance/results_table.md` ★ 절, `results/run_notes.md` PC 재현 절, 비교표 `results_pc_v2/compare_container_vs_pc.csv`.

> 저장소에 **없는 것**: 강의 슬라이드 PDF(`261105_1주차_물성예측_슬라이드.pdf`), `Context/`의 참고문헌 PDF, 실습 zip — 저작권·용량 때문에 제외(`.gitignore`). `261105_Week1_Practice/`의 코드·데이터는 교수님 실습 자료이며 `data/README_data.md`의 원 출처를 인용할 것.

---

## 작업 폴더 지도 (로컬 `C:\Users\mthog\CEPL ML\` 기준 — 2026-11-09부터; 이전 OneDrive 바탕 화면 폴더는 사본)

- 스터디 자료: `261105_1주차_물성예측_슬라이드.pdf` (50쪽, 최재형, CEPL, 2026.11)
- 팀 물성: **mp (녹는점)** — 19,945개, K, 어려움(잡음 큼) (48쪽)
- 다음 주: 팀별 5분 발표 = **11쪽 전체 흐름** 순서 (표 1장 + 그림 2장 + 코드 요약, 49쪽)

## 폴더 지도

```
CEPL ML/
├─ README.md                         ← 이 파일
├─ 261105_1주차_물성예측_슬라이드.pdf     원본 슬라이드
├─ Claude outputs/                   학습지침(1–10쪽), 학습코치 프롬프트
├─ Context/                          40–41쪽 참고자료 PDF (README: 목록·상태·발표 절별 읽을 곳)
├─ env/                              42–46쪽 환경 설정 (한 번) — 00_설치안내.md 부터
├─ 261105_Week1_Practice/            47쪽 교수님 실습 폴더 자리 — zip을 여기에 풀기 (README: 파일 구성, 과제 4단계)
├─ 261109_Week2_Practice/            2주차 실습 꾸러미 (+ [추가] *_mp.ipynb · *_executed · mp_xai_ext.py 등)
├─ mp_week2/                         2주차 결과·공부 자료 (00 덱 정리 → 01 GIN → 02 불확실성 → 03 XAI)
├─ mp_mae_push/                      MAE 개선 실험 (Claude Science) — 지시서 00·01·02 → phase1–3 → REPORT.md, figs_labmeeting/
└─ mp_presentation/                  발표 작업 — 11쪽 흐름 = 폴더 순서
   ├─ README.md                      발표 구성(절 ↔ 폴더 ↔ 코드 함수), 5분 시간표, mp 특이사항, 체크리스트
   ├─ 01_data/                       SMILES 수집 — 데이터 카드 (무엇·몇 개·단위·분포·왜 어려운가)
   ├─ 02_preprocessing/              정제·표현·분할 — 어떻게 했나, 어떤 표현자(feature)
   ├─ 03_modeling/                   모델 — 왜 썼나, 어떤 특징 (6종 + GIN), 코드 4파일 요약
   ├─ 04_performance/                평가 — MAE 표, parity·학습곡선, 과소·과적합 진단
   ├─ 05_analysis_xai/               후분석(틀린 분자·골격·구간별 오차) + XAI(SHAP) — 컨테이너 예측 기준
   ├─ 05_analysis_xai_pc/            후분석 — PC 예측 기준 (발표 채택)
   ├─ results/                       컨테이너 실행 산출물 원본 (표·그림·예측 CSV, run_notes.md)
   ├─ results_pc_v2/                 PC mlstudy 재현 산출물 (노트북·train.py·같은-분할 비교·부트스트랩·로그)
   ├─ scripts/                       [추가] 데이터 카드 / 오차 분석 / SHAP / 같은-분할 비교 / 슬라이드 그림 스크립트
   └─ slides/                        10장 구성·예상 질문 (mp_week1_발표_슬라이드.md), figs/ 컨테이너판 · figs_pc/ PC판 그림
```

## 발표 흐름 ↔ 폴더 ↔ 슬라이드 (11쪽)

| 11쪽 단계 | SMILES 수집 | 정제 | 표현 | 분할 | 모델 | 평가 | (31쪽) 설명 |
|---|---|---|---|---|---|---|---|
| 2024 코드 함수 | `load_data` | `gen_dataset` | `mol_to_features` | `split_dataset` | `BGNN (GIN)` | `parity, MAE` | 4주차 SHAP |
| 폴더 | `01_data` | `02_preprocessing` | `02_preprocessing` | `02_preprocessing` | `03_modeling` | `04_performance` | `05_analysis_xai` |
| 발표 질문 | 데이터는 무엇 | 어떻게 정제 | 어떤 feature | 어떻게 나눔 | 왜 이 모델·특징 | 좋나 나쁘나, 과소·과적합 | 어디서 틀리나, 무엇을 보나 |
| 근거 쪽 | 2, 12, 13, 48 | 12, 13 | 15–18 | 14 | 19–22 | 9, 10, 23–24, 37–38 | 13, 14, 20, 22–24, 31, 49 |

## 시작 순서

1. **환경** — `env/00_설치안내.md` → PowerShell에서 `env\setup_all.ps1` (Miniforge 없을 때) 또는 Miniforge Prompt에서 `env\setup_env.bat` (있을 때) → `python env\check_env.py` 가 `ok`
2. **실습 폴더** — 교수님 `261105_Week1_Practice.zip`을 `261105_Week1_Practice/`에 풀기 → 그 안에서 `python feature.py` (44쪽 기대 출력)
3. **참고자료** — Chrome 다운로드 폴더의 `p40-*`, `p41-*` PDF를 `Context/`로 이동 (`Context/README.md` 첫 절)
4. **과제** (49쪽 4단계, 팀 물성 mp) — 노트북 `PROP = 'mp'` 결과표 → `python train.py --prop mp` 그림 2장 → 코드 4파일 한 줄 → 산출물은 `mp_presentation/results/`
5. **발표 준비** — `mp_presentation/01…05` README의 질문에 답 채우기 → 덱 10장(Claude 슬라이드, `slides/mp_week1_발표_슬라이드.md`) → 5분 리허설

## 원칙 (슬라이드에서)

- 수치는 **분할 종류와 단위(K)** 를 항상 함께 (14쪽)
- test 성능을 보고 설정을 고르지 않음 (49쪽); 지는 케이스도 그대로 (24쪽)
- 슬라이드에 없는 해석·스크립트는 **[추가]** 로 구분
