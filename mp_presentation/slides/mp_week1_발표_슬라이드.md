# 발표 슬라이드 — Claude 슬라이드 아티팩트 (2026-11-09)

**덱 링크**: https://claude.ai/artifact/XSxjVS9df9pVJHw7Rxivp5 (브라우저에서 편집 · 발표 모드 · Share › Export로 PPTX/PDF 내보내기)
그림 원본: `figs/` (이 폴더), 수치 원본: `../results/`. 슬라이드 순서 = 11쪽 흐름.

| # | 슬라이드 | 핵심 한 줄 | 그림 | 시간 |
|---|---|---|---|---|
| 1 | 표지 — mp(녹는점) 예측 | 11쪽 흐름 5단계 띠 | – | 0:10 |
| 2 | ① 데이터: 19,945개, K, 잡음 큰 물성 | 반복 측정 86종 ≥ 50 K → 오차 하한; [추가] 결정 성질 | `01_data/fig_mp_hist.png` | 0:30 |
| 3 | ② 전처리: DB 정제 끝, 표현 3종 × 분할 2종 | random은 test 골격 79 % 기지, scaffold 0 % | – (표) | 0:40 |
| 4 | ③ 임베딩·모델링: 단순한 것부터, GIN | 6모델 가설 표 + 21쪽 식 + 147,009 파라미터 | – (표) | 0:40 |
| 5 | ④ 성능 — 결과표 | XGBoost+기술자 **29.5 / 37.4 K** (PC), 기준선 72.5 / 76.0, 4칸 ✗, 12쌍 scaffold > random | `figs_pc/fig_S4_results_table.png` | 0:40 |
| 6 | ④ 성능 — GIN 그림 2장 + 진단 | **36.0 K** (RMSE 49.2, R² 0.73); 가벼운 과적합 + 조기 종료(**48/78**, 79 epochs, 289 s); validate 최저 34.5; >500 K −58 K (n 41), <250 K +36 K (n 51) | `results_pc_v2/train_py_results/mp_GNN-GIN_mean_random_curve.png`, `_parity.png` | 0:40 |
| – | **— 아래 7·8·9장은 백업 (2026-11-09 검토 후): 덱 끝으로 옮기고 질문이 나오면 연다 —** | | | |
| 7 | ⑤ 분석 ① 구간별 오차 | 300–450 K 27–30 K (scaffold 300–350만 37), > 500 K 과소·< 250 K 과대, 550–600 K 57 → 84 K, RMSE−MAE 12–14 | `figs_pc/fig_S6_error_by_range_tall.png` | 0:25 |
| 8 | ⑤ 분석 ② 틀린 분자 | 값 오류 의심(끓는점·염) vs 모델 한계(대칭 고융점); XGB scaffold 상위 20 중 18 과소, GIN 상위 20 중 11 비고리(MW 중앙값 185) | `figs_pc/fig_S7_worst_molecules_row.png` | 0:25 |
| 9 | ⑤ 분석 ③ SHAP | BertzCT·TPSA·NumHDonors·RingCount ↑, FractionCSP3·NumRotatableBonds ↓ — 두 분할 상위 7 동일; 설명 검증 통과; 31쪽 답 | `figs_pc/fig_S6_shap_top10.png` | 0:30 |
| 10 | 코드 4줄 요약 · 결론 · 2주차 | 같은 test 500: XGB+기술자 **33.6 vs GIN 36.0 K** (2.4 K, CI [0.0, 4.7] 경계선), scaffold 36.5 vs 42.1 (5.6 K, 견고); 2주차 후보는 가설(validate 확인 후) | `figs_pc/fig_S7_same_split_ranking.png` | 0:40 |

합계 ≈ 5:20 → 리허설에서 3·4장을 줄이면 5:00. 발표자 노트(아티팩트의 각 슬라이드 `aside`)에 말할 문장이 들어 있다.

**2026-11-09 검토 후 구성**: 본 발표 = 1·2·3·4·5·6·10장 ≈ **4:00** (49쪽 요건: 표 1 + GIN 그림 2 + 코드 요약), 7·8·9장(1:20) = 백업. 남는 1분은 6장(GIN 진단)과 10장(결론)에 나눠 쓴다. 덱 아티팩트(claude.ai)는 이 폴더에서 고칠 수 없으므로 **덱에서 7–9장을 끝으로 옮기는 작업은 직접** 해야 한다.

**10장 결론 문장 보정**: "같은 test 500에서 XGB+기술자 32.7 vs GIN 36.5 K"는 컨테이너 실행 값이다. PC `mlstudy` 재현에서는 33.6 vs 36.0 K(차이 2.4 K, 부트스트랩 95 % CI [0.01, 4.67] — 경계선)이고 scaffold는 두 환경 모두 GIN이 5.4–5.6 K 뒤짐(CI 하한 ≈ 2.9 K). 발표 문장: **"random에서는 GIN이 2–4 K 뒤지지만 차이가 경계선이고, 새 골격(scaffold)에서는 분명히 5 K 이상 뒤진다."** (`../04_performance/results_table.md` 부트스트랩 표, `../results/run_notes.md` PC 재현 절)

**2026-11-09 밤 — PC판 채택·덱 교체 완료**: 덱(https://claude.ai/artifact/XSxjVS9df9pVJHw7Rxivp5) 5·6·7·8·9·10장의 수치·그림·발표자 노트를 PC `mlstudy` v2 수치로 교체했다 (그림 `slides/figs_pc/`, 후분석 `05_analysis_xai_pc/`, 스크립트 `scripts/make_slide_figs_pc.py`). 6장 표의 끝 영역 수치는 컨테이너판의 bin 1개 값(−41 K n 29, +39 K n 44)이 아니라 전 구간 값(>500 K −58 K n 41, <250 K +36 K n 51)으로 바로잡음. 8장 GIN 상자의 바이닐 뷰티레이트·2,2-다이클로로펜테인은 XGBoost random의 틀린 분자였으므로 출처를 바로 적음. 7–9장 백업 이동은 아직 안 함(순서 그대로).

**6장 캡션 (그림 축에 단위 없음)**: "mp [K], `train.py` random 80/10/10, test 500, 컨테이너 실행 (PC 재현 35.97 K)".

## 규칙 점검

- 수치 옆에 분할(random/scaffold)·단위(K) — 모든 슬라이드에서 지킴 (14쪽)
- 기준선보다 나쁜 4칸, 과적합 그림 그대로 (24쪽) — 5·6장
- 슬라이드에 없는 해석은 “[추가]”로 표시 — 2·7·8·9·10장
- 그림은 `results/` 원본 그대로 (GIN 2장) 또는 원본 CSV만 읽어 그린 것 (`scripts/make_slide_figs.py`)

## 예상 질문 답 (slides/README.md의 5개)

1. 왜 scaffold에서 더 나쁜가 → test 1,000개가 모두 1개짜리 골격(처음 보는 뼈대); random은 79 %를 train에서 봤다. 차이는 +7 K(XGB·RF+기술자) ~ +76 K(MLP+Morgan).
2. 튜닝을 왜 안 했나 → 24칸을 같은 조건에서 비교하는 공정한 기준선(37쪽); test 보고 설정을 고르지 않는 규칙(49쪽).
3. mp가 왜 어려운가 → 반복 측정 86종 ≥ 50 K; [추가] 결정 성질(다형·순도·분해), 2D 표현에 패킹 정보 없음 — Mirex·쿼터페닐이 그 예.
4. GIN이 고전 모델보다 못한 이유 → 같은 test 500에서 random 2.4 K(PC; 짝지은 부트스트랩 95 % CI [0.0, 4.7] → 경계선, 분자별로는 GIN이 나은 분자 47 %), scaffold 5.6 K(CI [2.9, 8.1], 견고) 뒤짐; RF+기술자와는 0.2 K 차이; 5,000개는 19쪽 “1만 개 이상”에 못 미침; 지문 모델 6종은 모두 이김. scaffold에서는 validate 잡음으로 조기 종료가 45 epoch에 일찍 걸림. 같은 seed라도 환경에 따라 epoch 수(컨테이너 145 / PC 79)와 MAE 0.5 K가 움직임 → 소수점 첫째 자리는 의미 없음.
5. 다음에 바꿀 것 → readout sum(작은 분자 오차) → GINE(결합 특징; 회전 가능 결합 SHAP scaffold 7위 / random 6위) → 전체 19,945개.
