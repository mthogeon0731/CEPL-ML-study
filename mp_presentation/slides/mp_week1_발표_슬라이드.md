# 발표 슬라이드 — Claude 슬라이드 아티팩트 (2026-11-09)

**덱 링크**: https://claude.ai/artifact/XSxjVS9df9pVJHw7Rxivp5 (브라우저에서 편집 · 발표 모드 · Share › Export로 PPTX/PDF 내보내기)
그림 원본: `figs/` (이 폴더), 수치 원본: `../results/`. 슬라이드 순서 = 11쪽 흐름.

| # | 슬라이드 | 핵심 한 줄 | 그림 | 시간 |
|---|---|---|---|---|
| 1 | 표지 — mp(녹는점) 예측 | 11쪽 흐름 5단계 띠 | – | 0:10 |
| 2 | ① 데이터: 19,945개, K, 잡음 큰 물성 | 반복 측정 86종 ≥ 50 K → 오차 하한; [추가] 결정 성질 | `01_data/fig_mp_hist.png` | 0:30 |
| 3 | ② 전처리: DB 정제 끝, 표현 3종 × 분할 2종 | random은 test 골격 79 % 기지, scaffold 0 % | – (표) | 0:40 |
| 4 | ③ 임베딩·모델링: 단순한 것부터, GIN | 6모델 가설 표 + 21쪽 식 + 147,009 파라미터 | – (표) | 0:40 |
| 5 | ④ 성능 — 결과표 | XGBoost+기술자 30.0 / 37.8 K, 기준선 72.5 / 76.0, 4칸 ✗, 12쌍 scaffold > random | `figs/fig_S4_results_table.png` | 0:40 |
| 6 | ④ 성능 — GIN 그림 2장 + 진단 | 36.5 K; 가벼운 과적합 + 조기 종료(114/144); 끝 영역 눌림 | `results/mp_GNN-GIN_mean_random_curve.png`, `_parity.png` | 0:40 |
| 7 | ⑤ 분석 ① 구간별 오차 | 300–450 K 27–32 K, > 500 K 과소·< 250 K 과대, scaffold 끝 82 K | `figs/fig_S6_error_by_range_tall.png` | 0:25 |
| 8 | ⑤ 분석 ② 틀린 분자 | 값 오류 의심(끓는점·염) vs 모델 한계(대칭 고융점) | `figs/fig_S7_worst_molecules_row.png` | 0:25 |
| 9 | ⑤ 분석 ③ SHAP | TPSA·RingCount·NumHDonors ↑, NumRotatableBonds ↓ — 설명 검증 통과; 31쪽 답 | `figs/fig_S6_shap_top10.png` | 0:30 |
| 10 | 코드 4줄 · 결론 · 2주차 | 같은 test: XGB+기술자 32.7 vs GIN 36.5 K (3위); sum readout → GINE → 전체 데이터 | `figs/fig_S7_same_split_ranking.png` | 0:40 |

합계 ≈ 5:20 → 리허설에서 3·4장을 줄이면 5:00. 발표자 노트(아티팩트의 각 슬라이드 `aside`)에 말할 문장이 들어 있다.

## 규칙 점검

- 수치 옆에 분할(random/scaffold)·단위(K) — 모든 슬라이드에서 지킴 (14쪽)
- 기준선보다 나쁜 4칸, 과적합 그림 그대로 (24쪽) — 5·6장
- 슬라이드에 없는 해석은 “[추가]”로 표시 — 2·7·8·9·10장
- 그림은 `results/` 원본 그대로 (GIN 2장) 또는 원본 CSV만 읽어 그린 것 (`scripts/make_slide_figs.py`)

## 예상 질문 답 (slides/README.md의 5개)

1. 왜 scaffold에서 더 나쁜가 → test 1,000개가 모두 1개짜리 골격(처음 보는 뼈대); random은 79 %를 train에서 봤다. 차이는 +7 K(XGB·RF+기술자) ~ +76 K(MLP+Morgan).
2. 튜닝을 왜 안 했나 → 24칸을 같은 조건에서 비교하는 공정한 기준선(37쪽); test 보고 설정을 고르지 않는 규칙(49쪽).
3. mp가 왜 어려운가 → 반복 측정 86종 ≥ 50 K; [추가] 결정 성질(다형·순도·분해), 2D 표현에 패킹 정보 없음 — Mirex·쿼터페닐이 그 예.
4. GIN이 고전 모델보다 못한 이유 → 같은 test 500에서 3.8 K 뒤짐; 5,000개는 19쪽 “1만 개 이상”에 못 미침; 지문 모델 6종은 모두 이김. scaffold에서는 validate 잡음으로 조기 종료가 45 epoch에 일찍 걸림.
5. 다음에 바꿀 것 → readout sum(작은 분자 오차) → GINE(결합 특징; 회전 가능 결합 SHAP 6위) → 전체 19,945개.
