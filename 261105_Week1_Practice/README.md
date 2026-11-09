# 261105_Week1_Practice — 교수님 실습 폴더 (슬라이드 47쪽) + mp 과제 메모

`261105_Week1_Practice.zip`을 여기에 풀어 둔 상태 (2026-11-06). 교수님의 원본 README는 이름이 겹쳐 `README_교수님원본.md`로 두었다 — **과제 규칙은 그 파일이 기준**. 이 파일은 mp 팀용 메모.

| 파일 | 하는 일 (47쪽) | 실제 코드에서 확인한 것 |
|---|---|---|
| `00_설치안내.md` | Miniforge, conda 환경, Jupyter, VS Code, Colab, 문제 해결 표 | `../env/00_설치안내.md`는 이 PC용 요약, 원본은 여기 |
| `00_python_basics.ipynb` | 예습: 8개 패키지를 logP로 한 셀씩 | 10분 |
| `01_multi_model_comparison.ipynb` | 오늘 실습: 표현 2종 × 모델 6종 × 분할 2종 결과표 | 첫 셀 `PROP`, `MAX_N = 5000` (수업용 상한, `None`이면 전체 19,945 — SVR 느려짐), `SEED = 42`. 분할 **80/20** (random: `train_test_split`, scaffold: 희귀 골격부터 test). 결과 `results_<PROP>.csv` 저장 |
| `feature.py` | 분자 → 그래프 (원자 147, 결합 16) | 원자 = 원소 11 + 전하 5 + 혼성 4 + H수 4 + 방향족 1 + 원자가 6 + 연결수 4 + 고리 1 + 반응중심 1 + 작용기 SMARTS ~110; 결합 = 종류 4 + 공액 1 + 회전/비고리/고리밖 3 + 양끝 원소 4+4 |
| `datamanager.py` | CSV 읽기 → 그래프 목록 → 80:10:10 분할 → y 표준화 | `load_data(prop, max_n)`, `gen_dataset`, `split_dataset(how='random'/'scaffold')` (test = validate = 10%; scaffold는 작은 골격부터 test → validate → 나머지 train), `Scaler` |
| `models.py` | GIN 모델 (연구실 BGNN_unary와 같은 구조) | x(147) → GIN 5층 (hidden 128, dropout 0.3) → 원자 벡터 64 → `global_mean_pool` → Linear(64→1). **PyG GIN은 edge_attr(결합 16)을 쓰지 않음** → `--conv GINE`이면 사용 |
| `train.py` | GNN 학습 한 번 → 결과표 한 줄 + 그림 | 기본 `--max_n 5000` (`0` = 전체), epochs ≤ 300, AdamW 1e-3, ReduceLROnPlateau(10, ×0.8), 기울기 ±10, 조기 종료 = validate MAE 30 epoch 정체, batch 64 (N<1만) / 256 (<10만). Scaler는 **train으로만 fit** (연구실 원본은 train+val+test — 14쪽 누수 예). 출력: `results/summary.csv` 한 줄(MAE, RMSE, R2, baseline_MAE, epochs, sec), `results/<이름>_parity.png`, `_curve.png`, **`_test_pred.csv` (smiles, y, pred, sd)** |
| `data/mp.csv` | 19,945행, 열 `smiles, value, unit(K), source` | 출처·정제 = `data/README_data.md` (아래 요약) |

## mp 데이터 요약 (`data/README_data.md`에서)

- 출처: Bradley, Williams & Lang, *Jean-Claude Bradley Open Melting Point Dataset*, figshare (2014) [4]. `source` 열은 모두 `Bradley_ONS_MP` (세부 출처 PHYSPROP·Alfa Aesar 등은 원본 열에만)
- 정제: 원본 27,976 → NaN 0, 잘못된 SMILES 0, **중복 병합 8,031 (중앙값)**, `donotuse` 행 제거 → 19,945. 값 범위 68 ~ 790 K
- **잡음**: 반복 측정이 있는 4,679종 중 410종은 값 차이 ≥ 10 K, 86종은 ≥ 50 K → 중앙값으로 합쳤지만 **이 잡음이 오차 하한**
- 난이도 "어려움": 결정 패킹은 분자 그래프로 잘 안 보이고 측정 잡음이 크다
- 팀 배정 제안(README_data §3): `logSolubility` + `mp` — 일반 용해도 식 logS ≈ 0.5 − 0.01(mp[°C] − 25) − logP 로 세 물성을 엮을 수 있음 → 발표 1절 "mp가 왜 중요한가"의 한 줄

## 1주차 과제 4단계 (49쪽, `README_교수님원본.md` §4) — 팀 물성 = mp

| 단계 | 할 일 | 명령 / 위치 | 제출물 → 보관 |
|---|---|---|---|
| 1 결과표 | 노트북 첫 셀 `PROP = 'mp'` → 끝까지 실행, 기준선 MAE와 비교 | Jupyter: `01_multi_model_comparison.ipynb` | `results_mp.csv` + 결과표 셀 캡처 → `../mp_presentation/results/` |
| 2 GNN 그대로 | 코드 수정 없이 한 번 실행 (CPU 약 3분; `--max_n 5000` 기본) | `python train.py --prop mp` | `results/mp_GNN-GIN_mean_random_parity.png`, `_curve.png` → `../mp_presentation/results/` |
| 3 코드 읽기 | 4파일 한 줄씩 | 위 표의 "실제 코드에서 확인한 것" 참고 | `../mp_presentation/03_modeling/code_summary.md` |
| 4 발표 | 1의 최고 모델 vs 2의 GNN MAE (⚠ 노트북 test 20% vs train.py test 10% — 같은 5,000개 표본이지만 test 집합은 다름) | 표 1장 + 그림 2장 + 코드 요약 | `../mp_presentation/slides/` |

- 선택 (노트북 9절): `MAX_N = None` 전체 데이터 (learning curve), Morgan radius 1/2/3, descriptor+fingerprint `np.hstack`, **RF `feature_importances_` 상위 10 descriptor 화학 해석**, 틀린 분자 5개 그리기 (마지막 셀)
- 2주차 예고: `train.py`에 이미 옵션이 있음 — `--split scaffold`, `--conv GINE` (결합 특징), `--pooling sum` (GIN 논문 readout), `--bayes` (BGNN)
- **규칙: test 성능을 보고 설정을 고르지 않음** (튜닝은 train 안에서 `cross_val_score`)

## 실행 순서

```
conda activate mlstudy
cd "C:\Users\mthog\CEPL ML\261105_Week1_Practice"
python -c "import rdkit, sklearn, xgboost; print('ok')"
python feature.py            # piperidine  x: (6, 147)  edge_index: (2, 12)  edge_attr: (12, 16)
jupyter notebook             # → 01_multi_model_comparison.ipynb, 첫 셀 PROP = 'mp'
python train.py --prop mp    # 과제 2단계 → results/
```
