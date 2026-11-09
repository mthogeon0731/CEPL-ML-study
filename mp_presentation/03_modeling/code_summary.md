# 코드 요약 — 과제 3단계 (49쪽): 파일 4개, 각 한 줄

작성 2026-11-09. `261105_Week1_Practice/` 코드를 **읽고 실행해서** 확인한 내용 (실행 결과는 `../results/train_log_gin_mp.txt`).

## 한 줄 요약 (발표 S7)

| 파일 | 하는 일 (입력 → 출력) | 11쪽 단계 |
|---|---|---|
| `feature.py` | RDKit Mol → 그래프 텐서: 원자 x (n_atoms × **147**), edge_index (2 × 2·n_bonds), 결합 edge_attr (2·n_bonds × **16**). 원자 147 = 원소 11 + 전하 5 + 혼성 4 + H 수 4 + 방향족 1 + 원자가 6 + 연결수 4 + 고리 1 + 반응중심 1 + 작용기 SMARTS **110** | 표현 |
| `datamanager.py` | `data/mp.csv` → (`max_n` 표본, seed 42) → PyG `Data` 목록 → **80:10:10** random/scaffold 분할 → y 표준화 `Scaler` (train 통계로만 fit) | 정제·분할 |
| `models.py` | `GraphRegressor`: GIN 5층 (hidden 128, 마지막 층 출력 64, dropout 0.3) → `global_mean_pool` → Linear(64→1). `bayes=True`면 마지막 층만 확률 가중치(BGNN). **PyG GIN은 edge_attr를 쓰지 않음** (`--conv GINE`이면 사용) | 모델 |
| `train.py` | 학습 한 번 = 결과표 한 줄: scaled-MSE 손실, AdamW 1e-3, ReduceLROnPlateau(10, ×0.8), 기울기 ±10 자르기, validate MAE 30 epoch 정체 시 조기 종료 → 최선 가중치 복원 → `results/summary.csv` + `_parity.png` + `_curve.png` + `_test_pred.csv` | 학습·평가 |

확인 출력 (44쪽): `python feature.py` → `piperidine  x: (6, 147)  edge_index: (2, 12)  edge_attr: (12, 16)` ✔

## 21쪽 GIN 식 ↔ 코드 (자가점검용)

h_v^(k) = MLP^(k)( (1+ε)·h_v^(k−1) + Σ_{u∈N(v)} h_u^(k−1) ),  h_G = READOUT({h_v}) → Linear

| 식의 항 | 코드 (`models.py`, PyG) | 실제 값 |
|---|---|---|
| Σ_{u∈N(v)} h_u — 이웃 **합** (sum 집계 → 이웃 개수 구별, 21쪽 그림) | `GINConv`의 `aggr='add'` (PyG 기본) | — |
| (1+ε)·h_v | `GINConv(eps=0, train_eps=False)` (PyG 기본) → **ε = 0**, 자기 벡터를 그대로 더함 | ε = 0 |
| MLP^(k) | `MLP(in, 128, out)` = Linear → ReLU → Linear (norm 없음) | 층 1: 147→128→128, 층 2–4: 128→128→128, 층 5: 128→64→64 |
| 층 수 k = 1…5 | `GIN(..., num_layers=5)`; 층 사이 ReLU + dropout 0.3 (마지막 층 뒤에는 없음) | 5층 = 이웃 5칸까지 |
| READOUT | `global_mean_pool(z, data.batch)` (`--pooling sum`이면 `global_add_pool`) | mean |
| 선형층 | `self.head = nn.Linear(64, 1)` | 64 → 1 |
| 파라미터 수 | `model.n_params()` | **147,009** (7쪽: 직선 θ 2개 vs GNN θ 10만 개 이상) |

파라미터 147,009 = 층1 (147·128+128) + (128·128+128) = 35,456 / 층2–4 각 32,896 × 3 = 98,688 / 층5 (128·64+64) + (64·64+64) = 12,416 / head 65.

## 학습 루프 (`train.py`) — 9·10쪽과 연결

| 단계 | 코드 | 비고 |
|---|---|---|
| 손실 (학습) | `F.mse_loss(out, batch.y)` — y는 train 평균·표준편차로 표준화된 값 | 9쪽: 학습은 MSE |
| 보고 지표 | `val_mae = mean(|y_raw − inverse_transform(out)|)` — **K 단위** | 9쪽: 보고는 MAE |
| 최적화 | `AdamW(lr=1e-3)` + `ReduceLROnPlateau(patience=10, factor=0.8, min_lr=1e-6)` on validate MAE | w ← w − η·∂L/∂w |
| 기울기 자르기 | `p.register_hook(lambda g: g.clamp(-10, 10))` | 발산 방지 |
| 조기 종료 | validate MAE가 `1e-4` 이상 좋아지지 않는 epoch가 **30번** 쌓이면 중단 → `best_state` 복원 | 10쪽 |
| 배치 | N < 1만 → **64** (표본 5,000) / N < 10만 → 256 (`--max_n 0`, 전체 19,945) | 21쪽 |
| 누수 방지 | `Scaler('standard').fit(sets['train'])` — train만으로 fit | 14쪽 (연구실 원본은 train+val+test) |
| 출력 | `summary.csv` 한 줄(MAE, RMSE, R2, baseline_MAE, epochs, sec) + parity + 학습곡선 + test 예측 CSV | 23쪽 |
