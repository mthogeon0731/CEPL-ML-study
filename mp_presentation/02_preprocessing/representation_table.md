# 표현(feature) 표 — 분자를 어떻게 숫자로 바꿨나 (15–18쪽)

실행 2026-11-09 노트북 3절 출력: `descriptor: (5000, 217)  (30.0 s)`, `Morgan FP : (5000, 2048), 분자당 켜진 비트 평균 25.9개` (min 1, max 95). 그래프: `train.py` 출력 `node features = 147, edge features = 16`.

| 표현 | 차원 (shape) | 넣은 모델 | 스케일링 | mp에서 기대한 것 | 한계 (슬라이드) |
|---|---|---|---|---|---|
| (a) RDKit 기술자 `Descriptors.descList` 전부 (RDKit 2026.09) | **X_desc ∈ ℝ^(5000 × 217)**; NaN·inf → 0, ±1e6 clip | Ridge, kNN, SVR, RF, XGBoost, MLP | `Pipeline` 안 `StandardScaler` (Ridge, kNN, SVR, MLP). RF·XGB는 트리라 불필요 (16쪽) | TPSA·H-donor 수·고리 수·분자량 등 **분자간 상호작용·결정 패킹의 대리변수** | 다른 분자가 같은 값 가능; 기술자끼리 상관 큼 → 계수·중요도 해석 주의 (16쪽) |
| (b) Morgan 지문 `GetMorganGenerator(radius=2, fpSize=2048)` (ECFP4) | **X_fp ∈ {0,1}^(5000 × 2048)**, 분자당 켜진 비트 **25.9개** (1 ~ 95) → 밀도 1.3 % | Ridge, kNN, SVR, RF, XGBoost, MLP | 없음 (0/1) — 단, 노트북 `Pipeline`의 `StandardScaler`가 그대로 적용됨 → 희귀 비트가 큰 값으로 부풀려짐 ([추가], 4절 kNN·MLP 지는 케이스의 원인 후보) | 부분구조 유무 → 작용기 효과 | 비트 충돌, 3D·입체 없음, 작은 분자는 켜진 비트 적음 (17쪽); kNN은 유클리드 거리 (17쪽 Tanimoto 아님) |
| (c) 분자 그래프 `feature.py` | **x ∈ {0,1}^(n_atoms × 147)**, edge_index ∈ ℤ^(2 × 2·n_bonds), edge_attr ∈ {0,1}^(2·n_bonds × 16) — 분자마다 크기 다름 | GIN (`train.py`) | 입력은 0/1이라 없음; **y만 train 평균·표준편차로 표준화** (`Scaler('standard')`, 평가 때 되돌림) | 사람이 정한 특징 대신 구조를 직접 학습 (15쪽 "모델이 배우는 특징") | 데이터 많이 필요(19쪽 "1만 개 이상"), 학습 비용; **PyG GIN은 결합 특징 16개를 쓰지 않음** (GINE이면 사용) |

## 원자 147 · 결합 16의 구성 (`feature.py`, 18쪽)

| 원자 특징 (147) | 개수 | 결합 특징 (16) | 개수 |
|---|---|---|---|
| 원소 one-hot (H B C N O F P S Cl Br I) | 11 | 결합 종류 (단일/방향족/이중/삼중) | 4 |
| 형식 전하 (−2…+2) | 5 | 공액 | 1 |
| 혼성 (S, SP, SP2, SP3) | 4 | 회전 가능 / 비고리 / 고리 밖 단일 | 3 |
| 붙은 H 수 (0–3) | 4 | 시작 원자 (H C N O) | 4 |
| 방향족 | 1 | 끝 원자 (H C N O) | 4 |
| 원자가 (1–6) | 6 | | |
| 연결 수 (1–4) | 4 | | |
| 고리 안 | 1 | | |
| 반응 중심 (물성 예측에서는 항상 0) | 1 | | |
| 작용기 SMARTS 소속 | **110** | | |

피페리딘 확인 (44쪽): `x: (6, 147)  edge_index: (2, 12)  edge_attr: (12, 16)` — 결합 6개 × 양방향 = 엣지 12개.

## 발표용 3행 (S2)

| 표현 | shape | 모델 | 한 줄 |
|---|---|---|---|
| 기술자 217 | 5000 × 217 (실수, 표준화) | 6종 | 사람이 정한 물리화학 값 — 해석 쉬움 |
| Morgan 2048 | 5000 × 2048 (0/1, 켜진 비트 ~26) | 6종 | 부분구조 유무 — 넓지만 희소 |
| 그래프 | 원자 × 147, 결합 × 16 (GIN은 결합 미사용) | GIN | 구조를 모델이 직접 학습 |

## 자가점검 답

- 정제 중 DB 단계: NaN·무효 SMILES 제거, 중복 중앙값 병합, donotuse 제거 / 노트북 단계: 확인만 (0, 0) + 표본 추출 + 분할.
- shape: X_desc (N × 217), X_fp (N × 2048), 그래프 x (n_atoms × 147), edge_index (2 × 2·n_bonds), edge_attr (2·n_bonds × 16).
- random vs scaffold: "test 분자의 골격을 train에서 봤는가" — random 79 % 봤음 / scaffold 0 %.
- GIN이 결합 특징 16개를 버린다 → 2주차 `--conv GINE` 옵션이 그것을 쓰는 변경.
