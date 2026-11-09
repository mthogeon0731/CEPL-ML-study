# 02 전처리 — 어떻게 정제했고, 어떤 표현자(feature)를 썼나

11쪽 단계: **정제 → 표현 → 분할** (`gen_dataset`, `mol_to_features`, `split_dataset`) · 발표 0:30–1:20 · 슬라이드 S2

## ① 발표에서 답할 질문

1. **정제** — 누가, 어디서 했나: DB 단계(`data/README_data.md`)에서 canonical SMILES·중복 병합(중앙값)·`donotuse` 제거가 끝났고, 노트북 2절은 **확인만** 한다 (무효 0, 중복 0). 염·입체는 처리하지 않았다 (염 1개, 입체 표기 1,330개 — `01_data/data_card.md`).
2. **단위·변환** — 이미 K. log 변환 없음 (`datamanager.LOG_PROPS = {'viscosity'}` 뿐, 13쪽 표와 일치). GNN만 y를 train 통계로 **표준화**(`Scaler('standard')`) 후 평가 때 되돌림.
3. **표현(feature)** — (a) RDKit 기술자 `Descriptors.descList` 전부(NaN·inf → 0, ±1e6 clip) → StandardScaler, (b) Morgan 지문 `GetMorganGenerator(radius=2, fpSize=2048)` (ECFP4), (c) 분자 그래프 `feature.py` 원자 147 · 결합 16 — 단, **PyG `GIN`은 결합 특징을 쓰지 않는다** (`models.py` 주석; GINE이면 사용).
4. **분할** — 노트북: random 80/20 (`train_test_split`) vs scaffold 80/20 (희귀 골격부터 test). `train.py`: 80/10/10 (random / scaffold: 작은 골격부터 test → validate). 둘 다 seed 42, 표본 5,000개 (`MAX_N`/`--max_n`).
5. **누수 점검** — 같은 canonical SMILES 중복 0 → train∩test 교집합 0. 스케일러: 노트북은 `Pipeline` 안의 StandardScaler(train fit), `train.py`는 Scaler를 train으로만 fit (연구실 원본은 train+val+test — 14쪽 누수 예를 코드가 고쳤다).

## ② 슬라이드 근거

| 쪽 | 내용 | 발표에서 |
|---|---|---|
| 12 | canonical SMILES, 무효 → 제외 후 개수 기록, 중복(값 차이 작음 → 중앙값), 염 → 큰 조각·중성화, 입체 유지 여부 | 정제 로그 표 |
| 13 | 단위 표: bp, mp → K; 같은 분자 여러 값 → 표준편차 큰 행부터; source 열 | "DB에서 끝남" 한 줄 |
| 14 | random vs scaffold, Murcko 골격, 누수 4경로, 연구실 `scale_dataset`이 train+val+test로 fit | 분할 + 누수 점검 |
| 15 | 표현 3단계: 기술자(실수 ~217) / 지문(0·1 2048) / 그래프; 맞는 모델; 사람이 정한 특징 → 모델이 배우는 특징 | 표현 표 |
| 16 | 기술자: 값 크기 차이 → StandardScaler; 상관 큼 → 계수 해석 주의 | 스케일링 한 줄 |
| 17 | Morgan r=2, 2048비트, 충돌; 작은 분자는 켜진 비트 적음 (노트북이 "분자당 켜진 비트 평균"을 출력) | 지문 설정 |
| 18 | 피페리딘 x (6, 147), edge_index (2, 12), edge_attr (12, 16); 판마다 특징 수 다름 (노트북 표의 "74개"는 2024 판) | 차원 명시 |

## ③ 산출물 — `preprocessing_log.md`, `representation_table.md` (아래를 복사해 노트북 출력으로 빈칸만 채움)

### `preprocessing_log.md`

| 단계 | 처리 | 남은 행 | 버린/합친 행 | 어디서 | 근거 쪽 |
|---|---|---|---|---|---|
| 원본 | Bradley Open MP | 27,976 | – | DB | README_data |
| 무효 SMILES | RDKit 파싱 실패 제거 | 27,976 | 0 | DB | 12 |
| 중복 | 같은 canonical SMILES → **중앙값** | 19,945 | 8,031 | DB | 12 |
| 'donotuse' | 표시 행 제거 | 19,945 | 0 (pkl에서 이미) | DB | 12 |
| 염·혼합물 | **안 함** (`.` 포함 1개) | 19,945 | 0 | – | 12 |
| 입체 | 표기 유지 (1,330개), 그래프·지문은 입체 미반영 | 19,945 | 0 | – | 12 |
| 단위 | K 그대로, log 변환 없음 | 19,945 | – | – | 13 |
| 표본 | `MAX_N = 5000`, seed 42 | 5,000 | – | 노트북 1절 / `train.py --max_n` | – |
| 노트북 확인 | 무효 ( ), 중복 ( ) | 5,000 | | 노트북 2절 출력 | 12 |
| 분할 (노트북) | random 80/20: train ( ) / test ( ); scaffold 80/20: train ( ) / test ( ) | | | 노트북 4절 출력 | 14 |
| 분할 (train.py) | 80/10/10: train ( ) / validate ( ) / test ( ) | | | `[split_dataset:random]` 출력 줄 | 14 |
| 누수 점검 | train∩test canonical SMILES = 0 (중복 0이므로); Scaler fit = train만 | | | | 14 |

### `representation_table.md`

| 표현 | 차원 (shape) | 넣은 모델 | 스케일링 | mp에서 기대 | 한계 (슬라이드) |
|---|---|---|---|---|---|
| (a) RDKit 기술자 `Descriptors.descList` | X_desc (5000 × ( )) ← 노트북 출력 `descriptor: (…)` | Ridge, kNN, SVR, RF, XGBoost, MLP (6종 모두) | Pipeline 안 StandardScaler (Ridge, kNN, SVR, MLP); RF·XGB는 불필요 | MolWt·TPSA·H-donor·방향족 고리 등 분자간 상호작용 대리변수 | 다른 분자가 같은 값 가능; 상관 큼 (16쪽) |
| (b) Morgan 지문 r=2 | X_fp (5000 × 2048), 켜진 비트 평균 ( )개 | 6종 모두 | 없음 (0/1) — Pipeline의 StandardScaler는 그대로 통과 | 부분구조 유무 → 작용기 효과 | 비트 충돌, 3D·입체 없음, 작은 분자 정보 부족 (17쪽) |
| (c) 분자 그래프 `feature.py` | x (n_atoms × 147), edge_index (2 × 2·n_bonds), edge_attr (2·n_bonds × 16) | GIN (`train.py`) | y 표준화 (train 통계) | 구조를 직접 학습 | 데이터 많이 필요, 학습 비용 (15쪽); **GIN은 결합 특징 미사용** |

## ④ 작업 순서

1. 노트북 2·3·4절의 출력 줄(무효/중복 개수, `descriptor: (…)`, `Morgan FP : (…)`, 분할 크기)을 위 표의 빈칸에 옮긴다.
2. `python train.py --prop mp` 출력의 `[load_data]`, `[gen_dataset]`, `[split_dataset:random]` 줄을 train.py 행에 옮긴다.
3. 발표 S2는 두 표를 **각 5행, 3행으로 줄여서** 넣는다 (정제는 "DB에서 끝났고 우리는 확인했다" 한 줄이면 충분).
4. [추가] "염·입체를 처리하지 않았다"는 약점이 아니라 **결정 사항**으로 말한다 — 12쪽 "물성별로 결정".

## ⑤ 자가점검

- [ ] 정제의 어느 단계가 DB에서, 어느 단계가 노트북에서 일어났는지 구분해 말할 수 있다
- [ ] 세 표현의 shape를 N·n_atoms로 쓸 수 있다 (16–18쪽)
- [ ] random과 scaffold 분할의 차이를 "test가 처음 보는 골격인가"로 설명할 수 있다 (14쪽)
- [ ] GIN이 결합 특징 16개를 버린다는 것과 2주차 GINE 옵션을 연결할 수 있다
