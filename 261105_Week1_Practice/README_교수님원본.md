# 1주차 실습. 물성 예측 모델 여러 개 비교하기

## 1. 설치 (스터디 전에 한 번)

**자세한 순서와 문제 해결은 `00_설치안내.md`.** 설치 후 `00_python_basics.ipynb` 로 라이브러리를 한 번 둘러보고 온다 (10분).

요약: Miniforge 설치 후 Miniforge Prompt 에서 (Anaconda 는 200인 이상 기관에서 유료라 쓰지 않는다):

```bash
conda create -n mlstudy -c conda-forge python=3.11 rdkit scikit-learn xgboost pandas matplotlib jupyter ipykernel -y
conda activate mlstudy
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric shap
```

확인 두 가지:

```bash
python -c "import rdkit, sklearn, xgboost; print('ok')"      # ok 가 나오면 노트북 준비 끝
python feature.py                                             # 아래처럼 나오면 GNN 준비 끝
```

```
piperidine  x: (6, 147)  edge_index: (2, 12)  edge_attr: (12, 16)
```

## 2. 파일

| 파일 | 하는 일 |
|---|---|
| `00_설치안내.md` | Miniforge · conda 환경 · Jupyter · VS Code · Colab, 문제 해결 표 |
| `00_python_basics.ipynb` | **예습.** numpy · pandas · matplotlib · RDKit · scikit-learn · XGBoost · PyTorch · PyG 를 logP 데이터로 한 셀씩 |
| `01_multi_model_comparison.ipynb` | **오늘 실습.** 표현 2종 × 모델 6종 × 분할 2종 결과표 |
| `feature.py` | 분자 → 그래프 (원자 특징 147, 결합 특징 16) |
| `datamanager.py` | CSV 읽기 → 그래프 목록 → 80:10:10 분할 → y 표준화 |
| `models.py` | GIN 모델 (연구실 물성 예측 모델과 같은 구조) |
| `train.py` | GNN 학습 한 번 → 결과표 한 줄 + 그림 |
| `data/` | 물성 9종. 출처·단위는 `data/README_data.md` |

## 3. 팀 구성

2–3명이 한 팀, **팀당 물성 하나**. 아래 순서로 고른다 (위쪽이 데이터가 많고 깨끗하다).

| 순서 | 물성 (`PROP`) | 데이터 수 | 단위 | 난이도 |
|---|---|---|---|---|
| 1 | `logP` | 14,031 | log10 (옥탄올/물) | 쉬움 |
| 2 | `density` | 8,746 | g/mL | 쉬움 |
| 3 | `logSolubility` | 11,172 | log10(mol/L) | 중간 |
| 4 | `bp` | 2,159 | K | 쉬움 (데이터 적음) |
| 5 | `logPvap` | 3,092 | log10(mmHg) | 중간 |
| 6 | `mp` | 19,945 | K | 어려움 (잡음 큼) |
| 7 | `HL` | 1,536 | log10(atm·m³/mol) | 중간 |
| 8 | `viscosity` | 3,582 점 (957 분자) | cP | 어려움 (온도 의존) |
| 9 | `pKa` | 6,857 | 무차원 | 어려움 |

`bp`, `mp` 는 **켈빈**이다.

## 4. 1주차 과제 (다음 시간 팀별 5분 발표)

1. **결과표**: 노트북 첫 셀의 `PROP` 를 팀 물성으로 바꾸고 끝까지 실행. 모델 6종 × 표현 2종 × 분할 2종 MAE 표, 기준선 MAE 와 비교
2. **GNN 그대로 돌려 보기**: 코드는 고치지 말고 한 번 실행

   ```bash
   python train.py --prop 팀물성          # 예: python train.py --prop logP   (CPU 약 3분)
   ```

   `results/` 에 생긴 **parity 그림**과 **학습곡선 그림**을 가져온다. 1번 결과표의 최고 모델과 MAE 를 비교한다
3. **코드 읽기**: `feature.py`, `datamanager.py`, `models.py`, `train.py` 가 각각 무슨 일을 하는지 한 줄씩
4. 발표: 결과표 한 장 + GNN 그림 두 장 + 코드 한 줄 요약

**예고 (2주차 과제)**: GIN 구조를 배운 뒤, 이번에 돌린 GNN 에서 하나를 직접 바꿔(readout, 층 수, 결합 특징 등) 모델링하고 1주차 표와 비교한 그림을 가져온다.

노트북 마지막 절(팀 과제)의 "하나만 바꿔보기", "틀린 분자 보기"는 **선택**이다. 시간이 남는 팀만 한다.

규칙: test 성능을 보고 설정을 고르지 않는다.
