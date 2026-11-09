# scripts — 발표 보조 스크립트 [추가: 연구실 코드가 아님]

과제 1·2단계(결과표, GNN 그림)는 **교수님 코드**(`261105_Week1_Practice/`)로만 만든다. 여기 스크립트는 발표 1절(데이터 카드)과 5절(후분석·XAI)의 재료용.
모두 `mlstudy` 환경에서 실행 (43쪽 패키지만 사용: rdkit, pandas, matplotlib, scikit-learn, xgboost, shap).

| 스크립트 | 입력 | 출력 | 발표 절 |
|---|---|---|---|
| `mp_data_card.py` | 데이터 CSV (smiles, mp) | `data_card.md`, 분포·분자량 그림, `duplicates.csv` | 1 데이터, 2 정제 로그 |
| `error_analysis.py` | test 예측 CSV (smiles, y_true, y_pred) | `summary.md`, 틀린 분자 표·그림, 구간별·골격별·분자량별 MAE, 잔차 그림 | 4 진단, 5 후분석 |
| `shap_descriptors.py` | 데이터 CSV | XGBoost+기술자 SHAP 그림·표, `pred_test_xgb_<split>.csv`, `xgb_metrics.md` | 5 XAI |
| `mlutils.py` | (공용) | 열 자동 인식, canonical, Murcko, random/scaffold 분할, 기술자 행렬, MAE/RMSE | |

## 실행 예

```
conda activate mlstudy
cd "C:\Users\mthog\CEPL ML\mp_presentation"

python scripts\mp_data_card.py "..\261105_Week1_Practice\data\mp.csv" --out 01_data
python scripts\shap_descriptors.py "..\261105_Week1_Practice\data\mp.csv" --out 05_analysis_xai\xgb_scaffold --split scaffold
python scripts\error_analysis.py 05_analysis_xai\xgb_scaffold\pred_test_xgb_scaffold.csv --out 05_analysis_xai\xgb_scaffold --top 20
python scripts\error_analysis.py "..\261105_Week1_Practice\results\mp_GNN-GIN_mean_random_test_pred.csv" --out 05_analysis_xai\gin_random --top 20
```

- 열 이름이 다르면 `--smiles-col`, `--y-col` (예측 CSV는 `--true-col`, `--pred-col`)
- `data/mp.csv` 열 = `smiles, value, unit, source` → 자동 인식됨. `train.py`의 `_test_pred.csv` 열 = `smiles, y, pred, sd` → 자동 인식됨
- 빠른 시험: `shap_descriptors.py ... --max-rows 2000`. 전체 19,945개는 기술자 계산 ~1분 + XGBoost ~1분 (이 PC 기준 추정)
- 실행 예시 출력(전체 데이터, scaffold): `01_data/` (데이터 카드), `05_analysis_xai/example_xgb_descriptors_scaffold/`
- 그림 라벨은 영문 (Windows matplotlib 기본 글꼴에 한글이 없어 □로 깨지는 것을 피함)

## 한계

- 분할·표준화는 연구실 `datamanager.py`와 **다른 구현**이다 (scaffold는 DeepChem 방식: 큰 골격부터 train). 결과표의 수치와 직접 비교하지 말 것.
- `shap_descriptors.py`의 XGBoost 설정(n_estimators 600, depth 6, lr 0.05)은 노트북의 XGBoost와 다를 수 있다. 발표에서는 "보조 모델"로 명시.
