# results — 노트북·train.py 산출물 모아 두는 곳

발표에 들어가는 모든 수치·그림의 **원본**은 여기. 각 절 README의 표는 이 폴더의 파일에서 옮겨 적는다.

## 파일 이름 규칙

| 파일 | 출처 | 쓰는 절 |
|---|---|---|
| `mae_table_mp.csv` / `.png` | `01_multi_model_comparison.ipynb` 결과표 (PROP = 'mp') — 노트북 마지막 셀에서 `df.to_csv(...)` | S4, `04_performance/results_table.md` |
| `fig_parity_gin_test.png` | `261105_Week1_Practice/results/` 의 parity 그림 복사 (원본 이름을 노트에 기록) | S5 |
| `fig_learning_curve_gin.png` | 같은 폴더의 학습곡선 복사 | S5 |
| `train_log_gin_mp.txt` | `python train.py --prop mp` 터미널 출력 전체 (결과표 한 줄 포함) — `python train.py --prop mp > ..\mp_presentation\results\train_log_gin_mp.txt 2>&1` | S4 GIN 행, S7 |
| `pred_test_<model>_<split>.csv` | 후분석용 test 예측 (smiles, y_true, y_pred) | `05_analysis_xai/` |
| `run_notes.md` | run마다: 날짜, seed, 코드 판(파일 수정 여부), 걸린 시간 | 재현 |

## 기록 규칙

- 같은 실험을 다시 돌리면 덮어쓰지 말고 `_v2` 붙이기 (발표 수치가 어느 run인지 추적)
- 과제 2단계(코드 수정 없음)의 결과와, 예측 저장용으로 코드를 고친 뒤의 결과를 구분해 적기
- 분할 종류·seed·단위(K)를 파일 안 또는 `run_notes.md`에 반드시
