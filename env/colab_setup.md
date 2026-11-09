# Google Colab 대안 (슬라이드 45쪽)

설치가 끝까지 안 될 때 브라우저만으로. colab.research.google.com → 새 노트북 → 아래 셀을 순서대로.

**셀 1 — 패키지** (Colab에는 numpy·pandas·matplotlib·scikit-learn·PyTorch가 이미 있음)

```python
!pip install rdkit xgboost torch_geometric shap
```

**셀 2 — 실습 zip 올린 뒤** (왼쪽 파일 창에 `261105_Week1_Practice.zip`을 끌어다 놓기)

```python
!unzip -o 261105_Week1_Practice.zip
%cd 261105_Week1_Practice
```

**셀 3 — 확인** (44쪽)

```python
import numpy, pandas, matplotlib, rdkit, sklearn, xgboost, torch, torch_geometric; print('ok')
!python feature.py
```

**셀 4 — 과제 2단계 GNN** (49쪽, 팀 물성 = mp)

```python
!python train.py --prop mp
```

## 주의 (45쪽)

- 세션이 끊기면 올린 파일과 결과가 지워짐 → `results/`의 parity·학습곡선 그림은 **바로 내려받기**
  (파일 창에서 우클릭 ▸ 다운로드, 또는 `from google.colab import files; files.download('results/파일명.png')`)
- Colab 기본 런타임은 CPU. 43쪽과 같이 GPU 없이도 충분 (mp 19,945개 GNN 학습 = 수 분)
- 노트북(`01_multi_model_comparison.ipynb`)은 파일 창에서 더블클릭해 열면 Colab에서 그대로 실행됨. 첫 셀 `PROP = 'mp'`
