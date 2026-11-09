"""check_env.py — 설치 확인 (슬라이드 44쪽)

실행:  conda activate mlstudy  →  python check_env.py

1) 44쪽의 8개 패키지(numpy, pandas, matplotlib, rdkit, sklearn, xgboost, torch, torch_geometric) + shap(43쪽) import, 버전 출력
2) 그래프 smoke test: 피페리딘 C1CCNCC1 → RDKit → PyG Data → GINConv 1층 forward
   ※ 여기서 만드는 원자 특징은 확인용 최소 버전(원소 one-hot 등 10개)이다.
     연구실 feature.py(원자 147, 결합 16)와 다르므로, 실습 폴더 안에서 `python feature.py`로 한 번 더 확인할 것
     (44쪽 기대 출력: piperidine x: (6, 147)  edge_index: (2, 12)  edge_attr: (12, 16)).

종료 코드: 모두 통과 0, 하나라도 실패 1
"""
from __future__ import annotations

import importlib
import platform
import sys

REQUIRED = [
    # (import 이름, 표시 이름, 44쪽 역할)
    ("numpy", "numpy", "배열 X, 오차 계산"),
    ("pandas", "pandas", "data/*.csv, 결과표"),
    ("matplotlib", "matplotlib", "parity plot, 학습곡선"),
    ("rdkit", "RDKit", "SMILES, 기술자, Morgan 지문"),
    ("sklearn", "scikit-learn", "Ridge, kNN, SVR, RF, MLP, 분할"),
    ("xgboost", "XGBoost", "부스팅 나무"),
    ("torch", "PyTorch", "GNN 학습 (train.py)"),
    ("torch_geometric", "PyTorch Geometric", "분자 그래프, GIN (models.py)"),
    ("shap", "shap", "4주차 XAI"),
]


def check_imports() -> bool:
    print(f"python {sys.version.split()[0]} ({platform.system()} {platform.machine()})  prefix={sys.prefix}")
    env_name = sys.prefix.replace("\\", "/").rstrip("/").split("/")[-1]
    if env_name != "mlstudy":
        print(f"  [warn] 현재 환경 이름이 'mlstudy'가 아닙니다: {env_name}  → conda activate mlstudy (46쪽 4행)")
    print(f"{'package':<20}{'version':<14}status   role")
    ok_all = True
    for mod, name, role in REQUIRED:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{name:<20}{ver:<14}ok       {role}")
        except Exception as e:  # noqa: BLE001
            ok_all = False
            print(f"{name:<20}{'-':<14}FAIL     {type(e).__name__}: {e}")
    return ok_all


def smoke_test_graph() -> bool:
    """피페리딘 → 그래프 → GINConv forward. 44쪽 'python feature.py' 와 같은 종류의 확인."""
    try:
        import torch
        from rdkit import Chem
        from torch_geometric.data import Data
        from torch_geometric.nn import GINConv, global_mean_pool
    except Exception as e:  # noqa: BLE001
        print(f"[smoke] 건너뜀 (import 실패): {e}")
        return False

    mol = Chem.MolFromSmiles("C1CCNCC1")  # piperidine, 18쪽 예
    elements = ["C", "N", "O", "S", "F", "Cl", "Br", "P"]
    feats = []
    for a in mol.GetAtoms():
        onehot = [1.0 if a.GetSymbol() == e else 0.0 for e in elements]
        feats.append(onehot + [float(a.GetDegree()), float(a.GetTotalNumHs())])
    x = torch.tensor(feats, dtype=torch.float)                      # (원자 수, 10)
    src, dst = [], []
    for b in mol.GetBonds():                                        # 결합 1개 = 양방향 2열 (18쪽)
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]
    edge_index = torch.tensor([src, dst], dtype=torch.long)         # (2, 2 x 결합 수)
    data = Data(x=x, edge_index=edge_index)

    torch.manual_seed(0)
    conv = GINConv(torch.nn.Sequential(torch.nn.Linear(x.size(1), 16), torch.nn.ReLU(), torch.nn.Linear(16, 16)))
    h = conv(data.x, data.edge_index)                               # 21쪽 식: MLP((1+eps) h_v + sum h_u)
    h_G = global_mean_pool(h, torch.zeros(x.size(0), dtype=torch.long))  # readout = mean pool (21쪽 연구실 설정)
    print(f"[smoke] piperidine x: {tuple(x.shape)}  edge_index: {tuple(edge_index.shape)}  "
          f"GINConv out: {tuple(h.shape)}  readout h_G: {tuple(h_G.shape)}  torch threads={torch.get_num_threads()}")
    assert tuple(x.shape) == (6, 10) and tuple(edge_index.shape) == (2, 12)
    return True


if __name__ == "__main__":
    imports_ok = check_imports()
    graph_ok = smoke_test_graph()
    if imports_ok and graph_ok:
        print("ok")  # 44쪽 기대 출력
        sys.exit(0)
    print("FAIL - 위 표에서 FAIL 행의 메시지 첫 줄을 00_설치안내.md ⑥ (46쪽 표)에서 찾아보세요")
    sys.exit(1)
