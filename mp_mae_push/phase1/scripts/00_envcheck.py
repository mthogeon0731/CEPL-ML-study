"""00_envcheck.py : 실행 환경·하드웨어 확인 (1단계 준비)."""
import importlib
import os
import platform
import sys

import psutil

print('python', sys.version.split()[0], '|', platform.platform())
print('cpu physical', psutil.cpu_count(logical=False), 'logical', psutil.cpu_count(logical=True))
vm = psutil.virtual_memory()
print(f'ram total {vm.total / 2**30:.1f} GiB | available {vm.available / 2**30:.1f} GiB')
for name in ['numpy', 'pandas', 'sklearn', 'xgboost', 'rdkit', 'torch', 'torch_geometric', 'shap',
             'lightgbm', 'catboost', 'optuna', 'thermo', 'mordred', 'chemprop', 'codecarbon', 'transformers']:
    try:
        m = importlib.import_module(name)
        print(f'  {name:16s} {getattr(m, "__version__", "?")}')
    except Exception as e:  # noqa: BLE001
        print(f'  {name:16s} MISSING ({type(e).__name__})')
try:
    import torch
    print('cuda available', torch.cuda.is_available(), '| torch threads', torch.get_num_threads())
except Exception as e:  # noqa: BLE001
    print('torch check failed', type(e).__name__)
from rdkit.Chem import Descriptors
print('rdkit descriptors', len(Descriptors.descList))
print('cwd', os.getcwd())
