"""[추가] 같은 분할에서 비교: train.py(GIN)의 80/10/10 random 분할을 그대로 재현해,
노트북의 고전 모델(기술자/지문 × 6종)을 같은 train 4,000개로 학습하고 같은 test 500개로 평가한다.
→ 49쪽 4단계 "1의 최고 모델 vs 2의 GIN" 비교를 같은 test 집합에서 할 수 있게 한다.

출력: results/fair_compare_gin_split.csv, fair_compare_gin_split.md
"""
import random, sys, time, json
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
RDLogger.DisableLog('rdApp.*')

PRACTICE = sys.argv[1]; OUT = sys.argv[2]; SPLIT = sys.argv[3] if len(sys.argv) > 3 else 'random'
sys.path.insert(0, PRACTICE)
from datamanager import load_data          # 교수님 코드의 표본 추출 그대로 (seed 42, max_n 5000)

df = load_data('mp', max_n=5000, seed=42)
# split_dataset(how='random', seed=42) 재현: gen_dataset 이 건너뛴 SMILES 가 없으므로 인덱스가 1:1
rng = random.Random(42); N = len(df); n = int(np.ceil(N * 0.1))
if SPLIT == 'random':
    idx = list(range(N)); rng.shuffle(idx)
    test, validate, train = idx[:n], idx[n:2 * n], idx[2 * n:]
else:   # datamanager.split_dataset(how='scaffold') 재현
    from rdkit.Chem.Scaffolds import MurckoScaffold
    groups = {}
    for i, smi in enumerate(df['smiles']):
        groups.setdefault(MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False), []).append(i)
    sets = list(groups.values()); rng.shuffle(sets); sets.sort(key=len)
    test, validate, train = [], [], []
    for g in sets:
        if len(test) + len(g) <= n: test += g
        elif len(validate) + len(g) <= n: validate += g
        else: train += g
gin = pd.read_csv(f'{PRACTICE}/results/mp_GNN-GIN_mean_{SPLIT}_test_pred.csv')
assert list(df['smiles'].values[test]) == list(gin['smiles']), '재현한 test 집합이 train.py 와 다름'
assert np.allclose(df['value'].values[test], gin['y'].values, atol=1e-3)
print(f'split reproduced: train {len(train)} / validate {len(validate)} / test {len(test)}')
train_mean = float(df['value'].values[train].mean())
print('train mean [K]', train_mean)

mols = [Chem.MolFromSmiles(s) for s in df['smiles']]
t0 = time.time()
X_desc = np.array([[f(m) for _, f in Descriptors.descList] for m in mols], dtype=float)
X_desc = np.clip(np.nan_to_num(X_desc, nan=0.0, posinf=0.0, neginf=0.0), -1e6, 1e6)
mfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X_fp = np.array([mfp.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)
y = df['value'].values
print(f'features {X_desc.shape} {X_fp.shape} ({time.time() - t0:.0f} s)')

# 노트북 5절 get_models() 와 같은 설정
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
SEED = 42
def get_models():
    return {
        'Ridge':        make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'kNN':          make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights='distance')),
        'SVR':          make_pipeline(StandardScaler(), SVR(C=10, epsilon=0.1)),
        'RandomForest': RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEED),
        'XGBoost':      XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6,
                                     subsample=0.8, colsample_bytree=0.5, n_jobs=-1, random_state=SEED),
        'MLP':          make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-3,
                                     early_stopping=True, max_iter=300, random_state=SEED)),
    }

rows = []
yt = y[test]
rows.append(dict(feature='-', model='baseline(train mean)', MAE=mean_absolute_error(yt, np.full(len(test), train_mean)),
                 RMSE=np.nan, R2=np.nan, sec=0.0))
for feat_name, X in {'Descriptor': X_desc, 'MorganFP': X_fp}.items():
    for name, model in get_models().items():
        t0 = time.time(); model.fit(X[train], y[train]); p = model.predict(X[test])
        rows.append(dict(feature=feat_name, model=name, MAE=mean_absolute_error(yt, p),
                         RMSE=mean_squared_error(yt, p) ** 0.5, R2=r2_score(yt, p), sec=time.time() - t0))
        print(f'{feat_name:10s} {name:12s} MAE={rows[-1]["MAE"]:.3f}')
        if feat_name == 'Descriptor' and name == 'XGBoost':
            pd.DataFrame({'smiles': df['smiles'].values[test], 'y_true': yt, 'y_pred': p, 'train_mean': train_mean}
                         ).to_csv(f'{OUT}/pred_test_XGBoost_Descriptor_ginsplit_{SPLIT}.csv', index=False)
g = gin
rows.append(dict(feature='Graph (147/16)', model='GIN (train.py)', MAE=mean_absolute_error(g['y'], g['pred']),
                 RMSE=mean_squared_error(g['y'], g['pred']) ** 0.5, R2=r2_score(g['y'], g['pred']), sec=np.nan))
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/fair_compare_gin_split{"" if SPLIT == "random" else "_" + SPLIT}.csv', index=False)
json.dump({'train_mean_K': train_mean, 'n_train': len(train), 'n_validate': len(validate), 'n_test': len(test)},
          open(f'{OUT}/fair_compare_gin_split{"" if SPLIT == "random" else "_" + SPLIT}_info.json', 'w'), indent=2)
print(res.round(3).to_string(index=False))
