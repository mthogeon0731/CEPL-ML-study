"""datamanager.py : CSV -> PyG 그래프 데이터셋 -> train/validate/test 분할 -> 스케일링

연구실 코드의 흐름(load_data -> gen_dataset -> transform -> split_dataset -> Scaler)을 한 파일로 줄였다.
원본: 9.ml_gnn/code/data_pipeline/datamanager.py, model/modeling.py, model/architecture.py(Scaler)

    from datamanager import load_data, gen_dataset, split_dataset, Scaler
    df = load_data('logP', max_n=5000)
    dataset = gen_dataset(df)                       # list of torch_geometric.data.Data
    sets = split_dataset(dataset, how='scaffold')   # {'train', 'validate', 'test'}
"""
import os
import random

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data

from feature import mol_to_features

DATADIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
LOG_PROPS = {'viscosity'}   # 연구실 원본도 점도는 ln(cP)로 학습했다 (select_transformation)


# ---------------------------------------------------------------- 1. 불러오기
def load_data(prop, max_n=None, seed=42):
    """data/<prop>.csv 를 읽는다. 열: smiles, value, unit, source (+ viscosity 는 T_K)."""
    df = pd.read_csv(os.path.join(DATADIR, f'{prop}.csv'))
    if max_n and len(df) > max_n:
        df = df.sample(max_n, random_state=seed).reset_index(drop=True)
    print(f'[load_data] {prop}: {len(df):,} rows, unit = {df["unit"].iloc[0]}')
    return df


# ---------------------------------------------------------------- 2. 그래프로 바꾸기
def gen_dataset(df, prop=None):
    """DataFrame -> PyG Data 리스트. 읽히지 않는 SMILES 는 건너뛴다."""
    dataset = []
    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(row['smiles'])
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        x, edge_index, edge_attr = mol_to_features(mol)
        y = float(row['value'])
        if prop in LOG_PROPS:
            y = np.log(y)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                    y=torch.tensor([[y]], dtype=torch.float), smiles=row['smiles'])
        if 'T_K' in row:
            data.T = torch.tensor([[row['T_K']]], dtype=torch.float)
        dataset.append(data)
    print(f'[gen_dataset] {len(dataset):,} graphs, node features = {dataset[0].x.shape[1]}, '
          f'edge features = {dataset[0].edge_attr.shape[1]}')
    return dataset


# ---------------------------------------------------------------- 3. 분할
def split_dataset(dataset, portion=0.1, how='random', seed=42):
    """연구실 기본값(modeling.split_dataset): test = 전체의 10%, validate = 전체의 10%, 나머지 train -> 80:10:10.

    how='random'   : 무작위 (연구실 원본 방식)
    how='scaffold' : 시험 세트에는 학습에서 못 본 Murcko 뼈대만 들어간다
    """
    rng = random.Random(seed)
    N = len(dataset); n = int(np.ceil(N * portion))
    if how == 'random':
        idx = list(range(N)); rng.shuffle(idx)
        test, validate, train = idx[:n], idx[n:2 * n], idx[2 * n:]
    elif how == 'scaffold':
        groups = {}
        for i, d in enumerate(dataset):
            s = MurckoScaffold.MurckoScaffoldSmiles(smiles=d.smiles, includeChirality=False)
            groups.setdefault(s, []).append(i)
        sets = list(groups.values()); rng.shuffle(sets)
        sets.sort(key=len)                  # 작은(희귀한) 뼈대부터 test, validate 로
        test, validate, train = [], [], []
        for g in sets:
            if len(test) + len(g) <= n:
                test += g
            elif len(validate) + len(g) <= n:
                validate += g
            else:
                train += g
    else:
        raise ValueError(how)
    out = {'train': [dataset[i] for i in train],
           'validate': [dataset[i] for i in validate],
           'test': [dataset[i] for i in test]}
    print(f'[split_dataset:{how}] ' + ' / '.join(f'{k} {len(v):,}' for k, v in out.items()))
    return out


# ---------------------------------------------------------------- 4. 스케일링
class Scaler:
    """y 를 학습 세트 통계로 표준화(standard) 또는 0-1 정규화(minmax). 신경망은 y 크기에 민감하다."""

    def __init__(self, mode='standard', eps=1e-7):
        self.mode, self.eps = mode, eps

    def fit(self, dataset):
        Y = torch.cat([d.y for d in dataset])
        self.mean, self.std = Y.mean(), Y.std()
        self.min, self.max = Y.min(), Y.max()
        return self

    def transform(self, y):
        if self.mode == 'standard':
            return (y - self.mean) / (self.std + self.eps)
        return (y - self.min) / (self.max - self.min + self.eps)

    def inverse_transform(self, y):
        if self.mode == 'standard':
            return y * (self.std + self.eps) + self.mean
        return y * (self.max - self.min + self.eps) + self.min
