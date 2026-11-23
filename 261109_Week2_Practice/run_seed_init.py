"""[추가] 같은 데이터·같은 분할(seed 42)에서 가중치 초기화·셔플 seed 만 바꿔 train.run 을 다시 돌린다.
train.run(seed=s) 는 표본 추출·분할까지 seed 로 바꾸므로, load_data / split_dataset 의 seed 를 42 로 고정해 감싼다.
    python run_seed_init.py --seed 7 --pooling sum --split scaffold
"""
import argparse, os
import datamanager, train

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--pooling', default='mean')
ap.add_argument('--split', default='random')
ap.add_argument('--data_seed', type=int, default=42)
a = ap.parse_args()
train.load_data = lambda prop, max_n=None, seed=42: datamanager.load_data(prop, max_n=max_n, seed=a.data_seed)
train.split_dataset = lambda ds, how='random', seed=42: datamanager.split_dataset(ds, how=how, seed=a.data_seed)
out = os.path.join(train.HERE, 'results_seeds', f'seed{a.seed}')
train.run('mp', pooling=a.pooling, split=a.split, seed=a.seed, outdir=out)
