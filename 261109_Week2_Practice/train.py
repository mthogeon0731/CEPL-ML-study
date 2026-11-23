"""train.py : GNN 학습 한 번 = 결과표 한 줄

    python train.py --prop logP                          # GIN, random split, 연구실 기본 설정
    python train.py --prop logP --split scaffold
    python train.py --prop logP --conv GINE              # 결합 특징을 쓰는 GIN
    python train.py --prop logP --pooling sum            # GIN 논문의 sum readout
    python train.py --prop logP --bayes                  # 불확실성 (BGNN-GIN)

결과: results/summary.csv 에 한 줄 추가, results/<이름>_parity.png, results/<이름>_curve.png
학습 설정은 연구실 modeling.train_model 이 실제로 쓴 값을 따른다 (AdamW 1e-3, ReduceLROnPlateau
patience 10 / factor 0.8, 기울기 ±10 자르기, dropout GNN 0.3 · BGNN 0, batch 64/256/1024 by N). 조기 종료는 validate MAE 가 patience_stop 에폭 동안 안 좋아지면 멈추고
가장 좋았던 가중치로 돌아간다.
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch_geometric.loader import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datamanager import load_data, gen_dataset, split_dataset, Scaler
from models import GraphRegressor

HERE = os.path.dirname(os.path.abspath(__file__))


def predict(model, loader, scaler, bayes):
    model.eval()
    ys, mus, sds = [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch)
            if bayes:
                mu, sd = out.mean(1, keepdim=True), out.std(1, keepdim=True)
            else:
                mu, sd = out, torch.zeros_like(out)
            ys.append(batch.y_raw); mus.append(mu); sds.append(sd)   # 정답은 원래 단위
    y = torch.cat(ys); mu = scaler.inverse_transform(torch.cat(mus))
    sd = torch.cat(sds) * (scaler.std + scaler.eps)
    return y.numpy().ravel(), mu.numpy().ravel(), sd.numpy().ravel()


def lab_batch_size(n):
    """연구실 modeling.set_batch_size: 1만 미만 64, 10만 미만 256, 그 이상 1024."""
    return 64 if n < 10000 else 256 if n < 100000 else 1024


def run(prop='logP', conv='GIN', split='random', pooling='mean', bayes=False, max_n=5000,
        epochs=300, batch_size=None, lr=1e-3, num_layers=5, hidden=128, fingerprint=64,
        dropout=None, patience_stop=30, seed=42, outdir=None, verbose=True):
    torch.manual_seed(seed); np.random.seed(seed)
    if dropout is None:          # 연구실 modeling.train_model 실제 학습값: GNN 0.3, BGNN 0
        dropout = 0.0 if bayes else 0.3
    outdir = outdir or os.path.join(HERE, 'results'); os.makedirs(outdir, exist_ok=True)
    name = f'{prop}_{"BGNN" if bayes else "GNN"}-{conv}_{pooling}_{split}'

    # 1) 데이터
    df = load_data(prop, max_n=max_n, seed=seed)
    sets = split_dataset(gen_dataset(df, prop), how=split, seed=seed)
    batch_size = batch_size or lab_batch_size(len(df))
    # 스케일러는 train 으로만 fit 한다. 연구실 원본은 train+validate+test 로 fit 한다 (y 평균·표준편차만이라 영향은 작지만 원칙상 누수)
    scaler = Scaler('standard').fit(sets['train'])
    for k in sets:                                   # y 는 스케일된 값으로 학습, 평가 때 되돌린다
        for d in sets[k]:
            d.y_raw = d.y.clone(); d.y = scaler.transform(d.y_raw)
    loaders = {k: DataLoader(v, batch_size=batch_size, shuffle=(k == 'train')) for k, v in sets.items()}

    # 2) 모델
    model = GraphRegressor(sets['train'][0].x.shape[1], sets['train'][0].edge_attr.shape[1], conv=conv,
                           num_layers=num_layers, hidden_channels=hidden, fingerprint_channels=fingerprint,
                           dropout=dropout, pooling=pooling, bayes=bayes)
    for p in model.parameters():
        p.register_hook(lambda g: g.clamp(-10, 10))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.8, min_lr=1e-6)
    n_train = len(sets['train'])
    if verbose:
        print(f'[model] {name}: {model.n_params():,} parameters')

    # 3) 학습
    log = {'train_loss': [], 'val_mae': []}
    best, best_state, bad, t0 = np.inf, None, 0, time.time()
    for ep in range(epochs):
        model.train(); tot = 0.
        for batch in loaders['train']:
            opt.zero_grad()
            out = model(batch)
            if bayes:   # 간소화한 ELBO: 평균 예측의 MSE + KL / N  (연구실 원본은 표본 표준편차로 우도를 계산)
                loss = F.mse_loss(out.mean(1, keepdim=True), batch.y) + model.kl() / n_train
            else:
                loss = F.mse_loss(out, batch.y)
            loss.backward(); opt.step(); tot += loss.item() * batch.num_graphs
        sets_val = predict(model, loaders['validate'], scaler, bayes)
        val_mae = np.mean(np.abs(sets_val[0] - sets_val[1]))
        sched.step(val_mae)
        log['train_loss'].append(tot / n_train); log['val_mae'].append(val_mae)
        if val_mae < best - 1e-4:
            best, best_state, bad = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or bad == 0):
            print(f'  epoch {ep:3d}  train loss {log["train_loss"][-1]:.4f}  val MAE {val_mae:.3f}'
                  f'  lr {opt.param_groups[0]["lr"]:.1e}' + ('  *' if bad == 0 else ''))
        if bad >= patience_stop:
            break
    model.load_state_dict(best_state)
    sec = time.time() - t0

    # 4) 평가
    y, mu, sd = predict(model, loaders['test'], scaler, bayes)
    res = dict(prop=prop, model=('BGNN-' if bayes else 'GNN-') + conv, pooling=pooling, split=split,
               n=len(df), epochs=ep + 1, sec=round(sec, 1),
               MAE=float(np.mean(np.abs(y - mu))), RMSE=float(np.sqrt(np.mean((y - mu) ** 2))),
               R2=float(1 - np.sum((y - mu) ** 2) / np.sum((y - y.mean()) ** 2)),
               baseline_MAE=float(np.mean(np.abs(y - torch.cat([d.y_raw for d in sets['train']]).mean().item()))))
    # Bayes 전용 지표. summary.csv 의 열을 항상 같게 하려고 GNN 에서는 빈 값으로 둔다
    res['mean_sd'] = float(sd.mean()) if bayes else np.nan
    res['coverage_2sd'] = float(np.mean(np.abs(y - mu) <= 2 * sd)) if bayes else np.nan   # 이상적이면 0.95 근처
    print(f'[test] {name}: MAE {res["MAE"]:.3f} | RMSE {res["RMSE"]:.3f} | R2 {res["R2"]:.3f} '
          f'| baseline MAE {res["baseline_MAE"]:.3f} | {res["epochs"]} epochs, {sec:.0f} s')

    summary = os.path.join(outdir, 'summary.csv')
    pd.DataFrame([res]).to_csv(summary, mode='a', header=not os.path.exists(summary), index=False)
    pd.DataFrame({'smiles': [d.smiles for d in sets['test']], 'y': y, 'pred': mu, 'sd': sd}).to_csv(
        os.path.join(outdir, f'{name}_test_pred.csv'), index=False)
    plot(y, mu, sd, log, res, os.path.join(outdir, name))
    return res


def plot(y, mu, sd, log, res, stem):
    lo, hi = min(y.min(), mu.min()), max(y.max(), mu.max()); pad = 0.05 * (hi - lo); lim = (lo - pad, hi + pad)
    fig, ax = plt.subplots(figsize=(4.2, 4))
    if res['model'].startswith('BGNN'):
        ax.errorbar(y, mu, yerr=2 * sd, fmt='none', ecolor='#999999', alpha=0.3, lw=0.6)
    ax.scatter(y, mu, s=8, alpha=0.5, color='#0072B2', label=f'{res["model"]}\nMAE = {res["MAE"]:.3f}')
    ax.plot(lim, lim, 'k', lw=1); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f'measured {res["prop"]}'); ax.set_ylabel(f'predicted {res["prop"]}')
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout(); fig.savefig(stem + '_parity.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(log['train_loss'], color='#0072B2', label='train loss (scaled MSE)')
    axes[1].plot(log['val_mae'], color='#E69F00', label='validate MAE')
    for ax, tag in zip(axes, 'ab'):
        ax.set_xlabel('epoch'); ax.legend(frameon=False)
        ax.text(-0.15, 1.04, tag, transform=ax.transAxes, fontsize=14, fontweight='bold')
    fig.tight_layout(); fig.savefig(stem + '_curve.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prop', default='logP')
    ap.add_argument('--conv', default='GIN', choices=['GIN', 'GINE', 'GCN', 'GAT'])
    ap.add_argument('--split', default='random', choices=['random', 'scaffold'])
    ap.add_argument('--pooling', default='mean', choices=['mean', 'sum'])
    ap.add_argument('--bayes', action='store_true')
    ap.add_argument('--max_n', type=int, default=5000, help='0 = 전체')
    ap.add_argument('--epochs', type=int, default=300)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    run(a.prop, a.conv, a.split, a.pooling, a.bayes, a.max_n or None, a.epochs, seed=a.seed)
