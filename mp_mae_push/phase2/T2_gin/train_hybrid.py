"""train_hybrid.py : T2 GIN 슬롯 — 수업 GIN(mean) + RDKit 기술자 217 하이브리드 readout 학습 (train.py 에서 파생).

    (작업 폴더 루트에서) run_mlstudy.ps1 T2_gin/src/train_hybrid.py --stage pilot
                                                               --stage main  --runs random:0,random:1,...
                                                               --stage control --runs random:1 --no_desc

train.py 와 다른 점 (실행 전 고정, README 에 근거):
  1. 분할은 mp_mae_push/splits/<how>.csv 고정 분할 (mpcommon.load_split). test 행은 읽지도 예측하지도 않는다.
  2. 조기 종료·LR 스케줄은 validation 이 아니라 train 안 10 % 내부 분할(random: 무작위, scaffold: Murcko 뼈대 묶음)로.
  3. readout 뒤에 기술자 217 (train 전체로 fit 한 표준화, ±10 클리핑) 를 이어 붙여 MLP head 로.
  4. 손실: SmoothL1 (beta = 0.25, 표준화된 y 단위 = 0.25 sigma_y) — 지표가 MAE 이고 라벨 잡음이 꼬리가 길어서.
  5. y 스케일러는 학습에 쓰는 90 % 로 fit.
같은 점: AdamW 1e-3, ReduceLROnPlateau(patience 10, factor 0.8, min 1e-6) — 내부 검증 MAE 기준, 기울기 ±10 자르기,
         dropout 0.3, batch 64/256/1024 by N (연구실 규칙, N = 전체 분자 수), 최대 300 epoch, patience 30.
"""
import argparse
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import copy
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                       # 수정한 사본(feature/models)이 수업 폴더보다 먼저
from feature import mol_to_features            # noqa: E402
from models import HybridGraphRegressor        # noqa: E402
import mpcommon as mc                          # noqa: E402

T2 = os.path.join(os.getcwd(), 'T2_gin')
LEDGER = os.path.join(T2, 'ledger.csv')
N_THREADS = 4
LOSS_BETA = 0.25
MAX_EPOCHS, PATIENCE = 300, 30


def lab_batch_size(n):
    return 64 if n < 10000 else 256 if n < 100000 else 1024


# ---------------------------------------------------------------- 데이터
def build_graphs():
    """mc.load_data() 행 순서 그대로의 그래프 목록 (작업 폴더 캐시)."""
    path = os.path.join(T2, 'cache', 'graphs.pt')
    if os.path.exists(path):
        return torch.load(path, weights_only=False), None
    df, mols, y = mc.load_data()
    with mc.CostMeter('featurize') as c:
        G = []
        for m in mols:
            x, ei, ea = mol_to_features(m)
            G.append((x, ei))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(G, path)
    return G, c


def desc_matrix(train_rows):
    X = np.load(os.path.join(mc.CACHE_DIR, 'X_desc217.npy')).astype(np.float64)
    mu, sd = X[train_rows].mean(0), X[train_rows].std(0)
    sd[sd < 1e-12] = 1.0
    Z = np.clip((X - mu) / sd, -10, 10)
    return np.nan_to_num(Z, nan=0.0).astype(np.float32)


def internal_split(rows, smiles, how, seed, frac=0.1):
    rng = random.Random(seed)
    rows = list(rows); n_in = int(round(frac * len(rows)))
    if how == 'random':
        rng.shuffle(rows)
        return np.array(rows[n_in:]), np.array(rows[:n_in])
    groups = {}
    for r in rows:
        groups.setdefault(MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles[r], includeChirality=False), []).append(r)
    gl = list(groups.values()); rng.shuffle(gl)
    inner, fit = [], []
    for g in gl:
        (inner if len(inner) + len(g) <= n_in else fit).extend(g)
    return np.array(fit), np.array(inner)


def make_data(rows, G, D, y_scaled, y_raw, use_desc):
    out = []
    for r in rows:
        x, ei = G[r]
        d = Data(x=x, edge_index=ei, y=torch.tensor([[y_scaled[r]]], dtype=torch.float),
                 y_raw=torch.tensor([[y_raw[r]]], dtype=torch.float), row=torch.tensor([r]))
        if use_desc:
            d.desc = torch.from_numpy(D[r:r + 1])
        out.append(d)
    return out


def predict(model, loader, mu_y, sd_y):
    model.eval(); P, Y, R = [], [], []
    with torch.no_grad():
        for b in loader:
            P.append(model(b)); Y.append(b.y_raw); R.append(b.row)
    p = (torch.cat(P) * sd_y + mu_y).numpy().ravel()
    return torch.cat(R).numpy(), torch.cat(Y).numpy().ravel(), p


# ---------------------------------------------------------------- 학습 1회
def run(how, seed, use_desc=True, max_train=None, stage='main', max_epochs=MAX_EPOCHS, tag=None, write=True):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    df, mols, y = mc.load_data()
    smiles = df['smiles'].tolist()
    idx, _ = mc.load_split(how)
    G, featc = build_graphs()
    if featc is not None and write:
        mc.append_ledger(LEDGER, dict(run_id='T2_featurize_graphs', track='T2_gin', model='-', features='graph147',
                                      split='all', n_train=len(G),
                                      notes='수업 feature.py 원자 147 그래프 생성 (19,945분자, 1회, 캐시)', **featc.as_row()))
    train_rows = idx['train']
    if max_train:
        train_rows = np.sort(np.random.default_rng(42).choice(train_rows, max_train, replace=False))
    fit_rows, inner_rows = internal_split(train_rows, smiles, how, seed)
    mu_y, sd_y = float(y[fit_rows].mean()), float(y[fit_rows].std(ddof=1))
    ys = (y - mu_y) / sd_y
    D = desc_matrix(idx['train']) if use_desc else None   # 표준화는 train 전체로 (validation 미사용)
    n_total = (max_train / len(idx['train']) * len(y)) if max_train else len(y)
    bs = lab_batch_size(n_total)
    sets = {k: make_data(r, G, D, ys, y, use_desc) for k, r in
            (('fit', fit_rows), ('inner', inner_rows), ('val', idx['validate']))}
    ld = {k: DataLoader(v, batch_size=bs, shuffle=(k == 'fit')) for k, v in sets.items()}

    model = HybridGraphRegressor(G[0][0].shape[1], n_desc=217 if use_desc else 0)
    for p in model.parameters():
        p.register_hook(lambda g: g.clamp(-10, 10))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.8, min_lr=1e-6)
    feats = 'graph147+desc217' if use_desc else 'graph147'
    run_id = tag or f'T2_{"hyb" if use_desc else "graphonly"}_{how}_s{seed}' + (f'_n{max_train}' if max_train else '')
    print(f'[{run_id}] params {model.n_params():,}  fit {len(fit_rows)} inner {len(inner_rows)} val {len(idx["validate"])}'
          f'  bs {bs}  threads {torch.get_num_threads()}', flush=True)

    log, best, best_state, bad = [], np.inf, None, 0
    with mc.CostMeter(stage) as c:
        for ep in range(max_epochs):
            t_ep = time.perf_counter(); model.train(); tot = 0.
            for b in ld['fit']:
                opt.zero_grad()
                loss = F.smooth_l1_loss(model(b), b.y, beta=LOSS_BETA)
                loss.backward(); opt.step(); tot += loss.item() * b.num_graphs
            _, yi, pi = predict(model, ld['inner'], mu_y, sd_y)
            inner_mae = float(np.mean(np.abs(yi - pi)))
            inner_loss = float(F.smooth_l1_loss(torch.tensor((pi - mu_y) / sd_y), torch.tensor((yi - mu_y) / sd_y),
                                                beta=LOSS_BETA))
            sched.step(inner_mae)
            log.append(dict(epoch=ep, train_loss=tot / len(fit_rows), inner_loss=inner_loss, inner_MAE=inner_mae,
                            lr=opt.param_groups[0]['lr'], sec=time.perf_counter() - t_ep))
            if inner_mae < best - 1e-4:
                best, best_state, bad = inner_mae, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if ep % 10 == 0 or bad == 0:
                print(f'  ep {ep:3d} train {log[-1]["train_loss"]:.4f} inner loss {inner_loss:.4f} '
                      f'inner MAE {inner_mae:.2f} lr {log[-1]["lr"]:.1e} {log[-1]["sec"]:.1f}s' + (' *' if bad == 0 else ''),
                      flush=True)
            if bad >= PATIENCE:
                break
        model.load_state_dict(best_state)
        t0 = time.perf_counter()
        rows, yv, pv = predict(model, ld['val'], mu_y, sd_y)
        infer = (time.perf_counter() - t0) / len(rows) * 1000
    best_ep = int(np.argmin([l['inner_MAE'] for l in log]))
    m = mc.metrics(yv, pv)
    print(f'[{run_id}] epochs {len(log)} best {best_ep} inner MAE {best:.2f} | val MAE {m["MAE"]:.3f} RMSE {m["RMSE"]:.3f} '
          f'| {c.wall_s:.0f} s, {c.cpu_core_h:.3f} core-h, RAM {c.peak_ram_gb:.2f} GB', flush=True)
    if write:
        os.makedirs(os.path.join(T2, 'preds'), exist_ok=True); os.makedirs(os.path.join(T2, 'logs'), exist_ok=True)
        pd.DataFrame(dict(row=rows, y=yv, pred=pv)).to_csv(os.path.join(T2, 'preds', f'{run_id}_val.csv'), index=False)
        pd.DataFrame(log).to_csv(os.path.join(T2, 'logs', f'{run_id}_curve.csv'), index=False)
        params = (f'GIN5x128 fp64 mean | head (256,128) drop0.3 | SmoothL1 beta{LOSS_BETA} | AdamW1e-3 RLROP(10,0.8) '
                  f'| bs{bs} | ES inner10%({how}) patience{PATIENCE} max{max_epochs} | torch threads {N_THREADS} '
                  f'| epochs {len(log)} best_ep {best_ep} inner_MAE {best:.3f}')
        mc.append_ledger(LEDGER, dict(run_id=run_id, track='T2_gin', model='GIN-mean-hybrid' if use_desc else 'GIN-mean',
                                      features=feats, data_variant='full' if not max_train else f'train_sub{max_train}',
                                      objective=f'SmoothL1(beta={LOSS_BETA} sd)', split=how, seed=seed, params=params,
                                      val_MAE=round(m['MAE'], 4), val_RMSE=round(m['RMSE'], 4), n_train=len(fit_rows),
                                      infer_s_per_1k=round(infer, 4),
                                      notes=f'inner {len(inner_rows)} rows for ES; peak RAM incl. dataset', **c.as_row()))
    return dict(run_id=run_id, val=m, epochs=len(log), best_ep=best_ep, wall_s=c.wall_s, core_h=c.cpu_core_h,
                ram=c.peak_ram_gb, sec_per_epoch=float(np.mean([l['sec'] for l in log])), n_fit=len(fit_rows), bs=bs)


def probe(how='random', n_epochs=3):
    """전체 train·전체 batch 크기로 n_epochs 만 돌려 epoch 당 시간·CPU 를 잰다 (외삽용)."""
    return run(how, 0, max_epochs=n_epochs, stage='pilot_probe', tag=f'T2_probe_full_{how}_{n_epochs}ep', write=True)


PRIORITY = ['hyb:random:0', 'hyb:random:1', 'hyb:random:2', 'hyb:scaffold:0', 'hyb:scaffold:1', 'hyb:scaffold:2',
            'graphonly:random:1', 'graphonly:scaffold:1']


def _rid(spec):
    kind, how, seed = spec.split(':')
    return f'T2_{kind}_{how}_s{seed}'


def run_queue(specs, budget, per_default=1.3):
    """각 실행 전에: ledger 누적 + 추정 x (자기 1 + 다른 프로세스 진행 중 + 더 높은 우선순위 미시작) <= budget 이면 실행."""
    st = os.path.join(T2, 'state'); os.makedirs(st, exist_ok=True)
    for spec in specs:
        rid = _rid(spec); kind, how, seed = spec.split(':')
        if os.path.exists(os.path.join(T2, 'preds', f'{rid}_val.csv')):
            print('skip (exists)', rid, flush=True); open(os.path.join(st, rid + '.done'), 'w').close(); continue
        L = pd.read_csv(LEDGER, encoding='utf-8-sig')
        used = float(L['cpu_core_h'].sum())
        full = L[L['run_id'].isin([_rid(x) for x in PRIORITY])]
        per = float(full['cpu_core_h'].max()) if len(full) else per_default
        inflight = [x for x in PRIORITY if _rid(x) != rid and os.path.exists(os.path.join(st, _rid(x) + '.started'))
                    and not os.path.exists(os.path.join(st, _rid(x) + '.done'))]
        higher = [x for x in PRIORITY[:PRIORITY.index(spec)] if not os.path.exists(os.path.join(st, _rid(x) + '.started'))]
        need = used + per * (1 + len(inflight) + len(higher))
        print(f'[guard] {rid}: used {used:.3f} per {per:.3f} inflight {len(inflight)} higher_pending {len(higher)} '
              f'-> {need:.3f} vs {budget}', flush=True)
        if need > budget:
            print(f'BUDGET skip {rid}', flush=True); open(os.path.join(st, rid + '.skipped'), 'w').close(); continue
        open(os.path.join(st, rid + '.started'), 'w').close()
        try:
            run(how, int(seed), use_desc=(kind == 'hyb'), stage='main' if kind == 'hyb' else 'control', tag=rid)
        finally:
            open(os.path.join(st, rid + '.done'), 'w').close()
        print('DONE', rid, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='main', choices=['pilot', 'main', 'control', 'probe'])
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--runs', default='')
    ap.add_argument('--no_desc', action='store_true')
    ap.add_argument('--budget_core_h', type=float, default=1e9, help='이 누적값을 넘을 것 같으면 남은 실행을 건너뜀')
    a = ap.parse_args()
    N_THREADS = a.threads
    torch.set_num_threads(N_THREADS); torch.set_num_interop_threads(1)
    if a.stage == 'probe':
        r = run('random', 0, max_epochs=3, stage='pilot_probe', tag=f'T2_probe_full_random_3ep_t{N_THREADS}')
        print('PROBE', N_THREADS, r['sec_per_epoch'], r['core_h'], flush=True)
    elif a.stage == 'pilot':
        r = run('random', 0, max_train=5000, stage='pilot')
        print('PILOT', r, flush=True)
        r2 = probe('random', 3)
        print('PROBE', r2, flush=True)
    else:   # --stage main : 우선순위 예산 가드가 있는 실행 대기열 (두 프로세스가 state 파일로 공유)
        run_queue(a.runs.split(','), a.budget_core_h)
