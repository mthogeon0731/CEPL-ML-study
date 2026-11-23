"""gin_p3.py : 3단계 GIN 하이브리드 재학습 → validation 재현 점검용 예측 + test 예측(라벨 없이 저장).

T2_gin/src/train_hybrid.py 의 run() 을 그대로 따른다 (설정·순서·난수 사용 동일). 바뀐 점만:
  1. 분할 how 의 test 행 라벨을 NaN 으로 가린 뒤 학습한다 (test 라벨은 이 스크립트에서 쓰이지 않음).
  2. 학습 후 최적 가중치로 validation 과 test 를 예측한다. test 파일은 row, pred 만 (y 없음).
  3. 가중치(state_dict)를 P3/models/ 에 저장한다. ledger 는 프로세스별 파일(P3/work/gin/ledger_<run>.csv)에 쓴 뒤 합친다.
  4. torch 1 스레드 고정 (T2 본 실행과 같음). 그래프 캐시는 P3/work/graphs.pt.
  5. --fit trainval: train+validation 으로 학습(보조 결과; 내부 10 % 조기 종료는 같은 규칙, 기술자 표준화도 학습 행으로).
사용 (작업 공간 루트에서, mlstudy 래퍼):
  run_mlstudy.ps1 P3/scripts/gin/gin_p3.py --stage graphs
  run_mlstudy.ps1 P3/scripts/gin/gin_p3.py --how random --seed 0 [--fit trainval]
"""
import argparse
import copy
import os
import random
import sys
import time
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402
from torch_geometric.data import Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                        # T2 의 models.py 사본 (HybridGraphRegressor)
from feature import mol_to_features             # noqa: E402  (수업 폴더 feature.py, PYTHONPATH)
from models import HybridGraphRegressor         # noqa: E402
import mpcommon as mc                           # noqa: E402

WS = os.getcwd()
WORK = os.path.join(WS, 'P3', 'work', 'gin')
MODELS = os.path.join(WS, 'P3', 'models')
N_THREADS = 1
LOSS_BETA = 0.25
MAX_EPOCHS, PATIENCE = 300, 30


def lab_batch_size(n):
    return 64 if n < 10000 else 256 if n < 100000 else 1024


def build_graphs():
    path = os.path.join(WS, 'P3', 'work', 'graphs.pt')
    if os.path.exists(path):
        return torch.load(path, weights_only=False), None
    df, mols, y = mc.load_data()
    with mc.CostMeter('featurize') as c:
        G = []
        for m in mols:
            x, ei, ea = mol_to_features(m)
            G.append((x, ei))
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


def run(how, seed, fit_set='train', use_desc=True, max_epochs=MAX_EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    df, mols, y = mc.load_data()
    smiles = df['smiles'].tolist()
    idx, _ = mc.load_split(how)
    y = np.asarray(y, float).copy()
    y[idx['test']] = np.nan                         # test 라벨 차단
    G, featc = build_graphs()
    train_rows = idx['train'] if fit_set == 'train' else np.concatenate([idx['train'], idx['validate']])
    fit_rows, inner_rows = internal_split(train_rows, smiles, how, seed)
    assert np.isfinite(y[train_rows]).all()
    mu_y, sd_y = float(y[fit_rows].mean()), float(y[fit_rows].std(ddof=1))
    ys = (y - mu_y) / sd_y
    D = desc_matrix(train_rows) if use_desc else None
    bs = lab_batch_size(len(y))
    sets = {k: make_data(r, G, D, ys, y, use_desc) for k, r in
            (('fit', fit_rows), ('inner', inner_rows), ('val', idx['validate']), ('test', idx['test']))}
    ld = {k: DataLoader(v, batch_size=bs, shuffle=(k == 'fit')) for k, v in sets.items()}
    model = HybridGraphRegressor(G[0][0].shape[1], n_desc=217 if use_desc else 0)
    for p in model.parameters():
        p.register_hook(lambda g: g.clamp(-10, 10))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.8, min_lr=1e-6)
    run_id = f'P3_gin_{how}_{fit_set}_s{seed}'
    print(f'[{run_id}] params {model.n_params():,} fit {len(fit_rows)} inner {len(inner_rows)} val {len(idx["validate"])} '
          f'test {len(idx["test"])} bs {bs} threads {torch.get_num_threads()}', flush=True)
    log, best, best_state, bad = [], np.inf, None, 0
    with mc.CostMeter('최종') as c:
        for ep in range(max_epochs):
            t_ep = time.perf_counter(); model.train(); tot = 0.
            for b in ld['fit']:
                opt.zero_grad()
                loss = F.smooth_l1_loss(model(b), b.y, beta=LOSS_BETA)
                loss.backward(); opt.step(); tot += loss.item() * b.num_graphs
            _, yi, pi = predict(model, ld['inner'], mu_y, sd_y)
            inner_mae = float(np.mean(np.abs(yi - pi)))
            sched.step(inner_mae)
            log.append(dict(epoch=ep, train_loss=tot / len(fit_rows), inner_MAE=inner_mae, lr=opt.param_groups[0]['lr'],
                            sec=time.perf_counter() - t_ep))
            if inner_mae < best - 1e-4:
                best, best_state, bad = inner_mae, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if ep % 10 == 0 or bad == 0:
                print(f'  ep {ep:3d} train {log[-1]["train_loss"]:.4f} inner MAE {inner_mae:.2f} lr {log[-1]["lr"]:.1e} '
                      f'{log[-1]["sec"]:.1f}s' + (' *' if bad == 0 else ''), flush=True)
            if bad >= PATIENCE:
                break
        model.load_state_dict(best_state)
        rv, yv, pv = predict(model, ld['val'], mu_y, sd_y)
        t0 = time.perf_counter()
        rt, _, pt = predict(model, ld['test'], mu_y, sd_y)
        infer = (time.perf_counter() - t0) / len(rt) * 1000
    best_ep = int(np.argmin([l['inner_MAE'] for l in log]))
    os.makedirs(WORK, exist_ok=True); os.makedirs(MODELS, exist_ok=True)
    torch.save(dict(state_dict=best_state, mu_y=mu_y, sd_y=sd_y, how=how, seed=seed, fit_set=fit_set),
               os.path.join(MODELS, f'{run_id}.pt'))
    pd.DataFrame(dict(row=rv, y=yv, pred=pv)).to_csv(os.path.join(WORK, f'{run_id}_val.csv'), index=False)
    pd.DataFrame(dict(row=rt, pred=pt)).to_csv(os.path.join(WORK, f'{run_id}_test.csv'), index=False)
    pd.DataFrame(log).to_csv(os.path.join(WORK, f'{run_id}_curve.csv'), index=False)
    vmae = float(np.mean(np.abs(yv - pv))) if fit_set == 'train' else float('nan')
    print(f'[{run_id}] epochs {len(log)} best {best_ep} inner MAE {best:.3f} | val MAE {vmae:.4f} | {c.wall_s:.0f} s '
          f'{c.cpu_core_h:.3f} core-h RAM {c.peak_ram_gb:.2f} GB', flush=True)
    params = (f'GIN5x128 fp64 mean | head (256,128) drop0.3 | SmoothL1 beta{LOSS_BETA} | AdamW1e-3 RLROP(10,0.8) | bs{bs} '
              f'| ES inner10%({how}) patience{PATIENCE} max{max_epochs} | torch threads {N_THREADS} | epochs {len(log)} '
              f'best_ep {best_ep} inner_MAE {best:.3f}')
    led = os.path.join(WORK, f'ledger_{run_id}.csv')
    if os.path.exists(led):
        os.remove(led)
    mc.append_ledger(led, dict(run_id=run_id, track='P3', model='GIN-mean-hybrid', features='graph147+desc217',
                               data_variant=fit_set, objective=f'SmoothL1(beta={LOSS_BETA} sd)', split=how, seed=seed,
                               params=params, val_MAE=(round(vmae, 4) if fit_set == 'train' else ''), n_train=len(fit_rows),
                               infer_s_per_1k=round(infer, 4), stage='최종',
                               notes=f'predict=validate+test(라벨 없이 저장); inner {len(inner_rows)} ES',
                               **{k: v for k, v in c.as_row().items() if k != 'stage'}))
    if featc is not None:
        mc.append_ledger(os.path.join(WORK, 'ledger_P3_gin_graphs.csv'),
                         dict(run_id='P3_gin_graphs', track='P3', model='-', features='graph147', split='all', n_train=len(G),
                              notes='수업 feature.py 그래프 생성', **featc.as_row()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='run', choices=['run', 'graphs'])
    ap.add_argument('--how', default='random')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--fit', default='train', choices=['train', 'trainval'])
    a = ap.parse_args()
    torch.set_num_threads(N_THREADS); torch.set_num_interop_threads(1)
    if a.stage == 'graphs':
        G, c = build_graphs()
        if c is not None:
            os.makedirs(WORK, exist_ok=True)
            mc.append_ledger(os.path.join(WORK, 'ledger_P3_gin_graphs.csv'),
                             dict(run_id='P3_gin_graphs', track='P3', model='-', features='graph147', split='all', n_train=len(G),
                                  notes='수업 feature.py 그래프 생성 (19,945분자, 1회)', **c.as_row()))
        print('graphs', len(G), flush=True)
    else:
        run(a.how, a.seed, a.fit)
