"""s4_near_duplicates.py : 근접 중복 점검 (세트 간 InChIKey 전체·첫 블록 겹침, Morgan Tanimoto 1.0, 최근접 유사도)과
그것이 기준선 test MAE 에 주는 영향. 모델 적합 없음 — 저장된 기준선 예측만 사용.
출력: phase1/tables/near_duplicates.csv, near_dup_block1_pairs.csv, near_dup_effect.csv, test_nn_similarity_mae.csv,
      overlap_test_rows_{random,scaffold}.csv
"""
import itertools
import os

import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

import mpcommon as mc

TAB = os.path.join(mc.ROOT, 'phase1', 'tables')
PRED = os.path.join(mc.ROOT, 'phase1', 'preds')
LEDGER = os.path.join(mc.ROOT, 'phase1', 'ledger_errdecomp.csv')

df, mols, y = mc.load_data()
smiles = df['smiles'].values
N = len(y)
FL = pd.read_csv(os.path.join(mc.CACHE_DIR, 'mol_flags.csv'))
assert (FL['smiles'].values == smiles).all()

cost = mc.CostMeter('탐색').__enter__()
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)      # 입체 무시(기본), 비트 지문
fps = [gen.GetFingerprint(m) for m in mols]

nd_rows, eff_rows, nn_rows, pair_rows = [], [], [], []


def mae(v):
    return float(np.mean(v)) if len(v) else np.nan


for how in ('random', 'scaffold'):
    idx, s = mc.load_split(how)
    s = s.sort_values('row').reset_index(drop=True)
    assert (s['smiles'].values == smiles).all()
    key, blk = s['inchikey'].values, s['inchikey'].str[:14].values
    sets = {k: idx[k] for k in ('train', 'validate', 'test')}

    # --- 세트 쌍별 겹침 수 ---
    nn = {}
    for a, b in (('test', 'train'), ('validate', 'train'), ('test', 'validate')):
        ra, rb = sets[a], sets[b]
        for crit, arr in (('InChIKey 전체', key), ('InChIKey 첫 블록(입체 무시)', blk)):
            cnt_b = pd.Series(arr[rb]).value_counts()
            hit = pd.Series(arr[ra]).map(cnt_b).fillna(0).astype(int).values
            nd_rows.append(dict(section='세트 간 겹침', split=how, set_a=a, set_b=b, criterion=crit, n_a=len(ra), n_b=len(rb),
                                n_a_with_match=int((hit > 0).sum()), n_pairs=int(hit.sum())))
        fb = [fps[i] for i in rb]
        best, best_row, n1 = np.zeros(len(ra)), np.zeros(len(ra), int), np.zeros(len(ra), int)
        for j, i in enumerate(ra):
            sim = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fb))
            k = int(sim.argmax())
            best[j], best_row[j], n1[j] = sim[k], rb[k], int((sim >= 1.0).sum())
        nn[(a, b)] = (best, best_row, n1)
        nd_rows.append(dict(section='세트 간 겹침', split=how, set_a=a, set_b=b, criterion='Morgan(r2, 2048 bit) Tanimoto = 1.0',
                            n_a=len(ra), n_b=len(rb), n_a_with_match=int((n1 > 0).sum()), n_pairs=int(n1.sum())))
        # Tanimoto 1.0 인데 첫 블록이 다른 것(지문 충돌: 동족체·위치 이성질체 등)
        same_blk = pd.Series(blk[ra]).isin(set(blk[rb])).values
        nd_rows.append(dict(section='세트 간 겹침', split=how, set_a=a, set_b=b, criterion='Tanimoto = 1.0 이지만 첫 블록은 다름(지문 충돌)',
                            n_a=len(ra), n_b=len(rb), n_a_with_match=int(((n1 > 0) & ~same_blk).sum()), n_pairs=np.nan))
        print(f'[{how}] {a} vs {b}: full={nd_rows[-4]["n_a_with_match"]} block1={nd_rows[-3]["n_a_with_match"]} '
              f'tani1={nd_rows[-2]["n_a_with_match"]} (pairs {nd_rows[-2]["n_pairs"]})', flush=True)

    # --- test 행별 겹침 표 ---
    te, tr, va = sets['test'], sets['train'], sets['validate']
    kt, kv = set(key[tr]), set(key[va])
    bt, bv = set(blk[tr]), set(blk[va])
    blk_to_train = pd.Series(tr).groupby(blk[tr]).apply(list).to_dict()
    blk_to_val = pd.Series(va).groupby(blk[va]).apply(list).to_dict()
    T = pd.DataFrame(dict(row=te, smiles=smiles[te], inchikey=key[te], y=y[te]))
    T['full_in_train'] = [k in kt for k in key[te]]
    T['full_in_val'] = [k in kv for k in key[te]]
    T['block1_in_train'] = [b in bt for b in blk[te]]
    T['block1_in_val'] = [b in bv for b in blk[te]]
    T['tani1_in_train'] = nn[('test', 'train')][2] > 0
    T['tani1_in_val'] = nn[('test', 'validate')][2] > 0
    T['nn_train_sim'] = nn[('test', 'train')][0]
    T['nn_train_row'] = nn[('test', 'train')][1]
    T['nn_train_y'] = y[T['nn_train_row'].values]
    T['train_rows_same_block1'] = [';'.join(map(str, blk_to_train.get(b, []))) for b in blk[te]]
    T['val_rows_same_block1'] = [';'.join(map(str, blk_to_val.get(b, []))) for b in blk[te]]
    T['train_y_same_block1'] = [';'.join(f'{y[i]:.2f}' for i in blk_to_train.get(b, [])) for b in blk[te]]
    T['overlap_block1_train_or_val'] = T['block1_in_train'] | T['block1_in_val']
    T['overlap_any_incl_tani1'] = T['overlap_block1_train_or_val'] | T['tani1_in_train'] | T['tani1_in_val']
    T[T['overlap_any_incl_tani1']].to_csv(os.path.join(TAB, f'overlap_test_rows_{how}.csv'), index=False, encoding='utf-8-sig')

    # --- 영향: 겹치는 test 분자만 / 뺀 나머지 ---
    preds = {'refit_njobs8': pd.read_csv(os.path.join(PRED, f'baseline_xgb_desc217_{how}_test.csv'))}
    if how == 'random':
        o = pd.read_csv(os.path.join(mc.ROOT, 'reference', 'orig_baseline_random_test_pred.csv'))
        assert (o['smiles'].values == smiles[te]).all() and np.allclose(o['y'].values, y[te])
        preds['orig_27.50'] = o.assign(row=te)
    subsets = {'InChIKey 전체가 train 에 있음': T['full_in_train'].values,
               '첫 블록이 train 에 있음': T['block1_in_train'].values,
               'Tanimoto 1.0 인 train 분자 있음': T['tani1_in_train'].values,
               '위 셋 중 하나(train)': (T['block1_in_train'] | T['tani1_in_train']).values,
               '첫 블록이 train 또는 validate 에 있음 (9절 감사 1번 기준)': T['overlap_block1_train_or_val'].values,
               '첫 블록 또는 Tanimoto 1.0 (train 또는 validate)': T['overlap_any_incl_tani1'].values}
    for pname, p in preds.items():
        assert (p['row'].values == te).all()
        ae = np.abs(p['pred'].values - y[te])
        for sname, mask in subsets.items():
            rng = np.random.default_rng(0)
            d = []
            for _ in range(2000):                      # (나머지 MAE - 전체 MAE) 의 부트스트랩 (test 표본 변동)
                bi = rng.integers(0, len(ae), len(ae))
                mb = mask[bi]
                d.append(ae[bi][~mb].mean() - ae[bi].mean())
            lo, hi = np.percentile(d, [2.5, 97.5])
            eff_rows.append(dict(split=how, pred_source=pname, subset=sname, n_test=len(ae), n_subset=int(mask.sum()),
                                 MAE_subset_K=mae(ae[mask]), n_rest=int((~mask).sum()), MAE_rest_K=mae(ae[~mask]), MAE_all_K=mae(ae),
                                 delta_rest_minus_all_K=mae(ae[~mask]) - mae(ae), delta_ci_lo=lo, delta_ci_hi=hi,
                                 nn_label_abs_diff_subset_K=np.nan))
        # 겹치는 분자에서, 같은 첫 블록 train 라벨과 test 라벨의 차이(= '베끼기' 모델의 오차)
        m = T['block1_in_train'].values
        cp = [abs(y[r] - np.mean([y[i] for i in blk_to_train[blk[r]]])) for r in te[m]]
        eff_rows.append(dict(split=how, pred_source=pname, subset="참고: 같은 첫 블록 train 라벨을 그대로 쓴 '베끼기' 오차", n_test=len(ae),
                             n_subset=int(m.sum()), MAE_subset_K=mae(cp), n_rest=np.nan, MAE_rest_K=np.nan, MAE_all_K=mae(ae),
                             delta_rest_minus_all_K=np.nan, delta_ci_lo=np.nan, delta_ci_hi=np.nan, nn_label_abs_diff_subset_K=mae(cp)))

    # --- 최근접 유사도 구간별 MAE (test: 기준선 예측 / validation: 제공된 val 예측) ---
    edges = np.round(np.arange(0, 1.01, 0.1), 1)
    def nn_table(set_name, rows_, sim, pdict):
        for pname, p in pdict.items():
            ae = np.abs(p['pred'].values - y[rows_])
            for lo_, hi_ in zip(edges[:-1], edges[1:]):
                mask = (sim >= lo_) & (sim < hi_)
                lab = f'{lo_:.1f}-{hi_:.1f}'
                ci = mc.boot_ci(ae[mask]) if mask.sum() >= 5 else (np.nan, np.nan)
                nn_rows.append(dict(split=how, set=set_name, pred_source=pname, nn_sim_bin=lab, n=int(mask.sum()), frac=float(mask.mean()),
                                    MAE_K=mae(ae[mask]), MAE_ci_lo=ci[0], MAE_ci_hi=ci[1], mean_nn_sim=float(sim[mask].mean()) if mask.any() else np.nan))
            mask = sim >= 1.0
            ci = mc.boot_ci(ae[mask]) if mask.sum() >= 5 else (np.nan, np.nan)
            nn_rows.append(dict(split=how, set=set_name, pred_source=pname, nn_sim_bin='1.0', n=int(mask.sum()), frac=float(mask.mean()),
                                MAE_K=mae(ae[mask]), MAE_ci_lo=ci[0], MAE_ci_hi=ci[1], mean_nn_sim=1.0 if mask.any() else np.nan))
    nn_table('test', te, nn[('test', 'train')][0], preds)
    v = pd.read_csv(os.path.join(PRED, f'baseline_xgb_desc217_{how}_val.csv'))
    assert (v['row'].values == va).all()
    nn_table('validate', va, nn[('validate', 'train')][0], {'refit_njobs8': v})

# ---- 데이터 전체: 입체 미표기 수, 같은 첫 블록 안 라벨 차이 ---------------------------------------
s = pd.read_csv(os.path.join(mc.SPLIT_DIR, 'random.csv')).sort_values('row').reset_index(drop=True)
key, blk = s['inchikey'].values, s['inchikey'].str[:14].values
for crit, col in (('이중결합 입체 가능한데 미표기', 'dblbond_stereo_unspecified'), ('입체 중심 가능한데 미표기', None),
                  ('입체 가능한데 미표기(중심 또는 이중결합)', 'stereo_possible_unspecified'), ('입체 표기 있음', 'stereo_annotated')):
    cnt = int((FL['n_unspec_center'] > 0).sum()) if col is None else int(FL[col].sum())
    nd_rows.append(dict(section='데이터 전체', split='-', set_a='all', set_b='-', criterion=crit, n_a=N, n_b=np.nan, n_a_with_match=cnt, n_pairs=np.nan))
nd_rows.append(dict(section='데이터 전체', split='-', set_a='all', set_b='-', criterion='InChIKey 전체 고유 수', n_a=N, n_b=np.nan,
                    n_a_with_match=int(pd.Series(key).nunique()), n_pairs=np.nan))
nd_rows.append(dict(section='데이터 전체', split='-', set_a='all', set_b='-', criterion='InChIKey 첫 블록 고유 수', n_a=N, n_b=np.nan,
                    n_a_with_match=int(pd.Series(blk).nunique()), n_pairs=np.nan))
nd_rows.append(dict(section='데이터 전체', split='-', set_a='all', set_b='-', criterion='첫 블록을 다른 행과 공유하는 행 수', n_a=N, n_b=np.nan,
                    n_a_with_match=int(pd.Series(blk).duplicated(keep=False).sum()), n_pairs=np.nan))


def marks(smi):
    return ('@' in smi), ('/' in smi or '\\' in smi)


for b, g in pd.DataFrame(dict(row=np.arange(N), blk=blk, key=key)).groupby('blk'):
    if len(g) < 2:
        continue
    for (ia, ka), (ib, kb) in itertools.combinations(zip(g['row'], g['key']), 2):
        (a_at, a_ez), (b_at, b_ez) = marks(smiles[ia]), marks(smiles[ib])
        a_any, b_any = a_at or a_ez, b_at or b_ez
        if ka == kb:
            typ = 'InChIKey 전체 동일(SMILES 만 다름)'
        elif a_any != b_any:
            typ = '한쪽만 입체 표기(표기 vs 미표기)' + (' — E/Z' if (a_ez or b_ez) and not (a_at or b_at) else '')
        elif a_any and b_any:
            typ = '둘 다 표기 — E/Z 만 다름' if (a_ez and b_ez and not a_at and not b_at) else '둘 다 표기 — 입체 중심 포함'
        else:
            typ = '둘 다 미표기(동위원소·양성자화 등 다른 층)'
        pair_rows.append(dict(block1=b, type=typ, row_a=ia, row_b=ib, smiles_a=smiles[ia], smiles_b=smiles[ib], y_a=y[ia], y_b=y[ib],
                              abs_diff_K=abs(y[ia] - y[ib])))
PR = pd.DataFrame(pair_rows)
PR.to_csv(os.path.join(TAB, 'near_dup_block1_pairs.csv'), index=False, encoding='utf-8-sig')
for typ, g in list(PR.groupby('type')) + [('같은 첫 블록 전체', PR)]:
    d = g['abs_diff_K'].values
    nd_rows.append(dict(section='같은 첫 블록 쌍의 라벨 차이', split='-', set_a='all', set_b='all', criterion=typ, n_a=N, n_b=np.nan,
                        n_a_with_match=np.nan, n_pairs=len(g), mean_abs_diff_K=d.mean(), median_abs_diff_K=np.median(d),
                        q90_abs_diff_K=np.quantile(d, 0.9), frac_lt1=np.mean(d < 1), frac_ge10=np.mean(d >= 10), frac_ge50=np.mean(d >= 50)))

cost.__exit__(None, None, None)
ND = pd.DataFrame(nd_rows)
ND.to_csv(os.path.join(TAB, 'near_duplicates.csv'), index=False, encoding='utf-8-sig')
EF = pd.DataFrame(eff_rows)
EF.to_csv(os.path.join(TAB, 'near_dup_effect.csv'), index=False, encoding='utf-8-sig')
NN = pd.DataFrame(nn_rows)
NN.to_csv(os.path.join(TAB, 'test_nn_similarity_mae.csv'), index=False, encoding='utf-8-sig')
mc.append_ledger(LEDGER, dict(run_id='errdecomp_s4_near_duplicates', track='errdecomp', model='-(저장된 기준선 예측 사용)',
                              features='Morgan r2 2048 bit (유사도 계산용)', data_variant='-', objective='-', split='random+scaffold',
                              seed=0, params='{}', n_train=0, notes='근접 중복 점검; 모델 적합 없음', **cost.as_row()))
pd.set_option('display.width', 250)
print(ND[ND.section != '세트 간 겹침'].drop(columns=['split', 'set_a', 'set_b', 'n_b']).round(3).to_string(index=False))
print(EF[['split', 'pred_source', 'subset', 'n_subset', 'MAE_subset_K', 'MAE_rest_K', 'MAE_all_K', 'delta_rest_minus_all_K', 'delta_ci_lo', 'delta_ci_hi']].round(3).to_string(index=False))
print(NN[NN.set == 'test'][['split', 'pred_source', 'nn_sim_bin', 'n', 'MAE_K']].round(2).to_string(index=False))
