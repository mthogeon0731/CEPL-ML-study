"""s3_error_structure.py : 오차 구조 분석 — train 5-fold out-of-fold(OOF) 예측 + validation 예측만 사용 (test 잔차 미사용).

random : KFold(5, shuffle, random_state 42) / scaffold : GroupKFold(5, Murcko 뼈대 단위). 기준선 XGB, n_jobs=8, random_state 42.
출력: phase1/preds/oof_baseline_{random,scaffold}_train.csv, cache/mol_flags.csv,
      phase1/tables/resid_by_mp_bin.csv, mae_heavy_x_mp.csv, mae_by_family.csv, top50_oof_errors.csv, top50_oof_errors_scaffold.csv
"""
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

import mpcommon as mc

TAB = os.path.join(mc.ROOT, 'phase1', 'tables')
PRED = os.path.join(mc.ROOT, 'phase1', 'preds')
LEDGER = os.path.join(mc.ROOT, 'phase1', 'ledger_errdecomp.csv')
N_JOBS, SEED = 8, 42

df, mols, y = mc.load_data()
X = mc.desc217(mols)
smiles = df['smiles'].values
N = len(y)

# ---- 분자 계열 표지 (구조에서만 계산) ----------------------------------------------------------
PAT = {k: Chem.MolFromSmarts(v) for k, v in dict(
    nitro='[N+](=O)[O-]', cooh='[CX3](=[OX1])[OX2H1]', amide='[CX3](=[OX1])[NX3]').items()}
HAL = {9, 17, 35, 53}


def flags_of(m, smi):
    heavy = m.GetNumHeavyAtoms()
    n_ring = m.GetRingInfo().NumRings()
    n_hal = sum(a.GetAtomicNum() in HAL for a in m.GetAtoms())
    nitro_atoms = {i for match in m.GetSubstructMatches(PAT['nitro']) for i in match}
    charged = [a.GetIdx() for a in m.GetAtoms() if a.GetFormalCharge() != 0]
    si = Chem.FindPotentialStereo(m, cleanIt=False, flagPossible=True)
    un_c = sum(1 for s in si if s.type == Chem.StereoType.Atom_Tetrahedral and s.specified == Chem.StereoSpecified.Unspecified)
    un_b = sum(1 for s in si if s.type == Chem.StereoType.Bond_Double and s.specified == Chem.StereoSpecified.Unspecified)
    ranks = list(Chem.CanonicalRankAtoms(m, breakTies=False))
    sym_ratio = len(set(ranks)) / max(heavy, 1)
    return dict(
        heavy=heavy, n_ring=n_ring, n_hal=n_hal, hbd=rdMolDescriptors.CalcNumHBD(m),
        rotb=rdMolDescriptors.CalcNumRotatableBonds(m), sym_ratio=sym_ratio, n_unspec_center=un_c, n_unspec_dbl=un_b,
        multi_fragment='.' in smi,
        charged_any=len(charged) > 0,
        charged_excl_nitro=any(i not in nitro_atoms for i in charged),
        stereo_annotated=('@' in smi) or ('/' in smi) or ('\\' in smi),
        stereo_possible_unspecified=(un_c + un_b) > 0,
        dblbond_stereo_unspecified=un_b > 0,
        acyclic=n_ring == 0,
        polycyclic_ge3=n_ring >= 3,
        halogen_heavy=(n_hal >= 4) or (n_hal / max(heavy, 1) >= 0.30),
        carboxylic_acid=m.HasSubstructMatch(PAT['cooh']),
        amide_urea=m.HasSubstructMatch(PAT['amide']),
        hbd_ge3=rdMolDescriptors.CalcNumHBD(m) >= 3,
        rigid_rotb0=rdMolDescriptors.CalcNumRotatableBonds(m) == 0,
        sym2d_high=sym_ratio <= 0.5,
    )


FL = pd.DataFrame([flags_of(m, s) for m, s in zip(mols, smiles)])
FL.insert(0, 'row', np.arange(N))
FL.insert(1, 'smiles', smiles)
FL.to_csv(os.path.join(mc.CACHE_DIR, 'mol_flags.csv'), index=False, encoding='utf-8-sig')
FAMS = ['multi_fragment', 'charged_any', 'charged_excl_nitro', 'stereo_annotated', 'stereo_possible_unspecified',
        'dblbond_stereo_unspecified', 'acyclic', 'polycyclic_ge3', 'halogen_heavy', 'carboxylic_acid', 'amide_urea',
        'hbd_ge3', 'rigid_rotb0', 'sym2d_high']
FAM_KO = dict(multi_fragment="여러 조각('.' 포함: 염·혼합물)", charged_any='전하 원자 있음(나이트로 포함)',
              charged_excl_nitro='전하 원자 있음(나이트로 제외)', stereo_annotated='입체 표기 있음(@ 또는 / \\)',
              stereo_possible_unspecified='입체 가능한데 미표기(중심 또는 이중결합)', dblbond_stereo_unspecified='이중결합 입체 가능한데 미표기',
              acyclic='사슬형(고리 0)', polycyclic_ge3='다환(고리 >= 3)', halogen_heavy='할로젠 다량(>= 4개 또는 >= 30 %)',
              carboxylic_acid='카복실산', amide_urea='아마이드/우레아', hbd_ge3='수소결합 주개 >= 3',
              rigid_rotb0='회전 가능 결합 0(강직)', sym2d_high='2D 대칭 높음(고유 순위/무거운 원자 <= 0.5)')
print('[flags] n per family:', {k: int(FL[k].sum()) for k in FAMS}, flush=True)

# ---- OOF 예측 ---------------------------------------------------------------------------------
splits = {how: mc.load_split(how)[0] for how in ('random', 'scaffold')}
oof = {}
for how in ('random', 'scaffold'):
    tr = splits[how]['train']
    if how == 'random':
        folds = list(KFold(5, shuffle=True, random_state=SEED).split(tr))
        cv_name = 'KFold(5, shuffle, random_state=42)'
    else:
        groups = [MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False) for s in smiles[tr]]
        folds = list(GroupKFold(5).split(tr, groups=groups))
        cv_name = 'GroupKFold(5, Murcko scaffold)'
    p, fold_id = np.full(len(tr), np.nan), np.full(len(tr), -1)
    with mc.CostMeter('탐색') as c:
        for k, (a, b) in enumerate(folds):
            m = XGBRegressor(**mc.BASE_XGB, n_jobs=N_JOBS, random_state=SEED)
            m.fit(X[tr[a]], y[tr[a]])
            p[b], fold_id[b] = m.predict(X[tr[b]]), k
            print(f'[oof:{how}] fold {k}: n_fit={len(a)} n_out={len(b)} MAE={np.mean(np.abs(p[b] - y[tr[b]])):.3f}', flush=True)
    assert not np.isnan(p).any()
    o = pd.DataFrame(dict(row=tr, smiles=smiles[tr], y=y[tr], pred=p, fold=fold_id))
    o.to_csv(os.path.join(PRED, f'oof_baseline_{how}_train.csv'), index=False, encoding='utf-8-sig')
    mm = mc.metrics(o.y, o.pred)
    mc.append_ledger(LEDGER, dict(
        run_id=f'errdecomp_s3_oof_{how}', track='errdecomp', model='XGBoost(base)', features='RDKit desc 217',
        data_variant=f'train 5-fold OOF, {cv_name}', objective='reg:squarederror', split=how, seed=SEED,
        params=json.dumps(dict(mc.BASE_XGB, n_jobs=N_JOBS, random_state=SEED)), val_MAE=round(mm['MAE'], 4),
        val_RMSE=round(mm['RMSE'], 4), n_train=int(len(tr) * 0.8),
        notes='val_MAE/val_RMSE 칸은 OOF 값(5개 적합 합산 비용); test 미사용', **c.as_row()))
    print(f'[oof:{how}] OOF MAE={mm["MAE"]:.3f} RMSE={mm["RMSE"]:.3f} R2={mm["R2"]:.3f} n={len(o)}', flush=True)
    oof[how] = o

# ---- 분석 대상 4개 (split x source) ------------------------------------------------------------
sets = {}
for how in ('random', 'scaffold'):
    sets[(how, 'oof_train')] = oof[how][['row', 'y', 'pred']].copy()
    v = pd.read_csv(os.path.join(PRED, f'baseline_xgb_desc217_{how}_val.csv'))
    assert np.allclose(v['y'].values, y[v['row'].values])
    sets[(how, 'validation')] = v[['row', 'y', 'pred']].copy()
for k, d in sets.items():
    d['resid'] = d['pred'] - d['y']
    d['abs'] = d['resid'].abs()
    for col in ['heavy'] + FAMS:
        d[col] = FL[col].values[d['row'].values]

MP_EDGES, MP_LAB = [-np.inf, 250, 350, 450, 550, np.inf], ['<250', '250-350', '350-450', '450-550', '>550']
HV_EDGES, HV_LAB = [-np.inf, 10, 20, 30, 45, np.inf], ['<=10', '11-20', '21-30', '31-45', '>45']

# (i) MP 구간별 잔차 — 참값 구간(y)과, 보정 가능성 판단용으로 예측값 구간(pred) 둘 다
rows = []
for (how, src), d in sets.items():
    for by in ('y', 'pred'):
        b = pd.cut(d[by], MP_EDGES, labels=MP_LAB, right=False)
        for lab in MP_LAB:
            g = d[b == lab]
            if len(g) == 0:
                rows.append(dict(split=how, source=src, bin_by=by, mp_bin_K=lab, n=0)); continue
            lo, hi = mc.boot_ci(g['resid'].values)
            alo, ahi = mc.boot_ci(g['abs'].values)
            rows.append(dict(split=how, source=src, bin_by=by, mp_bin_K=lab, n=len(g), MAE_K=g['abs'].mean(), MAE_ci_lo=alo,
                             MAE_ci_hi=ahi, mean_resid_K=g['resid'].mean(), resid_ci_lo=lo, resid_ci_hi=hi,
                             frac_of_set=len(g) / len(d), contrib_to_MAE_K=len(g) * (g['abs'].mean() - d['abs'].mean()) / len(d),
                             set_MAE_K=d['abs'].mean(), set_n=len(d)))
R1 = pd.DataFrame(rows)
R1.to_csv(os.path.join(TAB, 'resid_by_mp_bin.csv'), index=False, encoding='utf-8-sig')

# (ii) 무거운 원자 수 x MP 구간
rows = []
for (how, src), d in sets.items():
    hb = pd.cut(d['heavy'], HV_EDGES, labels=HV_LAB, right=True)
    mb = pd.cut(d['y'], MP_EDGES, labels=MP_LAB, right=False)
    for hl in HV_LAB + ['all']:
        for ml in MP_LAB + ['all']:
            sel = (np.ones(len(d), bool) if hl == 'all' else (hb == hl).values) & (np.ones(len(d), bool) if ml == 'all' else (mb == ml).values)
            g = d[sel]
            rows.append(dict(split=how, source=src, heavy_bin=hl, mp_bin_K=ml, n=len(g),
                             MAE_K=g['abs'].mean() if len(g) else np.nan, mean_resid_K=g['resid'].mean() if len(g) else np.nan))
R2 = pd.DataFrame(rows)
R2.to_csv(os.path.join(TAB, 'mae_heavy_x_mp.csv'), index=False, encoding='utf-8-sig')

# (iii) 계열별
rows = []
for (how, src), d in sets.items():
    allmae = d['abs'].mean()
    for fam in FAMS:
        g, rest = d[d[fam]], d[~d[fam]]
        if len(g) == 0:
            rows.append(dict(split=how, source=src, family=fam, family_ko=FAM_KO[fam], n=0, set_n=len(d), set_MAE_K=allmae)); continue
        lo, hi = mc.boot_ci(g['resid'].values)
        alo, ahi = mc.boot_ci(g['abs'].values)
        rows.append(dict(split=how, source=src, family=fam, family_ko=FAM_KO[fam], n=len(g), frac_of_set=len(g) / len(d),
                         MAE_K=g['abs'].mean(), MAE_ci_lo=alo, MAE_ci_hi=ahi, MAE_rest_K=rest['abs'].mean(),
                         mean_resid_K=g['resid'].mean(), resid_ci_lo=lo, resid_ci_hi=hi, mean_y_K=g['y'].mean(),
                         contrib_to_MAE_K=len(g) * (g['abs'].mean() - allmae) / len(d),
                         share_of_abs_err=g['abs'].sum() / d['abs'].sum(), set_n=len(d), set_MAE_K=allmae))
R3 = pd.DataFrame(rows)
R3.to_csv(os.path.join(TAB, 'mae_by_family.csv'), index=False, encoding='utf-8-sig')

# (iv) OOF |오차| 상위 50 — 어느 분할의 test 에도 속하지 않는 분자만
test_any = set(splits['random']['test']) | set(splits['scaffold']['test'])
for how, fname in (('random', 'top50_oof_errors.csv'), ('scaffold', 'top50_oof_errors_scaffold.csv')):
    d = sets[(how, 'oof_train')].sort_values('abs', ascending=False).reset_index(drop=True)
    d['rank_unfiltered'] = np.arange(1, len(d) + 1)
    assert not set(d['row']) & set(splits[how]['test'])
    in_other_test = d['row'].isin(test_any)
    n_drop50 = int(in_other_test.iloc[:50].sum())
    t = d[~in_other_test].head(50).copy()
    t['smiles'] = smiles[t['row'].values]
    t['families'] = [';'.join(f for f in FAMS if r[f]) for _, r in t.iterrows()]
    t['split'] = how
    t['n_ring'] = FL['n_ring'].values[t['row'].values]
    t['sym_ratio'] = FL['sym_ratio'].values[t['row'].values].round(3)
    t = t.rename(columns={'abs': 'abs_err_K', 'resid': 'resid_pred_minus_y_K', 'y': 'y_K', 'pred': 'oof_pred_K'})
    t[['split', 'rank_unfiltered', 'row', 'smiles', 'y_K', 'oof_pred_K', 'resid_pred_minus_y_K', 'abs_err_K', 'heavy', 'n_ring',
       'sym_ratio', 'families']].to_csv(os.path.join(TAB, fname), index=False, encoding='utf-8-sig')
    print(f'[top50:{how}] unfiltered top-50 중 다른 분할 test 에 속해 제외한 수 = {n_drop50}; '
          f'목록의 |오차| 범위 {t["abs_err_K"].min():.1f}-{t["abs_err_K"].max():.1f} K; 과소예측(resid<0) {int((t["resid_pred_minus_y_K"] < 0).sum())}개', flush=True)

pd.set_option('display.width', 250)
print(R1[R1.bin_by == 'y'][['split', 'source', 'mp_bin_K', 'n', 'MAE_K', 'mean_resid_K', 'resid_ci_lo', 'resid_ci_hi']].round(2).to_string(index=False))
print(R3[R3.source == 'oof_train'][['split', 'family', 'n', 'MAE_K', 'MAE_rest_K', 'mean_resid_K', 'contrib_to_MAE_K']].round(2).to_string(index=False))
