"""2단계 특징 계산: O3 입체 인식 특징(소수 열) + H03 Morgan count 지문(반경 2, 2,048).
행 순서 = mp.csv 행 순서(분할 파일 row). 캐시: mp_mae_push/cache/X_stereo_o3.npy, X_morgan_count_r2_2048.npy"""
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import t1lib as L
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdCIPLabeler
RDLogger.DisableLog('rdApp.*')

CACHE = os.path.join(L.WS, 'mp_mae_push', 'cache')
_, s = L.split('random')
s = s.sort_values('row')
assert (s['row'].to_numpy() == np.arange(len(s))).all()
smiles = s['smiles'].tolist()

# mp.csv smiles 와 같은지 확인
mp = pd.read_csv(os.path.join(L.mc.WEEK2, 'data', 'mp.csv'))
print('mp.csv cols', list(mp.columns), 'n', len(mp), flush=True)

O3_COLS = ['st_n_at_spec', 'st_n_db_spec', 'st_n_slash', 'st_n_pot_tet', 'st_n_pot_db', 'st_n_unspec_tet',
           'st_n_unspec_db', 'st_frac_spec', 'st_any_pot', 'st_n_R', 'st_n_S', 'st_n_E', 'st_n_Z',
           'st_morgan_chiral_extra', 'st_morgan_chiral_l1']


def o3_row(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return [np.nan] * len(O3_COLS)
    n_at = sum(1 for a in m.GetAtoms() if a.GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW))
    n_db = sum(1 for b in m.GetBonds() if b.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                                                             Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS))
    n_slash = smi.count('/') + smi.count('\\')
    pot_tet = pot_db = un_tet = un_db = 0
    try:
        for si in Chem.FindPotentialStereo(m):
            spec = si.specified == Chem.StereoSpecified.Specified
            if si.type == Chem.StereoType.Atom_Tetrahedral:
                pot_tet += 1; un_tet += (not spec)
            elif si.type == Chem.StereoType.Bond_Double:
                pot_db += 1; un_db += (not spec)
    except Exception:
        pass
    pot = pot_tet + pot_db
    frac = (pot - un_tet - un_db) / pot if pot else 1.0
    nR = nS = nE = nZ = 0
    try:
        rdCIPLabeler.AssignCIPLabels(m)
        for a in m.GetAtoms():
            if a.HasProp('_CIPCode'):
                c = a.GetProp('_CIPCode'); nR += c in ('R', 'r'); nS += c in ('S', 's')
        for b in m.GetBonds():
            if b.HasProp('_CIPCode'):
                c = b.GetProp('_CIPCode'); nE += c == 'E'; nZ += c == 'Z'
    except Exception:
        pass
    fc = GEN_CHI.GetSparseCountFingerprint(m).GetNonzeroElements()
    fa = GEN_ACH.GetSparseCountFingerprint(m).GetNonzeroElements()
    extra = len(fc) - len(fa)
    # 같은 원자 환경이 키랄성으로 갈라진 정도: 개수 분포 차이 (정렬한 count 벡터의 L1)
    va, vc = sorted(fa.values(), reverse=True), sorted(fc.values(), reverse=True)
    n = max(len(va), len(vc)); va += [0] * (n - len(va)); vc += [0] * (n - len(vc))
    l1 = int(np.abs(np.array(va) - np.array(vc)).sum())
    return [n_at, n_db, n_slash, pot_tet, pot_db, un_tet, un_db, frac, int(pot > 0), nR, nS, nE, nZ, extra, l1]


GEN_CHI = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=True)
GEN_ACH = rdFingerprintGenerator.GetMorganGenerator(radius=2, includeChirality=False)
GEN_CNT = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

p_o3 = os.path.join(CACHE, 'X_stereo_o3.npy')
if not os.path.exists(p_o3):
    with L.mc.CostMeter('특징') as c:
        Xo = np.array([o3_row(sm) for sm in smiles], dtype=np.float32)
    np.save(p_o3, Xo)
    pd.DataFrame(Xo, columns=O3_COLS).assign(row=np.arange(len(Xo))).to_csv(os.path.join(L.TDIR, 'features_O3_stereo.csv'), index=False)
    L.ledger(dict(run_id='s2_feat_O3', model='RDKit', features='O3 입체 15열', data_variant='전체 19,945분자', objective='',
                  split='-', params={'rdkit': __import__('rdkit').__version__, 'threads': 1}, n_train=len(Xo),
                  infer_s_per_1k=round(c.wall_s / len(Xo) * 1000, 4), notes='특징 계산 (단일 스레드)', **c.as_row()))
    print('O3', Xo.shape, 'wall', round(c.wall_s, 1), 'nan', int(np.isnan(Xo).sum()), flush=True)

p_mc = os.path.join(CACHE, 'X_morgan_count_r2_2048.npy')
if not os.path.exists(p_mc):
    with L.mc.CostMeter('특징') as c:
        Xm = np.zeros((len(smiles), 2048), dtype=np.uint8)
        nclip = 0
        for i, sm in enumerate(smiles):
            m = Chem.MolFromSmiles(sm)
            if m is None:
                continue
            v = GEN_CNT.GetCountFingerprintAsNumPy(m)
            nclip += int((v > 255).sum())
            Xm[i] = np.minimum(v, 255).astype(np.uint8)
    np.save(p_mc, Xm)
    L.ledger(dict(run_id='s2_feat_morgan', model='RDKit', features='F1 Morgan count r2 2048', data_variant='전체 19,945분자',
                  objective='', split='-', params={'radius': 2, 'fpSize': 2048, 'dtype': 'uint8 (255 상한)', 'threads': 1},
                  n_train=len(Xm), infer_s_per_1k=round(c.wall_s / len(Xm) * 1000, 4),
                  notes=f'특징 계산 (단일 스레드); 255 초과로 잘린 칸 {nclip}', **c.as_row()))
    print('Morgan', Xm.shape, 'wall', round(c.wall_s, 1), 'clipped', nclip, 'nonzero cols', int((Xm.sum(0) > 0).sum()), flush=True)
