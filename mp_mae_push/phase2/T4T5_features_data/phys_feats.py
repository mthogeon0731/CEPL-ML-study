"""phys_feats.py : F3 물리·결정 특징 (H05).

열 묶음(group) 이름은 GROUPS 에 정리한다. 2D 특징은 RDKit 그래프에서, 3D 특징은 ETKDGv3 conformer(MMFF 최적화)에서 계산한다.

회전 대칭수 σ
  sigma2d   : 무거운 원자 그래프(원자 번호·결합 차수 비교)의 자기동형 수(반사 포함 → 3D 회전 σ의 상한 대리값). 상한 MAX_AUT.
  sigma3d   : 자기동형 순열 중 conformer 좌표에서 '고유 회전(det=+1)'으로 실현되는 것의 수(Kabsch RMSD < TOL_A Å),
              conformer 여러 개 중 최댓값(sigma3d_max)과 MMFF 에너지 최저 conformer 값(sigma3d_low).
              무거운 원자만 본다(H 위치 무시 → 메틸 회전 등 내부 대칭은 세지 않음).
              직선형 분자(무거운 원자 2개 이상이 한 직선)는 Yalkowsky 관례의 유효값 10(비대칭·원뿔형) / 20(대칭·원통형),
              무거운 원자 1개는 σ = 1 로 둔다(데이터에 거의 없음).
유연성 τ (Yalkowsky): τ = SP3 + 0.5·SP2 + 0.5·RING − 1
  SP3/SP2 : 고리 밖, 무거운 원자 이웃 수 ≥ 2 인 sp3 / sp2 무거운 원자(헤테로 원자 포함). sp 원자는 세지 않음.
  RING    : 원자를 공유하는 고리끼리 묶은 고리계 수(tau). tau_linked 는 단일 결합으로 직접 이어진 고리계도 하나로 묶은 변형.
  tau_clip0 = max(tau, 0).
ΔSm = 50 − R ln σ + R ln φ, φ = c^τ  →  dSm_c2435, dSm_c285 (σ = sigma3d_max, τ = tau_clip0) [J/(K·mol)].
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors
from rdkit.Chem.rdchem import HybridizationType as HT

R_GAS = 8.314
MAX_AUT = 5000
TOL_A = 0.30

SMARTS = {
    'n_cooh': '[CX3](=O)[OX2H1,OX1-]',
    'n_amide': '[NX3][CX3](=[OX1])[#6]',
    'n_urea': '[NX3][CX3](=[OX1])[NX3]',
    'n_hydroxy': '[OX2H1][#6;!$([#6]=[O,S])]',
    'n_sulfonamide': '[SX4](=[OX1])(=[OX1])[NX3]',
}
_PATT = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
HAL = {9, 17, 35, 53}

COLS_2D = ['sigma2d_log', 'sym_class_ratio', 'n_aut_capped',
           'tau', 'tau_clip0', 'tau_linked', 'n_sp3_chain', 'n_sp2_chain', 'n_ringsys', 'n_ringsys_linked',
           'hbd', 'hba', 'hb_pairs', 'hb_minpair', 'n_cooh', 'n_amide', 'n_urea', 'n_hydroxy', 'n_sulfonamide',
           'ring_atom_frac', 'arom_atom_frac', 'rotb_frac', 'pi_max', 'pi_max_frac',
           'n_hal', 'hal_frac', 'n_brI']
COLS_3D = ['sigma3d_max_log', 'sigma3d_low_log', 'dSm_c2435', 'dSm_c285',
           'npr1', 'npr2', 'pbf', 'asphericity', 'eccentricity', 'spherocity', 'rgyr', 'conf_ok']
COLS = COLS_2D + COLS_3D

GROUPS = {
    'G_sym': ['sigma2d_log', 'sym_class_ratio', 'n_aut_capped', 'sigma3d_max_log', 'sigma3d_low_log'],
    'G_tau': ['tau', 'tau_clip0', 'tau_linked', 'n_sp3_chain', 'n_sp2_chain', 'n_ringsys', 'n_ringsys_linked'],
    'G_dSm': ['dSm_c2435', 'dSm_c285'],
    'G_hb': ['hbd', 'hba', 'hb_pairs', 'hb_minpair', 'n_cooh', 'n_amide', 'n_urea', 'n_hydroxy', 'n_sulfonamide'],
    'G_shape': ['npr1', 'npr2', 'pbf', 'asphericity', 'eccentricity', 'spherocity', 'rgyr', 'conf_ok'],
    'G_rigid': ['ring_atom_frac', 'arom_atom_frac', 'rotb_frac'],
    'G_pi': ['pi_max', 'pi_max_frac'],
    'G_hal': ['n_hal', 'hal_frac', 'n_brI'],
}
assert sorted(sum(GROUPS.values(), [])) == sorted(COLS)


def _automorphisms(m):
    return m.GetSubstructMatches(m, uniquify=False, useChirality=False, maxMatches=MAX_AUT)


def _ring_systems(m, linked=False):
    ri = m.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        s = set(ring)
        merged = [x for x in systems if x & s]
        for x in merged:
            s |= x
            systems.remove(x)
        systems.append(s)
    if linked and len(systems) > 1:
        changed = True
        while changed:
            changed = False
            for i in range(len(systems)):
                for j in range(i + 1, len(systems)):
                    a, b = systems[i], systems[j]
                    if any(m.GetBondBetweenAtoms(p, q) is not None for p in a for q in b):
                        systems[i] = a | b
                        systems.pop(j)
                        changed = True
                        break
                if changed:
                    break
    return len(systems)


def _pi_max(m):
    n = m.GetNumAtoms()
    parent = list(range(n))

    def f(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    atoms = set()
    for b in m.GetBonds():
        if b.GetIsConjugated() or b.GetIsAromatic():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            atoms.update((i, j))
            parent[f(i)] = f(j)
    if not atoms:
        return 0
    cnt = {}
    for a in atoms:
        r = f(a)
        cnt[r] = cnt.get(r, 0) + 1
    return max(cnt.values())


def feats_2d(m):
    """m: 무거운 원자 mol (H 없음)."""
    heavy = m.GetNumAtoms()
    aut = _automorphisms(m)
    n_aut = max(len(aut), 1)
    ranks = list(Chem.CanonicalRankAtoms(m, breakTies=False))
    sp3 = sp2 = 0
    for a in m.GetAtoms():
        if a.IsInRing() or a.GetDegree() < 2:
            continue
        h = a.GetHybridization()
        if h == HT.SP3:
            sp3 += 1
        elif h == HT.SP2:
            sp2 += 1
    rs = _ring_systems(m)
    rsl = _ring_systems(m, linked=True)
    tau = sp3 + 0.5 * sp2 + 0.5 * rs - 1
    hbd, hba = rdMolDescriptors.CalcNumHBD(m), rdMolDescriptors.CalcNumHBA(m)
    nb = m.GetNumBonds()
    ring_atoms = sum(a.IsInRing() for a in m.GetAtoms())
    arom = sum(a.GetIsAromatic() for a in m.GetAtoms())
    pim = _pi_max(m)
    nhal = sum(a.GetAtomicNum() in HAL for a in m.GetAtoms())
    nbri = sum(a.GetAtomicNum() in (35, 53) for a in m.GetAtoms())
    d = dict(sigma2d_log=np.log(n_aut), sym_class_ratio=len(set(ranks)) / heavy, n_aut_capped=float(len(aut) >= MAX_AUT),
             tau=tau, tau_clip0=max(tau, 0.0), tau_linked=sp3 + 0.5 * sp2 + 0.5 * rsl - 1,
             n_sp3_chain=sp3, n_sp2_chain=sp2, n_ringsys=rs, n_ringsys_linked=rsl,
             hbd=hbd, hba=hba, hb_pairs=hbd * hba, hb_minpair=min(hbd, hba),
             ring_atom_frac=ring_atoms / heavy, arom_atom_frac=arom / heavy,
             rotb_frac=(rdMolDescriptors.CalcNumRotatableBonds(m) / nb) if nb else 0.0,
             pi_max=pim, pi_max_frac=pim / heavy, n_hal=nhal, hal_frac=nhal / heavy, n_brI=nbri)
    for k, p in _PATT.items():
        d[k] = len(m.GetSubstructMatches(p))
    return d, aut


def _kabsch_rmsd_batch(X, perms):
    """X (n,3) 중심화 좌표, perms (k,n). 고유 회전만 허용한 최소 RMSD (k,)."""
    Y = X[perms]                       # (k,n,3)
    H = np.einsum('ni,knj->kij', X, Y)  # (k,3,3)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(np.einsum('kij,kjl->kil', U, Vt)))
    S = S.copy()
    S[:, 2] *= d
    e = 2 * (X ** 2).sum() - 2 * S.sum(axis=1)
    return np.sqrt(np.clip(e, 0, None) / len(X))


def sigma_from_coords(X, perms, tol=TOL_A):
    X = X - X.mean(axis=0)
    n = len(X)
    if n == 1:
        return 1
    # 직선형
    sv = np.linalg.svd(X, compute_uv=False)
    if sv[1] < 0.05 * max(sv[0], 1e-9) and sv[1] < 0.1:
        sym = any((np.asarray(p) != np.arange(n)).any() for p in perms)
        return 20 if sym else 10
    if len(perms) <= 1:
        return 1
    r = _kabsch_rmsd_batch(X, np.asarray(perms))
    return int((r < tol).sum())


def feats_3d(m, aut, n_conf=3, seed=0xf00d, max_iters=500):
    """m: 무거운 원자 mol. aut: feats_2d 의 자기동형 순열. 실패하면 NaN (conf_ok = 0)."""
    out = {c: np.nan for c in COLS_3D}
    out['conf_ok'] = 0.0
    mh = Chem.AddHs(m)
    ps = AllChem.ETKDGv3()
    ps.randomSeed = seed
    ps.numThreads = 1
    cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=n_conf, params=ps))
    if not cids:
        ps.useRandomCoords = True
        cids = list(AllChem.EmbedMultipleConfs(mh, numConfs=n_conf, params=ps))
    if not cids:
        return out
    energies = [0.0] * len(cids)
    try:
        if AllChem.MMFFHasAllMoleculeParams(mh):
            res = AllChem.MMFFOptimizeMoleculeConfs(mh, numThreads=1, maxIters=max_iters)
        else:
            res = AllChem.UFFOptimizeMoleculeConfs(mh, numThreads=1, maxIters=max_iters)
        energies = [e for _, e in res]
    except Exception:
        pass
    # AddHs 는 m 의 원자 번호를 유지하고 H를 뒤에 붙인다 → 앞 m.GetNumAtoms() 개가 m 의 원자(SMILES의 명시 [H] 포함)
    n_m = m.GetNumAtoms()
    sig = []
    for cid in cids:
        X = mh.GetConformer(cid).GetPositions()[:n_m]
        sig.append(sigma_from_coords(X, aut))
    low = int(np.argmin(energies))
    cid_low = cids[low]
    out['sigma3d_max_log'] = float(np.log(max(sig)))
    out['sigma3d_low_log'] = float(np.log(sig[low]))
    try:
        out['npr1'] = Descriptors3D.NPR1(mh, confId=cid_low)
        out['npr2'] = Descriptors3D.NPR2(mh, confId=cid_low)
        out['pbf'] = rdMolDescriptors.CalcPBF(mh, confId=cid_low)
        out['asphericity'] = Descriptors3D.Asphericity(mh, confId=cid_low)
        out['eccentricity'] = Descriptors3D.Eccentricity(mh, confId=cid_low)
        out['spherocity'] = Descriptors3D.SpherocityIndex(mh, confId=cid_low)
        out['rgyr'] = Descriptors3D.RadiusOfGyration(mh, confId=cid_low)
    except Exception:
        pass
    out['conf_ok'] = 1.0
    return out


def dsm_cols(sig3_log, tau_clip0):
    return dict(dSm_c2435=50 - R_GAS * sig3_log + R_GAS * np.log(2.435) * tau_clip0,
                dSm_c285=50 - R_GAS * sig3_log + R_GAS * np.log(2.85) * tau_clip0)


def featurize_smiles(smi, n_conf=3):
    m = Chem.MolFromSmiles(smi)
    if m is None or m.GetNumAtoms() == 0:
        return {c: np.nan for c in COLS}
    d2, aut = feats_2d(m)
    d3 = feats_3d(m, aut, n_conf=n_conf)
    sig = d3['sigma3d_max_log']
    if np.isfinite(sig):
        d3.update(dsm_cols(sig, d2['tau_clip0']))
    return {**d2, **d3}
