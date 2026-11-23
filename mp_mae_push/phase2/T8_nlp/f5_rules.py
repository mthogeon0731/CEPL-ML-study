"""f5_rules.py : LLM(M2)이 제안한 녹는점 구조 규칙을 RDKit 2D 특징으로 구현 (F5 묶음).

규칙 출처: prompts/m2_response_literature.txt (L, 39개), prompts/m2_response_data200.txt (D, 25개).
각 열이 어느 규칙에서 왔는지는 RULE_MAP, 열이 어느 묶음(빼기 실험 단위)에 속하는지는 GROUPS 에 있다.
구조(SMILES)만 쓰고 측정 물성은 쓰지 않는다.
"""
import math

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors as rd

RDLogger.DisableLog('rdApp.*')

SMARTS = {
    'intra_1': '[OX2H,NX3H1,NX3H2]-c:c-[CX3]=O',
    'intra_2': '[OX2H,NX3H1,NX3H2]-c:c-[N+](=O)[O-]',
    'intra_3': '[OX2H]-c:c-[CX3]=[NX2]',
    'intra_4': '[OX2H]-[CX3]=[CX3]-[CX3]=O',
    'amide_sec': '[CX3](=O)[NX3H1][#6]',
    'amide_prim': '[CX3](=O)[NX3H2]',
    'amide_tert': '[CX3](=O)[NX3H0]([#6])[#6]',
    'urea': '[NX3;H1,H2]C(=O)[NX3;H1,H2]',
    'imide_ring': '[#6;R](=[O,S])[NX3H1;R][#6;R](=[O,S])',
    'lactam_NH': '[NX3H1;R]C(=O)[NX3;R]',
    'cooh': '[CX3](=O)[OX2H1]',
    'cooh_ar': 'c[CX3](=O)[OX2H1]',
    'basic_amine': '[NX3;H2,H1,H0;!$(NC=[O,S]);!$(N-a);!$(N-[SX4]);!$(N-N);!$(N-O);!$(N#*);!$(N=*)]',
    'so3h': '[SX4](=O)(=O)[OX2H1]',
    'sulfate': '[OX2]S(=O)(=O)[OX2]',
    'quat_N': '[NX4+;!$([N+][O-])]',
    'nitro_ar': 'c[N+](=O)[O-]',
    'pushpull_p': '[NX3;H1,H2,OX2H]c1ccc([N+](=O)[O-])cc1',
    'pushpull_o': '[NX3;H1,H2]c:c[N+](=O)[O-]',
    'nitroso': '[#6][NX2]=O',
    'nitrile_ar': 'cC#N',
    'dicyano_p': 'N#Cc1ccc(C#N)cc1',
    'XB_ar': '[I,Br]-c',
    'XB_acc': '[nX2,$(O=*),$(N#C)]',
    'CF3': '[CX4](F)(F)F',
    'C6F5': 'c1(F)c(F)c(F)c(F)c(F)c1',
    'nH': '[nH]',
    'n_acc': '[nX2]',
    'donor_OHNH': '[OX2H,NX3H2,NX3H1]',
    'pyridone': 'O=c1cccc[nH]1',
    'OH_aliph': '[OX2H][CX4]',
    'OH_ring': '[OX2H][C;R]',
    'OH_phenol': 'c[OX2H]',
    'ester': '[#6][CX3](=O)[OX2][#6]',
    'ether': '[OD2]([#6])[#6]',
    'ketone': '[#6][CX3](=O)[#6]',
    'SO2NH2': '[SX4](=O)(=O)[NX3H2]',
    'SO2NH': '[SX4](=O)(=O)[NX3H1]',
    'sulfone': '[#6][SX4](=O)(=O)[#6]',
    'azo': 'c[NX2]=[NX2]c',
    'stilbene': 'c[CX3]=[CX3]c',
    'tolane': 'cC#Cc',
    'imine_ar': 'c[CX3]=[NX2]c',
    'flex_CH2': 'c[CH2]c',
    'flex_O': 'c[OX2]c',
    'flex_CC': 'c[CH2][CH2]c',
    'CH3': '[CH3X4]',
    'iso_term': '[CH1X4;!R]([CH3])[CH3]',
    'tBu': '[CX4;!R]([CH3])([CH3])[CH3]',
    'quatC': '[CX4;D4]',
    'tertC': '[CX4;D3]',
    'irreg_branch': '[CH1X4;!R]([CH3])[CH2]',
    'boronic': '[BX3]([OX2H])[OX2H]',
    'hydrazone_dc': 'c[NH]N=C(C=O)',
    'hydrazide': '[CX3](=O)[NH][NH2]',
    'acyl_hal': '[CX3](=O)[Cl,Br,I]',
    'SO2Cl': 'S(=O)(=O)Cl',
    'NCO': 'N=C=O',
    'thio_NH': '[NX3;H1,H2]C(=S)',
    'anilide_26': 'c1(C)cccc(C)c1N(C)C=O',
    'ester_link': 'cC(=O)O[c,$(C=O)]',
    'ortho_bridge': '[c;$(c-[!#1;!c])]:c-[CX4,CX3,NX2,NX3,OX2,SX2]-c',
}
PAT = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
for k, v in PAT.items():
    assert v is not None, k


def _n(m, k):
    return len(m.GetSubstructMatches(PAT[k], maxMatches=10000))


def _has(m, k):
    return int(m.HasSubstructMatch(PAT[k]))


def _ring_systems(m, rings):
    """bond 를 공유하는 고리를 합쳐 고리계(atom set, 고리 수) 목록."""
    systems = []
    for r in rings:
        rs = set(r)
        merged = [s for s in systems if len(s[0] & rs) >= 2]
        for s in merged:
            systems.remove(s)
        atoms = rs.union(*[s[0] for s in merged]) if merged else rs
        systems.append((atoms, 1 + sum(s[1] for s in merged)))
    return systems


def _benzene_patterns(m, ri, ranks):
    """고립 벤젠 고리(다른 고리와 원자 공유 없음)의 치환 배치."""
    out = dict(para2=0, meta2=0, ortho2=0, para2_ident=0, tri135=0, tri135_ident=0, tetra1245=0, hexa=0)
    for ring in ri.AtomRings():
        if len(ring) != 6:
            continue
        if not all(m.GetAtomWithIdx(a).GetSymbol() == 'C' and m.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
            continue
        if any(ri.NumAtomRings(a) > 1 for a in ring):
            continue
        rset = set(ring)
        subs = {}
        for pos, a in enumerate(ring):
            nb = [n.GetIdx() for n in m.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() not in rset]
            if nb:
                subs[pos] = nb[0]
        k = len(subs)
        P = sorted(subs)
        if k == 2:
            d = (P[1] - P[0]) % 6
            d = min(d, 6 - d)
            if d == 3:
                out['para2'] += 1
                out['para2_ident'] += int(ranks[subs[P[0]]] == ranks[subs[P[1]]])
            elif d == 2:
                out['meta2'] += 1
            else:
                out['ortho2'] += 1
        elif k == 3:
            if all(((P[i + 1] - P[i]) % 6) == 2 for i in range(2)):
                out['tri135'] += 1
                out['tri135_ident'] += int(len({ranks[subs[p]] for p in P}) == 1)
        elif k == 4:
            free = sorted(set(range(6)) - set(P))
            d = (free[1] - free[0]) % 6
            if min(d, 6 - d) == 3:
                out['tetra1245'] += 1
        elif k == 6:
            out['hexa'] += 1
    return out


def featurize(m):
    f = {}
    heavy = m.GetNumHeavyAtoms()
    hv = max(heavy, 1)
    ri = m.GetRingInfo()
    rings = ri.AtomRings()
    ranks = list(Chem.CanonicalRankAtoms(m, breakTies=False))
    # ---- G1 대칭 ----
    nuniq = len(set(ranks))
    f['sym_heavy_per_class'] = heavy / max(nuniq, 1)
    f['sym_class_ratio'] = nuniq / hv
    f['sym_score'] = 1 - nuniq / hv
    try:
        n_aut = len(m.GetSubstructMatches(m, uniquify=False, useChirality=False, maxMatches=5000))
    except Exception:
        n_aut = 1
    n_aut = max(n_aut, 1)
    f['sym_n_automorph'] = n_aut
    f['sym_log_automorph'] = math.log(n_aut)
    bz = _benzene_patterns(m, ri, ranks)
    for k, v in bz.items():
        f['bz_' + k] = v
    # ---- G2 유연성·모양·사슬 ----
    sp3 = sp2 = 0
    for a in m.GetAtoms():
        if a.IsInRing() or a.GetDegree() <= 1:
            continue
        h = a.GetHybridization()
        if h == Chem.HybridizationType.SP3:
            sp3 += 1
        elif h == Chem.HybridizationType.SP2:
            sp2 += 1
    systems = _ring_systems(m, rings)
    tau = max(0.0, sp3 + 0.5 * sp2 + 0.5 * len(systems) - 1)
    f['yalk_tau'] = tau
    f['yalk_ln_phi'] = tau * math.log(2.435)
    f['yalk_dSm'] = 50 - 8.314 * math.log(n_aut) + 8.314 * tau * math.log(2.435)
    ch2 = [a.GetIdx() for a in m.GetAtoms() if a.GetSymbol() == 'C' and not a.IsInRing()
           and a.GetHybridization() == Chem.HybridizationType.SP3 and a.GetTotalNumHs() == 2]
    L = 0
    if ch2:
        s = set(ch2)
        seen = set()
        for a0 in ch2:
            if a0 in seen:
                continue
            comp, stack = 0, [a0]
            seen.add(a0)
            while stack:
                a = stack.pop()
                comp += 1
                for n in m.GetAtomWithIdx(a).GetNeighbors():
                    j = n.GetIdx()
                    if j in s and j not in seen:
                        seen.add(j)
                        stack.append(j)
            L = max(L, comp)
    f['chain_CH2_len'] = L
    f['chain_sat'] = L / (L + 5.0)
    f['chain_even'] = int(L >= 2 and L % 2 == 0)
    f['chain_even_x_donor'] = f['chain_even'] * int(rd.CalcNumHBD(m) > 0)
    f['chain_CH2_frac'] = len(ch2) / hv
    f['iso_termini'] = _n(m, 'iso_term')
    f['tBu'] = _n(m, 'tBu')
    f['quatC'] = _n(m, 'quatC')
    f['tertC'] = _n(m, 'tertC')
    f['quat_tert_ratio'] = f['quatC'] / (f['tertC'] + 1.0)
    f['irreg_branch'] = _n(m, 'irreg_branch')
    f['bridgehead'] = rd.CalcNumBridgeheadAtoms(m)
    f['spiro'] = rd.CalcNumSpiroAtoms(m)
    nrot = rd.CalcNumRotatableBonds(m)
    f['cage_compact'] = int(f['bridgehead'] > 0 and nrot <= 1)
    D = Chem.GetDistanceMatrix(m) if heavy > 1 else np.zeros((1, 1))
    W = D.sum() / 2.0
    f['wiener_norm'] = W / ((heavy ** 3 - heavy) / 6.0) if heavy > 2 else 1.0
    f['diam_per_atom'] = float(D.max()) / hv
    f['methyl_frac'] = _n(m, 'CH3') / hv
    f['flex_link_CH2'] = _n(m, 'flex_CH2')
    f['flex_link_O'] = _n(m, 'flex_O')
    f['flex_link_CC'] = _n(m, 'flex_CC')
    f['ester_link'] = _n(m, 'ester_link')
    nhbd = rd.CalcNumHBD(m)
    f['small_nopolar'] = int(heavy <= 7 and nhbd == 0)
    # ---- G3 수소결합 모티프 ----
    intra_donors = set()
    for k in ('intra_1', 'intra_2', 'intra_3', 'intra_4'):
        for mt in m.GetSubstructMatches(PAT[k]):
            intra_donors.add(mt[0])
    f['hb_intra'] = len(intra_donors)
    f['hb_free_donor'] = max(0, nhbd - len(intra_donors))
    f['amide_sec'] = _n(m, 'amide_sec')
    f['amide_sec_extra'] = max(0, f['amide_sec'] - 1)
    f['amide_prim'] = _n(m, 'amide_prim')
    f['amide_tert'] = _n(m, 'amide_tert')
    f['urea'] = _n(m, 'urea')
    f['imide_ring'] = _n(m, 'imide_ring')
    f['lactam_NH'] = _n(m, 'lactam_NH')
    f['cooh'] = _n(m, 'cooh')
    f['cooh_ge2'] = int(f['cooh'] >= 2)
    f['cooh_ar'] = _n(m, 'cooh_ar')
    f['nH_arom'] = _n(m, 'nH')
    f['nacc_with_donor'] = int(_has(m, 'n_acc') and _has(m, 'donor_OHNH'))
    f['pyridone'] = _n(m, 'pyridone')
    f['OH_aliph'] = _n(m, 'OH_aliph')
    f['OH_ring'] = _n(m, 'OH_ring')
    nb = max(m.GetNumBonds(), 1)
    f['polyol_rigid'] = f['OH_aliph'] * (1 - nrot / nb)
    f['phenol_extra'] = max(0, _n(m, 'OH_phenol') - 1)
    f['SO2NH2'] = _n(m, 'SO2NH2')
    f['SO2NH'] = _n(m, 'SO2NH')
    f['sulfone'] = _n(m, 'sulfone')
    f['thio_NH'] = _n(m, 'thio_NH')
    f['boronic'] = _n(m, 'boronic')
    f['hydrazide'] = _n(m, 'hydrazide')
    f['hydrazone_dc'] = _n(m, 'hydrazone_dc')
    # ---- G4 평면성·접합·강직 연결 ----
    conj_adj = {}
    for b in m.GetBonds():
        if b.GetIsConjugated() or b.GetIsAromatic():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            conj_adj.setdefault(i, set()).add(j)
            conj_adj.setdefault(j, set()).add(i)
    best, seen = 0, set()
    for a0 in conj_adj:
        if a0 in seen:
            continue
        comp, stack = 0, [a0]
        seen.add(a0)
        while stack:
            a = stack.pop()
            comp += 1
            for j in conj_adj[a]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        best = max(best, comp)
    f['conj_largest_frac'] = best / hv
    f['conj_largest'] = best
    f['sp2arom_frac'] = sum(1 for a in m.GetAtoms() if a.GetIsAromatic()
                            or a.GetHybridization() == Chem.HybridizationType.SP2) / hv
    arom_rings = [r for r in rings if all(m.GetAtomWithIdx(a).GetIsAromatic() for a in r)]
    asys = _ring_systems(m, arom_rings)
    f['arom_sys_max_rings'] = max([s[1] for s in asys], default=0)
    f['arom_sys_max_atoms'] = max([len(s[0]) for s in asys], default=0)
    nbi = nbi_o = 0
    for b in m.GetBonds():
        if b.IsInRing() or b.GetBondType() != Chem.BondType.SINGLE:
            continue
        a1, a2 = b.GetBeginAtom(), b.GetEndAtom()
        if not (a1.GetIsAromatic() and a2.GetIsAromatic()):
            continue
        nbi += 1
        orth = 0
        for a in (a1, a2):
            for n in a.GetNeighbors():
                if n.GetIdx() in (a1.GetIdx(), a2.GetIdx()) or not n.GetIsAromatic():
                    continue
                if n.GetDegree() >= 3:
                    orth = 1
        nbi_o += orth
    f['biaryl'] = nbi
    f['biaryl_ortho'] = nbi_o
    f['link_azo'] = _n(m, 'azo')
    f['link_stilbene'] = _n(m, 'stilbene')
    f['link_tolane'] = _n(m, 'tolane')
    f['link_imine'] = _n(m, 'imine_ar')
    f['ortho_bridge'] = _n(m, 'ortho_bridge')
    fused_sp3 = 0
    for b in m.GetBonds():
        if ri.NumBondRings(b.GetIdx()) >= 2:
            if (b.GetBeginAtom().GetHybridization() == Chem.HybridizationType.SP3
                    and b.GetEndAtom().GetHybridization() == Chem.HybridizationType.SP3):
                fused_sp3 += 1
    f['fused_sp3_bonds'] = fused_sp3
    f['ring_sys_max_rings'] = max([s[1] for s in systems], default=0)
    # ---- G5 극성·전하·할로젠 ----
    charges = [a.GetFormalCharge() for a in m.GetAtoms()]
    nitro_pairs = _n(m, 'nitro_ar') + len(m.GetSubstructMatches(Chem.MolFromSmarts('[N+](=O)[O-]')))
    has_charge_nonnitro = int(sum(abs(c) for c in charges) > 2 * len(m.GetSubstructMatches(Chem.MolFromSmarts('[N+](=O)[O-]'))))
    acid = _has(m, 'cooh') or _has(m, 'so3h')
    f['zwitter_neutral'] = int((has_charge_nonnitro and sum(charges) == 0) or (acid and _has(m, 'basic_amine')))
    f['ionic_multi'] = int(len(Chem.GetMolFrags(m)) > 1 and any(charges))
    f['quat_N'] = _n(m, 'quat_N')
    f['so3h_sulfate'] = _n(m, 'so3h') + _n(m, 'sulfate')
    f['nitro_ar'] = _n(m, 'nitro_ar')
    f['nitro_pushpull'] = _n(m, 'pushpull_p') + _n(m, 'pushpull_o')
    f['nitroso'] = _n(m, 'nitroso')
    f['nitrile_ar'] = _n(m, 'nitrile_ar')
    f['dicyano_p'] = _n(m, 'dicyano_p')
    f['XB_donor'] = _n(m, 'XB_ar')
    f['XB_pair'] = f['XB_donor'] * _has(m, 'XB_acc')
    nF = sum(1 for a in m.GetAtoms() if a.GetSymbol() == 'F')
    f['F_frac'] = nF / hv
    f['CF3'] = _n(m, 'CF3')
    f['arene_perfluoroarene'] = int(_has(m, 'C6F5') and any(
        all(m.GetAtomWithIdx(a).GetIsAromatic() for a in r) and not any(
            n.GetSymbol() == 'F' for a in r for n in m.GetAtomWithIdx(a).GetNeighbors()) for r in rings))
    f['ester_ether_only'] = int(nhbd == 0 and (_has(m, 'ester') or _has(m, 'ether') or _has(m, 'ketone')))
    f['ester_ether_nodonor_n'] = (_n(m, 'ester') + _n(m, 'ether')) * int(nhbd == 0)
    f['reactive_lowmp'] = _n(m, 'acyl_hal') + _n(m, 'SO2Cl') + _n(m, 'NCO')
    f['anilide_26'] = _n(m, 'anilide_26')
    # ---- G6 입체 ----
    ce = cz = 0
    for b in m.GetBonds():
        st = b.GetStereo()
        if st in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS):
            ce += 1
        elif st in (Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOCIS):
            cz += 1
    f['db_trans'] = ce
    f['db_cis'] = cz
    cc = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
    nas = sum(1 for _, t in cc if t != '?')
    f['stereo_centers'] = len(cc)
    f['stereo_unassigned'] = len(cc) - nas
    f['stereo_assigned'] = nas
    return f


GROUPS = {
    'G1_symmetry': ['sym_heavy_per_class', 'sym_class_ratio', 'sym_score', 'sym_n_automorph', 'sym_log_automorph',
                    'bz_para2', 'bz_meta2', 'bz_ortho2', 'bz_para2_ident', 'bz_tri135', 'bz_tri135_ident',
                    'bz_tetra1245', 'bz_hexa'],
    'G2_flex_shape': ['yalk_tau', 'yalk_ln_phi', 'yalk_dSm', 'chain_CH2_len', 'chain_sat', 'chain_even',
                      'chain_even_x_donor', 'chain_CH2_frac', 'iso_termini', 'tBu', 'quatC', 'tertC',
                      'quat_tert_ratio', 'irreg_branch', 'bridgehead', 'spiro', 'cage_compact', 'wiener_norm',
                      'diam_per_atom', 'methyl_frac', 'flex_link_CH2', 'flex_link_O', 'flex_link_CC', 'ester_link',
                      'small_nopolar'],
    'G3_hbond': ['hb_intra', 'hb_free_donor', 'amide_sec', 'amide_sec_extra', 'amide_prim', 'amide_tert', 'urea',
                 'imide_ring', 'lactam_NH', 'cooh', 'cooh_ge2', 'cooh_ar', 'nH_arom', 'nacc_with_donor', 'pyridone',
                 'OH_aliph', 'OH_ring', 'polyol_rigid', 'phenol_extra', 'SO2NH2', 'SO2NH', 'sulfone', 'thio_NH',
                 'boronic', 'hydrazide', 'hydrazone_dc'],
    'G4_planar_conj': ['conj_largest_frac', 'conj_largest', 'sp2arom_frac', 'arom_sys_max_rings',
                       'arom_sys_max_atoms', 'biaryl', 'biaryl_ortho', 'link_azo', 'link_stilbene', 'link_tolane',
                       'link_imine', 'ortho_bridge', 'fused_sp3_bonds', 'ring_sys_max_rings'],
    'G5_polar_charge_hal': ['zwitter_neutral', 'ionic_multi', 'quat_N', 'so3h_sulfate', 'nitro_ar', 'nitro_pushpull',
                            'nitroso', 'nitrile_ar', 'dicyano_p', 'XB_donor', 'XB_pair', 'F_frac', 'CF3',
                            'arene_perfluoroarene', 'ester_ether_only', 'ester_ether_nodonor_n', 'reactive_lowmp',
                            'anilide_26'],
    'G6_stereo': ['db_trans', 'db_cis', 'stereo_centers', 'stereo_unassigned', 'stereo_assigned'],
}
COLS = [c for g in GROUPS.values() for c in g]
