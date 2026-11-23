"""n2_template.py : 구조(2D 그래프)에서 계산한 정보만으로 영어 서술 문장을 만든다 (N2).

물성값(측정·추정 logP 포함), 용도, 상온 상태는 쓰지 않는다. 분자량은 분자식에서 계산되는 값이라 포함한다.
1단계 오차 분해가 지목한 항목(2D 대칭, 다환, 강직성, 수소결합 주개 ≥ 3)을 명시적으로 넣는다.
"""
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors as rd

import f5_rules as f5

FG = [  # (SMARTS, 단수 이름, 복수 이름)
    ('[CX3](=O)[OX2H1]', 'carboxylic acid group', 'carboxylic acid groups'),
    ('[#6][CX3](=O)[OX2][#6]', 'ester group', 'ester groups'),
    ('[CX3](=O)[NX3H2]', 'primary amide', 'primary amides'),
    ('[CX3](=O)[NX3H1]', 'secondary amide', 'secondary amides'),
    ('[CX3](=O)[NX3H0]', 'tertiary amide', 'tertiary amides'),
    ('[NX3;H1,H2]C(=O)[NX3]', 'urea unit', 'urea units'),
    ('[CX3](=O)[NX3][CX3](=O)', 'imide unit', 'imide units'),
    ('[#6][CX3](=O)[#6]', 'ketone', 'ketones'),
    ('[CX3H1](=O)[#6]', 'aldehyde', 'aldehydes'),
    ('[OX2H][CX4]', 'aliphatic hydroxyl group', 'aliphatic hydroxyl groups'),
    ('[OX2H]c', 'phenolic hydroxyl group', 'phenolic hydroxyl groups'),
    ('[OD2]([#6;!$(C=O)])[#6;!$(C=O)]', 'ether linkage', 'ether linkages'),
    ('[NX3H2][CX4]', 'primary aliphatic amine', 'primary aliphatic amines'),
    ('[NX3H2]c', 'primary aromatic amine', 'primary aromatic amines'),
    ('[NX3H1;!$(NC=[O,S]);!$(NS(=O)=O)]([#6])[#6]', 'secondary amine', 'secondary amines'),
    ('[NX3H0;!$(NC=[O,S]);!$(NS(=O)=O);!$(N-[N,O]);!$([N+])]([#6])([#6])[#6]', 'tertiary amine', 'tertiary amines'),
    ('[N+](=O)[O-]', 'nitro group', 'nitro groups'),
    ('C#N', 'nitrile group', 'nitrile groups'),
    ('[SX4](=O)(=O)[NX3]', 'sulfonamide group', 'sulfonamide groups'),
    ('[#6][SX4](=O)(=O)[#6]', 'sulfone group', 'sulfone groups'),
    ('[SX4](=O)(=O)[OX2H]', 'sulfonic acid group', 'sulfonic acid groups'),
    ('[SX2H]', 'thiol group', 'thiol groups'),
    ('[#6][SX2][#6]', 'thioether linkage', 'thioether linkages'),
    ('C(=S)', 'thiocarbonyl group', 'thiocarbonyl groups'),
    ('[nX2]', 'pyridine-type aromatic nitrogen', 'pyridine-type aromatic nitrogens'),
    ('[nH]', 'pyrrole-type aromatic N-H', 'pyrrole-type aromatic N-H groups'),
    ('[NX2]=[NX2]', 'azo linkage', 'azo linkages'),
    ('[CX3]=[NX2;!R]', 'imine (C=N) bond', 'imine (C=N) bonds'),
    ('[CX3;!a]=[CX3;!a]', 'non-aromatic C=C double bond', 'non-aromatic C=C double bonds'),
    ('C#C', 'C≡C triple bond', 'C≡C triple bonds'),
    ('[CX4](F)(F)F', 'trifluoromethyl group', 'trifluoromethyl groups'),
    ('P', 'phosphorus atom', 'phosphorus atoms'),
    ('[BX3]', 'boron atom', 'boron atoms'),
]
FGP = [(Chem.MolFromSmarts(s), a, b) for s, a, b in FG]
NUM = ['no', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']


def w(n):
    return NUM[n] if 0 <= n < len(NUM) else str(n)


def plural(n, one, many):
    return f'{w(n)} {one if n == 1 else many}'


def describe(m):
    f = f5.featurize(m)
    heavy = m.GetNumHeavyAtoms()
    formula = rd.CalcMolFormula(m)
    mw = Descriptors.MolWt(m)
    ri = m.GetRingInfo()
    rings = ri.AtomRings()
    nar = rd.CalcNumAromaticRings(m)
    nali = len(rings) - nar
    systems = f5._ring_systems(m, rings)
    fused = [s for s in systems if s[1] > 1]
    S = [f'A molecule with formula {formula}, {heavy} heavy atoms and molecular weight {mw:.1f}.']
    if not rings:
        S.append('It is acyclic (no rings).')
    else:
        t = f'It has {plural(len(rings), "ring", "rings")} ({w(nar)} aromatic, {w(nali)} non-aromatic)'
        if fused:
            big = max(s[1] for s in fused)
            t += f', with {plural(len(fused), "fused ring system", "fused ring systems")} (largest has {w(big)} rings)'
        else:
            t += ', none fused'
        if f['bridgehead']:
            t += f', {plural(f["bridgehead"], "bridgehead atom", "bridgehead atoms")}'
        if f['spiro']:
            t += f', {plural(f["spiro"], "spiro atom", "spiro atoms")}'
        S.append(t + '.')
        nb = rd.CalcNumAromaticCarbocycles(m)
        nh = rd.CalcNumAromaticHeterocycles(m)
        if nar:
            S.append(f'Aromatic rings: {w(nb)} carbocyclic (benzene-type) and {w(nh)} heterocyclic.')
        pat = []
        for k, txt in [('bz_para2', 'para-disubstituted'), ('bz_meta2', 'meta-disubstituted'),
                       ('bz_ortho2', 'ortho-disubstituted'), ('bz_tri135', '1,3,5-trisubstituted'),
                       ('bz_tetra1245', '1,2,4,5-tetrasubstituted'), ('bz_hexa', 'hexasubstituted')]:
            if f[k]:
                pat.append(f'{plural(int(f[k]), txt + " benzene ring", txt + " benzene rings")}')
        if pat:
            S.append('Substitution pattern: ' + ', '.join(pat) + '.')
        if f['biaryl']:
            S.append(f'It contains {plural(int(f["biaryl"]), "direct aryl-aryl bond", "direct aryl-aryl bonds")}'
                     f'{", hindered by ortho substituents" if f["biaryl_ortho"] else ""}.')
    groups = []
    for p, a, b in FGP:
        n = len(m.GetSubstructMatches(p))
        if n:
            groups.append(plural(n, a, b))
    hal = {s: sum(1 for x in m.GetAtoms() if x.GetSymbol() == s) for s in ('F', 'Cl', 'Br', 'I')}
    name = {'F': 'fluorine', 'Cl': 'chlorine', 'Br': 'bromine', 'I': 'iodine'}
    hl = [plural(n, f'{name[s]} atom', f'{name[s]} atoms') for s, n in hal.items() if n]
    S.append('Functional groups: ' + (', '.join(groups) if groups else 'none beyond hydrocarbon') + '.')
    S.append('Halogens: ' + (', '.join(hl) if hl else 'none') + '.')
    hbd, hba = rd.CalcNumHBD(m), rd.CalcNumHBA(m)
    t = f'Hydrogen bonding: {plural(hbd, "donor", "donors")} and {plural(hba, "acceptor", "acceptors")}'
    if f['hb_intra']:
        t += f'; {plural(int(f["hb_intra"]), "donor is", "donors are")} positioned for an intramolecular hydrogen bond'
    if hbd >= 3:
        t += '; many donors'
    S.append(t + '.')
    nrot = rd.CalcNumRotatableBonds(m)
    fsp3 = rd.CalcFractionCSP3(m)
    t = f'Flexibility: {plural(nrot, "rotatable bond", "rotatable bonds")}, fraction of sp3 carbons {fsp3:.2f}'
    if f['chain_CH2_len'] >= 3:
        t += f', longest unbranched CH2 chain of {int(f["chain_CH2_len"])} carbons'
    if nrot == 0:
        t += '; the molecule is rigid'
    S.append(t + '.')
    S.append(f'The largest conjugated system spans {int(f["conj_largest"])} of {heavy} heavy atoms.')
    ch = [a.GetFormalCharge() for a in m.GetAtoms()]
    if any(ch):
        S.append(f'Formal charges: {sum(1 for c in ch if c > 0)} positive and {sum(1 for c in ch if c < 0)} '
                 f'negative atoms, net charge {sum(ch):+d}.')
    else:
        S.append('No formal charges.')
    nuniq = round(heavy * f['sym_class_ratio'])
    ratio = f['sym_class_ratio']
    lvl = 'high' if ratio <= 0.5 else ('moderate' if ratio <= 0.75 else 'low')
    S.append(f'Topological symmetry: {nuniq} distinct atom environments among {heavy} heavy atoms '
             f'({lvl} symmetry), {int(f["sym_n_automorph"])} graph automorphisms.')
    cc = int(f['stereo_centers'])
    if cc or f['db_trans'] or f['db_cis']:
        t = f'Stereochemistry: {plural(cc, "stereocenter", "stereocenters")} ({int(f["stereo_assigned"])} specified)'
        if f['db_trans'] or f['db_cis']:
            t += f', {int(f["db_trans"])} E and {int(f["db_cis"])} Z double bonds'
        S.append(t + '.')
    else:
        S.append('Stereochemistry: none.')
    return ' '.join(S)
