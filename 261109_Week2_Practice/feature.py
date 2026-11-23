"""feature.py : 분자(RDKit Mol) -> 그래프 텐서 (x, edge_index, edge_attr)

연구실 코드 `0.연구과제/Shipboard_RPB/9.ml_gnn/code/model/feature.py` 와 같은 특징 정의를 쓴다.
바꾼 것은 속도뿐이다 (SMARTS 를 미리 컴파일하고 분자당 한 번만 매칭). 결과 텐서는 같다.

    from feature import smiles_to_graph
    x, edge_index, edge_attr = smiles_to_graph('C1CCNCC1')   # piperidine
    x.shape  -> (6, 147)   원자 6개 x 원자 특징 147개
"""
import numpy as np
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')


# ---------------------------------------------------------------- 1. 작용기 SMARTS (연구실 원본 그대로)
FUNCTIONAL_GROUPS = {
    # retrieved from https://www.daylight.com/dayhtml_tutorials/languages/smarts/smarts_examples.html
    'Alkyl Carbon': '[CX4]',
    'Allenic Carbon': '[$([CX2](=C)=C)]',
    'Vinylic Carbon': '[$([CX3]=[CX3])]',
    'Acetylenic Carbon': '[$([CX2]#C)]',
    'Arene': 'c',
    'Carbonyl': '[CX3]=[OX1]',
    'Carbonyl with Carbon': '[CX3](=[OX1])C',
    'Carbonyl with Nitrogen': '[OX1]=CN',
    'Carbonyl with Oxygen': '[CX3](=[OX1])O',
    'Acyl Halide': '[CX3](=[OX1])[F,Cl,Br,I]',
    'Aldehyde': '[CX3H1](=O)[#6]',
    'Anhydride': '[CX3](=[OX1])[OX2][CX3](=[OX1])',
    'Amide': '[NX3][CX3](=[OX1])[#6]',
    'Amidinium': '[NX3][CX3]=[NX3+]',
    'Carbamate': '[NX3,NX4+][CX3](=[OX1])[OX2,OX1-]',
    'Carbamic ester': '[NX3][CX3](=[OX1])[OX2H0]',
    'Carbamic acid': '[NX3,NX4+][CX3](=[OX1])[OX2H,OX1-]',
    'Carbonic acid': '[CX3](=[OX1])(O)O',
    'Carbonic ester': 'C[OX2][CX3](=[OX1])[OX2]C',
    'Carboxylic acid': '[CX3](=O)[OX1H0-,OX2H1]',
    'Cyanamide': '[NX3][CX2]#[NX1]',
    'Ester': '[#6][CX3](=O)[OX2H0][#6]',
    'Ketone': '[#6][CX3](=O)[#6]',
    'Ether': '[OD2]([#6])[#6]',
    'Enamine': '[NX3][CX3]=[CX3]',
    'Primary or secondary amine, not amide': '[NX3;H2;!$(NC=[!#6]);!$(NC#[!#6])][#6]',
    'Two primary or secondary amines': '[NX3;H2,H1;!$(NC=O)].[NX3;H2,H1;!$(NC=O)]',
    'Enamine or Aniline Nitrogen': '[NX3][$(C=C),$(cc)]',
    'Azide group': '[$(*-[NX2-]-[NX2+]#[NX1]),$(*-[NX2]=[NX2+]=[NX1-])]',
    'Azide ion': '[$([NX1-]=[NX2+]=[NX1-]),$([NX1]#[NX2+]-[NX1-2])]',
    'Nitrogen': '[#7]',
    'Azo Nitrogen': '[NX2]=[NX2]',
    'Azoxy Nitrogen': '[$([NX2]=[NX3+]([O-])[#6]),$([NX2]=[NX3+0](=[O])[#6])]',
    'Diazo Nitrogen': '[$([#6]=[N+]=[N-]),$([#6-]-[N+]#[N])]',
    'Azole': '[$([nr5]:[nr5,or5,sr5]),$([nr5]:[cr5]:[nr5,or5,sr5])]',
    'Hydrazine': '[NX3][NX3]',
    'Hydrazone ': '[NX3][NX2]=[*]',
    'Substituted imine': '[CX3;$([C]([#6])[#6]),$([CH][#6])]=[NX2][#6]',
    'Substituted or un-substituted imine': '[$([CX3]([#6])[#6]),$([CX3H][#6])]=[$([NX2][#6]),$([NX2H])]',
    'Iminium': '[NX3+]=[CX3]',
    'Unsubstituted dicarboximide': '[CX3](=[OX1])[NX3H][CX3](=[OX1])',
    'Substituted dicarboximide': '[CX3](=[OX1])[NX3H0]([#6])[CX3](=[OX1])',
    'Dicarboxdiimide': '[CX3](=[OX1])[NX3H0]([NX3H0]([CX3](=[OX1]))[CX3](=[OX1]))[CX3](=[OX1])',
    'Nitrate group': '[$([NX3](=[OX1])(=[OX1])O),$([NX3+]([OX1-])(=[OX1])O)]',
    'Nitrile': '[NX1]#[CX2]',
    'Isonitrile': '[CX1-]#[NX2+]',
    'Nitro group': '[$([NX3](=O)=O),$([NX3+](=O)[O-])][!#8]',
    'Nitroso-group': '[NX2]=[OX1]',
    'Hydroxyl': '[OX2H]',
    'Hydroxyl in Alcohol': '[#6][OX2H]',
    'Hydroxyl in Carboxylic Acid': '[OX2H][CX3]=[OX1]',
    'Enol': '[OX2H][#6X3]=[#6]',
    'Phenol': '[OX2H][cX3]:[c]',
    'Peroxide groups': '[OX2,OX1-][OX2,OX1-]',
    'Thiol': '[#16X2H]',
    'Thioamide': '[NX3][CX3]=[SX1]',
    'Sulfide': '[#16X2H0]',
    'Sulfinate': '[$([#16X3](=[OX1])[OX2H0]),$([#16X3+]([OX1-])[OX2H0])]',
    'Sulfinic Acid': '[$([#16X3](=[OX1])[OX2H,OX1H0-]),$([#16X3+]([OX1-])[OX2H,OX1H0-])]',
    'Sulfone': '[$([#16X4](=[OX1])=[OX1]),$([#16X4+2]([OX1-])[OX1-])]',
    'Sulfonate': '[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H0]),$([#16X4+2]([OX1-])([OX1-])([#6])[OX2H0])]',
    'Sulfonamide': '[$([#16X4]([NX3])(=[OX1])(=[OX1])[#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[#6])]',
    'Sulfoxide': '[$([#16X3]=[OX1]),$([#16X3+][OX1-])]',
    'Sulfate': '[$([#16X4](=[OX1])(=[OX1])([OX2H,OX1H0-])[OX2][#6]),$([#16X4+2]([OX1-])([OX1-])([OX2H,OX1H0-])[OX2][#6])]',
    'Sulfuric acid ester': '[$([SX4](=O)(=O)(O)O),$([SX4+2]([O-])([O-])(O)O)]',
    'Sulfamate': '[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2][#6]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2][#6])]',
    'Sulfamic Acid': '[$([#16X4]([NX3])(=[OX1])(=[OX1])[OX2H,OX1H0-]),$([#16X4+2]([NX3])([OX1-])([OX1-])[OX2H,OX1H0-])]',
    'Sulfenic acid': '[#16X2][OX2H,OX1H0-]',
    'Sulfenate': '[#16X2][OX2H0]',
    'Any carbon attached to any halogen': '[#6][F,Cl,Br,I]',
    'Halogen': '[F,Cl,Br,I]',
    'Acyl Halide': '[CX3](=[OX1])[F,Cl,Br,I]',
    'sp2 cationic carbon': '[$([cX2+](:*):*)]',
    'Aromatic sp2 carbon': '[$([cX3](:*):*),$([cX2+](:*):*)]',
    'Any sp2 carbon': '[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]',
    'Any sp2 nitrogen': '[$([nX3](:*):*),$([nX2](:*):*),$([#7X2]=*),$([NX3](=*)=*),$([#7X3+](-*)=*),$([#7X3+H]=*)]',
    'Explicit Hydrogen on sp2-Nitrogen': '[$([#1X1][$([nX3](:*):*),$([nX2](:*):*),$([#7X2]=*),$([NX3](=*)=*),$([#7X3+](-*)=*),$([#7X3+H]=*)])]',
    'sp3 nitrogen': '[$([NX4+]),$([NX3]);!$(*=*)&!$(*:*)]',
    'Explicit Hydrogen on an sp3 N': '[$([#1X1][$([NX4+]),$([NX3]);!$(*=*)&!$(*:*)])]',
    'sp2 N in N-Oxide': '[$([$([NX3]=O),$([NX3+][O-])])]',
    'sp3 N in N-Oxide Exclusive': '[$([$([NX4]=O),$([NX4+][O-])])]',
    'sp3 N in N-Oxide Inclusive': '[$([$([NX4]=O),$([NX4+][O-,#0])])]',
    'Quaternary Nitrogen': '[$([NX4+]),$([NX4]=*)]',
    'Tricoordinate S double bonded to N': '[$([SX3]=N)]',
    'S double-bonded to Carbon': '[$([SX1]=[#6])]',
    'Triply bonded N': '[$([NX1]#*)]',
    'Divalent Oxygen': '[$([OX2])]',
    'Unbranched alkane groups': '[R0;D2][R0;D2][R0;D2][R0;D2]',
    'Unbranched chain groups': '[R0;D2]~[R0;D2]~[R0;D2]~[R0;D2]',
    'Long chain groups': '[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]~[AR0]',
    'Terminal S bonded to P': '[$([SX1]~P)]',
    'Nitrogen on -N-C=N-': '[$([NX3]C=N)]',
    'Nitrogen on -N-N=C-': '[$([NX3]N=C)]',
    'Nitrogen on -N-N=N-': '[$([NX3]N=N)]',
    'Oxygen in -O-C=N-': '[$([OX2]C=N)]',
    'S in aromatic 5-ring with lone pair': '[sX2r5]',
    'Aromatic 5-Ring O with Lone Pair': '[oX2r5]',
    'N in 5-sided aromatic ring': '[nX2r5]',
    'N in 5-ring arom': '[$([nX2r5]:[a-]),$([nX2r5]:[a]:[a-])]',
    'CIS or TRANS double bond in a ring': r'*/,\[R]=;@[R]/,\*',
    'CIS or TRANS double or aromatic bond in a ring': r'*/,\[R]=,:;@[R]/,\*',
    'Unfused benzene ring': '[cR1]1[cR1][cR1][cR1][cR1][cR1]1',
    'Anionic divalent Nitrogen': '[NX2-]',
    'Oxenium Oxygen': '[OX2H+]=*',
    'Oxonium Oxygen': '[OX3H2+]',
    'Carbocation': '[#6+]',
    'Hydrogen-bond acceptor': ['[#6,#7;R0]=[#8]', '[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]'],
    'Hydrogen-bond donor': ['[!$([#6,H0,-,-2,-3])]', '[!H0;#7,#8,#9]'],
    'Possible intramolecular H-bond': ['[O,N;!H0]-*~*-*=[$([C,N;R0]=O)]'],
}


for key, val in FUNCTIONAL_GROUPS.items():  # 값 형식 통일: 문자열 하나도 리스트로
    if not isinstance(val, list):
        FUNCTIONAL_GROUPS[key] = [val]

_FG_PATTERNS = [Chem.MolFromSmarts(s) for strings in FUNCTIONAL_GROUPS.values() for s in strings]
_ROTATABLE = Chem.MolFromSmarts('[!$(*#*)&!D1]-!@[!$(*#*)&!D1]')
_ACYCLIC = Chem.MolFromSmarts('*!@*')
_SINGLE_NOT_RING = Chem.MolFromSmarts('*-!@*')


def one_hot(value, choices):
    """value 가 choices 의 몇 번째인지 0/1 리스트로. 목록에 없으면 전부 0."""
    return [int(value == c) for c in choices]


# ---------------------------------------------------------------- 2. 원자(노드) 특징
ELEMENTS = [1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53]       # H B C N O F P S Cl Br I

def atom_features(atom, fg_bits):
    return (one_hot(atom.GetAtomicNum(), ELEMENTS)                       # 11 원소
            + one_hot(atom.GetFormalCharge(), [-2, -1, 0, 1, 2])         # 5  형식 전하
            + one_hot(int(atom.GetHybridization()), [1, 2, 3, 4])        # 4  혼성 (S, SP, SP2, SP3)
            + one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3])                # 4  붙은 수소 수
            + [int(atom.GetIsAromatic())]                                # 1  방향족
            + one_hot(atom.GetTotalValence(), [1, 2, 3, 4, 5, 6])        # 6  원자가
            + one_hot(atom.GetTotalDegree(), [1, 2, 3, 4])               # 4  연결 수
            + [int(atom.IsInRing())]                                     # 1  고리 안
            + [0]                                                        # 1  반응 중심 (물성 예측에서는 0)
            + fg_bits)                                                   # ~110 작용기 소속 여부


def functional_group_bits(mol):
    """원자마다 '이 원자가 각 작용기 매칭에 포함되는가' 0/1 배열 (n_atoms x n_groups)."""
    bits = np.zeros((mol.GetNumAtoms(), len(_FG_PATTERNS)), dtype=int)
    for j, patt in enumerate(_FG_PATTERNS):
        for match in mol.GetSubstructMatches(patt):
            bits[list(match), j] = 1
    return bits


# ---------------------------------------------------------------- 3. 결합(엣지) 특징
def _bond_set(mol, patt):
    return {frozenset(m) for m in mol.GetSubstructMatches(patt)}

def bond_features(bond, rot, acyc, snr):
    key = frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
    return (one_hot(bond.GetBondTypeAsDouble(), [1.0, 1.5, 2.0, 3.0])    # 4 단일/방향족/이중/삼중
            + [int(bond.GetIsConjugated())]                              # 1 공액
            + [int(key in rot), int(key in acyc), int(key in snr)]       # 3 회전가능/비고리/고리밖 단일
            + one_hot(bond.GetBeginAtom().GetAtomicNum(), [1, 6, 7, 8])  # 4 시작 원자
            + one_hot(bond.GetEndAtom().GetAtomicNum(), [1, 6, 7, 8]))   # 4 끝 원자


# ---------------------------------------------------------------- 4. 분자 -> 그래프
def mol_to_features(mol):
    fg = functional_group_bits(mol)
    x = [atom_features(a, fg[a.GetIdx()].tolist()) for a in mol.GetAtoms()]

    rot, acyc, snr = (_bond_set(mol, p) for p in (_ROTATABLE, _ACYCLIC, _SINGLE_NOT_RING))
    src, dst, edge_attr = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b, rot, acyc, snr)
        src += [i, j]; dst += [j, i]          # 무방향 그래프: 양방향 엣지 두 개
        edge_attr += [f, f]

    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float).reshape(-1, NUM_EDGE_FEATURES)
    return x, edge_index, edge_attr


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_features(mol)


NUM_EDGE_FEATURES = 16
NUM_NODE_FEATURES = 11 + 5 + 4 + 4 + 1 + 6 + 4 + 1 + 1 + len(_FG_PATTERNS)

# 특징 번호 -> 이름 (노트북에서 "이 원자는 어떤 칸이 1인가"를 읽을 때 쓴다)
NODE_FEATURE_NAMES = ([f'element={Chem.GetPeriodicTable().GetElementSymbol(z)}' for z in ELEMENTS]
                      + [f'charge={c}' for c in [-2, -1, 0, 1, 2]]
                      + [f'hybrid={h}' for h in ['S', 'SP', 'SP2', 'SP3']]
                      + [f'numH={n}' for n in range(4)] + ['aromatic']
                      + [f'valence={v}' for v in range(1, 7)] + [f'degree={d}' for d in range(1, 5)]
                      + ['in_ring', 'reaction_center']
                      + [f'FG:{k}' + (f'#{i}' if len(v) > 1 else '')
                         for k, v in FUNCTIONAL_GROUPS.items() for i in range(len(v))])
EDGE_FEATURE_NAMES = (['single', 'aromatic', 'double', 'triple', 'conjugated', 'rotatable', 'acyclic',
                       'single_not_in_ring'] + [f'begin={s}' for s in 'HCNO'] + [f'end={s}' for s in 'HCNO'])
assert len(NODE_FEATURE_NAMES) == NUM_NODE_FEATURES and len(EDGE_FEATURE_NAMES) == NUM_EDGE_FEATURES


if __name__ == '__main__':
    x, ei, ea = smiles_to_graph('C1CCNCC1')
    print('piperidine  x:', tuple(x.shape), ' edge_index:', tuple(ei.shape), ' edge_attr:', tuple(ea.shape))
    print('edge_index =')
    print(ei)
