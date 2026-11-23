"""s0_audit_pre.py : 누설 감사 (test 를 열기 전에 끝내는 항목 i–iv).

test 라벨은 읽지 않는다 (라벨-특징 상관은 train 행만). 결과: P3/tables/audit_*.csv, P3/notes/audit_pre.json
 (i)  InChIKey 전체·첫 블록 겹침: SMILES 에서 다시 계산 → 분할 파일 키와 대조 → train/validate/test 쌍별 겹침,
      test·validate 의 첫 블록 겹침 행이 중복 제거 목록(overlap=1)과 같은지. 외부 특허 정제본도 SMILES 에서 키를 다시 계산해 대조.
      T8 M2 규칙 프롬프트에 보인 train 200분자의 위치.
 (ii) 특징 라벨 유래 없음: 특징 코드 정적 점검(라벨 열 참조·외부 MP 값·LLM 호출), test 행 표본 재계산 대조(F0, F5, F1, O3, F3 일부),
      train 행 라벨-특징 상관 최대값. 최종 후보 특징에 보조 물성(H17) 열 없음.
 (iii) 모든 트랙 ledger 의 test_* 열, phase2 스크립트의 test 행 참조.
"""
import ast
import glob
import json
import os
import re
import sys
import time

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as P  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog('rdApp.*')
mc = P.mc
HB = r'C:\Users\mthog\CEPL ML\mp_mae_push'
WK2 = r'C:\Users\mthog\CEPL ML\261109_Week2_Practice'
OUT = {}
rows_i = []


def key(smi):
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToInchiKey(m) if m is not None else ''


def part_i():
    _, s = P.split('random')
    s = s.sort_values('row').reset_index(drop=True)
    assert (s['row'].to_numpy() == np.arange(len(s))).all()
    mp = pd.read_csv(os.path.join(WK2, 'data', 'mp.csv'))
    OUT['mp_rows'] = len(mp)
    OUT['split_smiles_eq_mp'] = bool((mp['smiles'].to_numpy() == s['smiles'].to_numpy()).all())
    t0 = time.time()
    k = np.array([key(x) for x in s['smiles']])
    OUT['inchikey_recompute_s'] = round(time.time() - t0, 1)
    OUT['inchikey_mismatch_vs_splitfile'] = int((k != s['inchikey'].to_numpy()).sum())
    OUT['inchikey_empty'] = int((k == '').sum())
    b1 = np.array([x.split('-')[0] for x in k])
    for how in P.SPLITS:
        idx, _ = P.split(how)
        sets = {n: idx[n] for n in ('train', 'validate', 'test')}
        for a, b in [('train', 'validate'), ('train', 'test'), ('validate', 'test')]:
            fa, fb = set(k[sets[a]]), set(k[sets[b]])
            ba, bb = set(b1[sets[a]]), set(b1[sets[b]])
            rows_i.append(dict(split=how, pair=f'{a}–{b}', full_key_shared=len(fa & fb), block1_shared=len(ba & bb),
                               rows_in_b_full=int(np.isin(k[sets[b]], list(fa)).sum()),
                               rows_in_b_block1=int(np.isin(b1[sets[b]], list(ba)).sum())))
        # 세트 안 중복
        for n, r in sets.items():
            rows_i.append(dict(split=how, pair=f'{n} 안', full_key_shared=int(len(r) - len(set(k[r]))),
                               block1_shared=int(len(r) - len(set(b1[r]))), rows_in_b_full='', rows_in_b_block1=''))
        # 중복 제거 목록과 대조
        d = pd.read_csv(os.path.join(P.INP, f'dedup_{how}.csv'))
        for which, others in [('test', ('train', 'validate')), ('validate', ('train', 'test'))]:
            ob = set(np.concatenate([b1[sets[o]] for o in others]))
            mine = np.isin(b1[sets[which]], list(ob)).astype(int)
            dd = d[d['set'] == which].set_index('row').loc[sets[which]]
            # 중복 제거 목록의 정의: test 는 train∪validate 와, validate 는 train 과 겹침 (1단계 정의) — 두 정의 모두 기록
            ob_tr = set(b1[sets['train']]) | (set(b1[sets['validate']]) if which == 'test' else set())
            mine2 = np.isin(b1[sets[which]], list(ob_tr)).astype(int)
            OUT[f'{how}_{which}_dedup_overlap_n'] = int(dd['overlap'].sum())
            OUT[f'{how}_{which}_recomputed_overlap_vs_all_other_sets'] = int(mine.sum())
            OUT[f'{how}_{which}_recomputed_overlap_vs_earlier_sets'] = int(mine2.sum())
            OUT[f'{how}_{which}_dedup_list_equals_recomputed(earlier)'] = bool((dd['overlap'].to_numpy() == mine2).all())
            OUT[f'{how}_{which}_dedup_inchikey_match'] = bool((dd['inchikey'].to_numpy() == k[sets[which]]).all())
            OUT[f'{how}_{which}_n_dedup'] = int((dd['overlap'] == 0).sum())
    # 외부 특허 정제본: SMILES 에서 키 다시 계산
    pc = pd.read_csv(os.path.join(P.INP, 'patent_clean.csv'))
    t0 = time.time()
    kp = np.array([key(x) for x in pc['smiles']])
    OUT['patent_n'] = len(pc)
    OUT['patent_key_recompute_s'] = round(time.time() - t0, 1)
    OUT['patent_key_mismatch_vs_file'] = int((kp != pc['inchikey'].to_numpy()).sum())
    OUT['patent_key_empty'] = int((kp == '').sum())
    bp = set(x.split('-')[0] for x in kp if x)
    fp = set(x for x in kp if x)
    bpf = set(x.split('-')[0] for x in pc['inchikey'])
    for how in P.SPLITS:
        idx, _ = P.split(how)
        for n in ('train', 'validate', 'test'):
            r = idx[n]
            rows_i.append(dict(split=how, pair=f'특허 정제본–{n}', full_key_shared=int(np.isin(k[r], list(fp)).sum()),
                               block1_shared=int(np.isin(b1[r], list(bp)).sum()), rows_in_b_full=int(np.isin(k[r], list(fp)).sum()),
                               rows_in_b_block1=int(np.isin(b1[r], list(bp | bpf)).sum())))
    # T8 M2 프롬프트에 보인 200분자
    sh = pd.read_csv(os.path.join(P.INP, 't8_llm_sample200.csv'))
    for how in P.SPLITS:
        idx, _ = P.split(how)
        OUT[f'llm200_all_in_train_{how}'] = bool(np.isin(sh['row'], idx['train']).all())
        vt = np.concatenate([idx['validate'], idx['test']])
        OUT[f'llm200_block1_in_val_or_test_{how}'] = int(np.isin(b1[sh['row'].to_numpy()], list(set(b1[vt]))).sum())
    OUT['llm200_smiles_match'] = bool((s.set_index('row').loc[sh['row'], 'smiles'].to_numpy() == sh['smiles'].to_numpy()).all())
    pd.DataFrame(rows_i).to_csv(os.path.join(P.TAB, 'audit_i_inchikey_overlap.csv'), index=False)
    return s, k, b1


def part_ii(s):
    """test 행 표본의 특징을 SMILES 에서 다시 계산해 캐시와 대조 + 정적 점검 + train 라벨 상관."""
    rng = np.random.default_rng(2026)
    te = np.union1d(P.split('random')[0]['test'], P.split('scaffold')[0]['test'])
    samp = np.sort(rng.choice(te, 150, replace=False))
    samp3 = samp[:25]
    mols = [Chem.MolFromSmiles(x) for x in s.loc[samp, 'smiles']]
    res = []
    sys.path.insert(0, WK2)
    import train_xgb  # noqa: E402
    X0 = np.load(os.path.join(mc.CACHE_DIR, 'X_desc217.npy'))
    X0n = train_xgb.featurize(mols, 'desc')
    res.append(dict(feature='F0 RDKit217', n_rows=len(samp), max_abs_diff=float(np.nanmax(np.abs(X0n - X0[samp]))),
                    rel_ok=bool(np.allclose(X0n, X0[samp], rtol=1e-6, atol=1e-6))))
    sys.path.insert(0, os.path.join(HB, 'phase2', 'T8_nlp'))
    import f5_rules  # noqa: E402
    F5 = np.load(os.path.join(mc.CACHE_DIR, 'F5_rules.npy'))
    F5n = np.array([[f5_rules.featurize(m)[c] for c in f5_rules.COLS] for m in mols], dtype=np.float32)
    res.append(dict(feature='F5 LLM 규칙 101', n_rows=len(samp), max_abs_diff=float(np.nanmax(np.abs(F5n - F5[samp]))),
                    rel_ok=bool(np.allclose(F5n, F5[samp], rtol=1e-5, atol=1e-5, equal_nan=True))))
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    F1 = np.load(os.path.join(mc.CACHE_DIR, 'X_morgan_count_r2_2048.npy'))
    F1n = np.array([np.minimum(gen.GetCountFingerprintAsNumPy(m), 255) for m in mols], dtype=np.uint8)
    res.append(dict(feature='F1 Morgan count', n_rows=len(samp), max_abs_diff=float(np.abs(F1n.astype(int) - F1[samp]).max()),
                    rel_ok=bool((F1n == F1[samp]).all())))
    # O3: T1 s2_features.o3_row 를 소스에서 떼어 실행 (모듈 import 는 전체 계산을 돌리므로 피함)
    src = open(os.path.join(HB, 'phase2', 'T1_boosting', 's2_features.py'), encoding='utf-8').read()
    tree = ast.parse(src)
    lines = src.splitlines()
    ns = {}
    exec('import numpy as np\nfrom rdkit import Chem\nfrom rdkit.Chem import rdFingerprintGenerator, rdCIPLabeler\n', ns)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'o3_row' or \
                isinstance(node, ast.Assign) and any(getattr(t, 'id', '') in ('GEN_CHI', 'GEN_ACH', 'O3_COLS') for t in node.targets):
            exec('\n'.join(lines[node.lineno - 1:node.end_lineno]), ns)
    O3 = np.load(os.path.join(mc.CACHE_DIR, 'X_stereo_o3.npy'))
    O3n = np.array([ns['o3_row'](x) for x in s.loc[samp, 'smiles']], dtype=np.float32)
    res.append(dict(feature='O3 입체 15', n_rows=len(samp), max_abs_diff=float(np.nanmax(np.abs(O3n - O3[samp]))),
                    rel_ok=bool(np.allclose(O3n, O3[samp], atol=1e-5, equal_nan=True))))
    # F3: 2D 열은 결정적, 3D 열은 배좌 생성(seed) — 25행만
    sys.path.insert(0, os.path.join(HB, 'phase2', 'T4T5_features_data'))
    try:
        import phys_feats  # noqa: E402
        cols = pd.read_csv(os.path.join(mc.CACHE_DIR, 'X_phys_cols.csv'))['col'].tolist()
        X3 = np.load(os.path.join(mc.CACHE_DIR, 'X_phys.npy'))
        t0 = time.time()
        out = []
        for smi in s.loc[samp3, 'smiles']:
            r = phys_feats.featurize_smiles(smi, n_conf=5)
            out.append([r[c] for c in cols] if isinstance(r, dict) else list(r))
        X3n = np.array(out, dtype=np.float32)
        diff = np.abs(X3n - X3[samp3])
        c2 = [i for i, c in enumerate(cols) if c in phys_feats.COLS_2D]
        c3 = [i for i, c in enumerate(cols) if c not in phys_feats.COLS_2D]
        res.append(dict(feature='F3 물리·결정 (2D 열)', n_rows=len(samp3), max_abs_diff=float(np.nanmax(diff[:, c2])),
                        rel_ok=bool(np.allclose(X3n[:, c2], X3[samp3][:, c2], atol=1e-4, equal_nan=True))))
        res.append(dict(feature='F3 물리·결정 (3D 열)', n_rows=len(samp3), max_abs_diff=float(np.nanmax(diff[:, c3])),
                        rel_ok=bool(np.allclose(X3n[:, c3], X3[samp3][:, c3], rtol=1e-3, atol=1e-3, equal_nan=True))))
        OUT['F3_recompute_s'] = round(time.time() - t0, 1)
    except Exception as e:  # noqa: BLE001
        res.append(dict(feature='F3 재계산 실패', n_rows=0, max_abs_diff=np.nan, rel_ok=False, note=repr(e)[:200]))
    pd.DataFrame(res).to_csv(os.path.join(P.TAB, 'audit_ii_feature_recompute.csv'), index=False)
    # train 행 라벨-특징 상관 (분할 random train 만)
    tr = P.split('random')[0]['train']
    y = P.y_masked('random')[tr]
    mats = {'F0': X0, 'F3': np.load(os.path.join(mc.CACHE_DIR, 'X_phys.npy')), 'F5': F5,
            'O3': O3, 'F1': F1.astype(np.float32)}
    cr = []
    for nm, M in mats.items():
        A = np.nan_to_num(M[tr].astype(float))
        sd = A.std(0)
        ok = sd > 0
        rr = np.zeros(A.shape[1])
        Az = (A[:, ok] - A[:, ok].mean(0)) / sd[ok]
        rr[ok] = (Az * ((y - y.mean()) / y.std())[:, None]).mean(0)
        j = int(np.argmax(np.abs(rr)))
        cr.append(dict(block=nm, n_cols=A.shape[1], max_abs_pearson_r_train=float(np.abs(rr).max()), argmax_col=j,
                       n_cols_abs_r_gt_0_9=int((np.abs(rr) > 0.9).sum())))
    pd.DataFrame(cr).to_csv(os.path.join(P.TAB, 'audit_ii_label_corr_train.csv'), index=False)
    # 정적 점검
    code = {'F0 train_xgb.featurize': os.path.join(WK2, 'train_xgb.py'), 'GIN graph feature.py': os.path.join(WK2, 'feature.py'),
            'F3 phys_feats.py': os.path.join(HB, 'phase2', 'T4T5_features_data', 'phys_feats.py'),
            'F5 f5_rules.py': os.path.join(HB, 'phase2', 'T8_nlp', 'f5_rules.py'),
            'O3/F1 s2_features.py': os.path.join(HB, 'phase2', 'T1_boosting', 's2_features.py')}
    st = []
    for nm, pth in code.items():
        src = open(pth, encoding='utf-8').read()
        tree = ast.parse(src)
        mods = sorted({a.name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} |
                      {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module})
        st.append(dict(code=nm, imports=','.join(mods),
                       reads_files=bool(re.search(r'read_csv|np\.load|open\(', src)),
                       label_refs=len(re.findall(r"\['value'\]|\[\"value\"\]|mp\.csv|y_all|y\[", src)),
                       llm_calls=len(re.findall(r'host\.llm|anthropic|openai', src)),
                       external_mp=len(re.findall(r'patent|melting_point|ochem', src, re.I))))
    pd.DataFrame(st).to_csv(os.path.join(P.TAB, 'audit_ii_static_code.csv'), index=False)
    cols3 = pd.read_csv(os.path.join(mc.CACHE_DIR, 'X_phys_cols.csv'))['col'].tolist()
    OUT['F3_cols'] = cols3
    OUT['F3_has_aux_cols'] = [c for c in cols3 if re.search(r'aux|logS|logP|solub|dens|bp|vap|pred', c, re.I)]


def part_iii():
    """모든 ledger 의 test_* 열과 phase2 스크립트의 test 행 참조."""
    leds = {}
    r = host_ledgers()
    rows = []
    for name, pth in r:
        try:
            d = pd.read_csv(pth, encoding='utf-8-sig', low_memory=False)
        except Exception as e:  # noqa: BLE001
            rows.append(dict(ledger=name, n_rows=-1, note=repr(e)[:100]))
            continue
        tc = [c for c in d.columns if c.lower().startswith('test')]
        nz = np.zeros(len(d), bool)
        for c in tc:
            v = d[c]
            nz |= v.notna().to_numpy() & (v.astype(str).str.strip() != '').to_numpy()
        ids = d.loc[nz, 'run_id'].astype(str).tolist() if 'run_id' in d else []
        rows.append(dict(ledger=name, n_rows=len(d), test_cols=','.join(tc), n_rows_test_filled=int(nz.sum()),
                         run_ids_test_filled=';'.join(ids[:20]), sha256=P.sha256(pth)))
    pd.DataFrame(rows).to_csv(os.path.join(P.TAB, 'audit_iii_ledgers.csv'), index=False)
    # 스크립트 test 참조
    hits = []
    for pth in glob.glob(os.path.join(HB, 'phase2', '**', '*.py'), recursive=True):
        for i, l in enumerate(open(pth, encoding='utf-8', errors='replace').read().splitlines()):
            if re.search(r"\[['\"]test['\"]\]|== ?['\"]test['\"]|set ?== ?['\"]test|'test'\)|test_MAE|_test\.csv", l):
                hits.append(dict(script=os.path.relpath(pth, HB), line=i + 1, text=l.strip()[:220]))
    pd.DataFrame(hits).to_csv(os.path.join(P.TAB, 'audit_iii_script_test_refs.csv'), index=False)
    OUT['script_test_ref_hits'] = len(hits)


def host_ledgers():
    out = []
    for vid in json.load(open(os.path.join(P.P3, 'notes', 'ledger_vids.json'))):
        out.append(('artifact:' + vid['vid'][:8] + ':' + vid['name'], vid['path']))
    for pth in glob.glob(os.path.join(HB, '**', '*ledger*.csv'), recursive=True):
        out.append(('host:' + os.path.relpath(pth, HB), pth))
    return out


if __name__ == '__main__':
    with mc.CostMeter('감사') as cm:
        s, k, b1 = part_i()
        part_ii(s)
        part_iii()
    OUT['cpu_core_h'] = cm.cpu_core_h
    json.dump(OUT, open(os.path.join(P.P3, 'notes', 'audit_pre.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    P.ledger(dict(run_id='P3_audit_pre', model='-', features='InChIKey·특징 재계산', data_variant='19,945 + 특허 194,801',
                  split='both', params={'steps': 'i,ii,iii'}, notes='누설 감사 i–iii (test 라벨 미사용)', **cm.as_row()))
    print(json.dumps({k_: v for k_, v in OUT.items() if k_ != 'F3_cols'}, ensure_ascii=False, indent=0))
