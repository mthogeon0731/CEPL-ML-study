"""교수님 노트북 01_multi_model_comparison.ipynb 를 PROP='mp' 로 바꾼 사본을 만든다.

교수님 셀은 한 글자도 바꾸지 않는다 (첫 셀의 PROP 만 'mp').
발표 재료 저장용 [추가] 셀을 (1) import 셀 뒤에 그림 자동 저장 훅, (2) 맨 끝에 결과·예측 저장 + Descriptor parity +
XGBoost SHAP / RF 중요도 셀로 덧붙인다. 결과표 수치에는 영향이 없다.
"""
import json, sys

SRC, DST = sys.argv[1], sys.argv[2]
nb = json.load(open(SRC, encoding='utf-8'))
cells = nb['cells']


def code(src):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': src.strip('\n').splitlines(keepends=True)}


def md(src):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': src.strip('\n').splitlines(keepends=True)}


# 1) 첫 코드 셀: PROP 만 바꾼다
first = ''.join(cells[1]['source'])
assert first.startswith("PROP = 'logP'"), first[:40]
cells[1]['source'] = first.replace("PROP = 'logP'", "PROP = 'mp'  ", 1).splitlines(keepends=True)

# 2) import 셀(인덱스 3) 뒤에 그림 자동 저장 훅
hook = code(r'''
# [추가 — mp 팀] 노트북이 그리는 그림을 results_nb_mp/ 에 PNG 로도 저장한다. 교수님 코드와 결과에는 영향 없음.
import os
OUT = 'results_nb_mp'; os.makedirs(OUT, exist_ok=True)
_FIG_NAMES = iter(['fig01_mp_hist', 'fig02_parity_random_MorganFP', 'fig03_parity_scaffold_MorganFP',
                   'fig04_bar_random_vs_scaffold', 'fig05_parity_random_Descriptor', 'fig06_parity_scaffold_Descriptor'])
_orig_show = plt.show
def _show_and_save(*a, **k):
    name = next(_FIG_NAMES, None) or f'fig_extra_{int(time.time() * 1000) % 100000}'
    plt.gcf().savefig(f'{OUT}/{name}.png', dpi=150, bbox_inches='tight')
    return _orig_show(*a, **k)
plt.show = _show_and_save
''')
cells.insert(4, hook)

# 3) 맨 끝에 저장·추가 분석 셀
extra = [
    md('''
## [추가 — mp 팀] 발표 재료 저장 (교수님 코드 아님)
결과표·기준선·24개 조합의 test 예측을 `results_nb_mp/` 에 저장하고, Descriptor parity 와 XGBoost SHAP / RF 중요도(§9 선택 과제)를 계산한다.
'''),
    code(r'''
# 위 "틀린 분자 5개" 그림 저장 (scaffold / MorganFP / XGBoost)
img = Draw.MolsToGridImage([df['mol'][te[i]] for i in worst], molsPerRow=5, subImgSize=(220, 180),
                           legends=[f'true {yt[i]:.1f} / pred {yp[i]:.1f} K' for i in worst])
path = f'{OUT}/fig07_worst5_XGBoost_MorganFP_scaffold.png'
if hasattr(img, 'save'):
    img.save(path)
else:
    open(path, 'wb').write(img.data)
print('saved', path)
'''),
    code(r'''
import json
res.to_csv(f'{OUT}/results_mp.csv', index=False)
table_out = table.sort_values('random')
table_out.to_csv(f'{OUT}/mae_table_mp.csv')
info = {'PROP': PROP, 'MAX_N': MAX_N, 'SEED': SEED, 'n_rows': int(len(df)), 'unit': unit,
        'baseline_MAE': {k: float(v) for k, v in base.items()},
        'n_desc': int(X_desc.shape[1]), 'fp_bits_mean': float(X_fp.sum(1).mean()),
        'fp_bits_min': int(X_fp.sum(1).min()), 'fp_bits_max': int(X_fp.sum(1).max()),
        'splits': {k: {'train': int(len(tr)), 'test': int(len(te))} for k, (tr, te) in splits.items()},
        'n_scaffolds_total': int(len(set(MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in df['mol']))),
        'y_describe': {k: float(v) for k, v in df['value'].describe().items()}}
# scaffold split 의 test 골격 수, train 과 겹치는 골격 수 (14쪽: test 는 처음 보는 골격인가)
scaf = [MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False) for m in df['mol']]
for k, (tr, te) in splits.items():
    s_tr, s_te = set(scaf[i] for i in tr), set(scaf[i] for i in te)
    info['splits'][k].update(n_scaffold_train=len(s_tr), n_scaffold_test=len(s_te), n_scaffold_overlap=len(s_tr & s_te),
                             frac_test_mols_with_seen_scaffold=float(np.mean([scaf[i] in s_tr for i in te])))
json.dump(info, open(f'{OUT}/run_info.json', 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
for (split_, feat_, model_), (yt_, yp_) in preds.items():
    tr_, te_ = splits[split_]
    pd.DataFrame({'smiles': df['smiles'].values[te_], 'y_true': yt_, 'y_pred': yp_, 'train_mean': y[tr_].mean()}
                 ).to_csv(f'{OUT}/pred_test_{model_}_{feat_}_{split_}.csv', index=False)
print(json.dumps(info, indent=2, ensure_ascii=False))
'''),
    code(r'''
# Descriptor 표현의 parity (본문은 MorganFP 만 그렸다)
parity_grid('random', 'Descriptor')
parity_grid('scaffold', 'Descriptor')
'''),
    code(r'''
# §9 선택 과제: RF feature_importances_ 상위 10 descriptor + XGBoost SHAP (두 분할, Descriptor 표현)
# 루프 안의 모델은 버려지므로 같은 설정·같은 seed 로 다시 fit 한다 (결정적 → 결과표의 모델과 같다).
import shap
xai = {}
for split_name in splits:
    tr, te = splits[split_name]
    rf = get_models()['RandomForest'].fit(X_desc[tr], y[tr])
    imp = pd.Series(rf.feature_importances_, index=desc_names).sort_values(ascending=False)
    imp.head(30).to_csv(f'{OUT}/rf_importance_Descriptor_{split_name}.csv', header=['importance'])
    xgb = get_models()['XGBoost'].fit(X_desc[tr], y[tr])
    print(split_name, 'refit XGBoost vs 결과표 예측 최대 차이 [K]:', float(np.abs(xgb.predict(X_desc[te]) - preds[(split_name, 'Descriptor', 'XGBoost')][1]).max()))
    sv = shap.TreeExplainer(xgb).shap_values(X_desc[te])
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(-mean_abs)[:20]
    rows_x = []
    for j in order:
        xj, sj = X_desc[te][:, j], sv[:, j]
        corr = np.corrcoef(xj, sj)[0, 1] if xj.std() > 0 and sj.std() > 0 else np.nan
        rows_x.append(dict(feature=desc_names[j], mean_abs_shap_K=mean_abs[j], corr_value_vs_shap=corr,
                           direction=('값 커지면 mp ↑' if corr > 0.2 else '값 커지면 mp ↓' if corr < -0.2 else '단조적이지 않음')))
    pd.DataFrame(rows_x).to_csv(f'{OUT}/shap_top20_XGBoost_Descriptor_{split_name}.csv', index=False)
    xai[split_name] = (imp.head(10), pd.DataFrame(rows_x).head(10))
    plt.figure()
    shap.summary_plot(sv, X_desc[te], feature_names=desc_names, max_display=15, show=False)
    plt.title(f'XGBoost + Descriptor, {split_name} test (n={len(te)}), SHAP [K]')
    plt.gcf().savefig(f'{OUT}/fig08_shap_beeswarm_XGBoost_Descriptor_{split_name}.png', dpi=150, bbox_inches='tight'); plt.close()
    plt.figure()
    shap.summary_plot(sv, X_desc[te], feature_names=desc_names, max_display=15, plot_type='bar', show=False)
    plt.title(f'XGBoost + Descriptor, {split_name} test, mean |SHAP| [K]')
    plt.gcf().savefig(f'{OUT}/fig09_shap_bar_XGBoost_Descriptor_{split_name}.png', dpi=150, bbox_inches='tight'); plt.close()
for split_name, (imp10, sh10) in xai.items():
    print(f'\n=== {split_name}: RF importance top 10 ===\n{imp10.round(4).to_string()}')
    print(f'\n=== {split_name}: XGBoost SHAP top 10 ===\n{sh10.round(3).to_string(index=False)}')
'''),
]
cells.extend(extra)
json.dump(nb, open(DST, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('written', DST, 'cells:', len(cells))
