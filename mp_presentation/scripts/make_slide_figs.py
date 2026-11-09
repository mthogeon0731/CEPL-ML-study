"""[추가] 발표 슬라이드용 그림. 수치는 모두 results_nb_mp/, results/ 의 원본 파일에서 읽는다 (재계산 없음)."""
import sys, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

NB, RES, OUT5, OUT = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'Noto Sans CJK KR', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.edgecolor': '#9a9994', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.grid': False,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.dpi': 170})
BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#9a9994'
TXT, TXT2 = '#0b0b0b', '#52514e'

res = pd.read_csv(f'{NB}/results_mp.csv')
info = json.load(open(f'{NB}/run_info.json'))
base = info['baseline_MAE']
gin = pd.read_csv(f'{RES}/summary.csv')
gin_r = gin[(gin.split == 'random')].iloc[0]
fair = pd.read_csv(f'{OUT}/../../results/fair_compare_gin_split.csv') if os.path.exists(f'{OUT}/../../results/fair_compare_gin_split.csv') else None

# ---------------------------------------------------------------- 1. 결과표 (히트맵)
models = ['XGBoost', 'RandomForest', 'MLP', 'Ridge', 'SVR', 'kNN']
cols = [('Descriptor', 'random'), ('Descriptor', 'scaffold'), ('MorganFP', 'random'), ('MorganFP', 'scaffold')]
col_lab = ['기술자 217\nrandom', '기술자 217\nscaffold', 'Morgan 2048\nrandom', 'Morgan 2048\nscaffold']
M = np.array([[res[(res.model == m) & (res.feature == f) & (res.split == s)].MAE.iloc[0] for f, s in cols] for m in models])
fig, ax = plt.subplots(figsize=(8.6, 4.6))
vmin, vmax = 25, 100
cmap = matplotlib.colormaps['Oranges']
for i in range(len(models)):
    for j in range(4):
        v = M[i, j]; b = base[cols[j][1]]
        c = cmap(0.1 + 0.75 * min((v - vmin) / (vmax - vmin), 1.0))
        ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=c, edgecolor='white', lw=2))
        best = v == M[:, j].min()
        txt = f'{v:.1f}' + (' ×' if v > b else '')
        ax.text(j + 0.5, i + 0.5, txt, ha='center', va='center', fontsize=12.5,
                fontweight='bold' if best else 'normal', color=TXT if v < 70 else 'white')
# 기준선 행
for j in range(4):
    b = base[cols[j][1]]
    ax.add_patch(plt.Rectangle((j, 6), 1, 1, facecolor='#f1f0ec', edgecolor='white', lw=2))
    ax.text(j + 0.5, 6.5, f'{b:.1f}', ha='center', va='center', fontsize=12, color=TXT2)
ax.set_xlim(0, 4); ax.set_ylim(7, 0)
ax.set_xticks(np.arange(4) + 0.5); ax.set_xticklabels(col_lab, fontsize=10.5)
ax.set_yticks(np.arange(7) + 0.5); ax.set_yticklabels(models + ['기준선 (train 평균)'], fontsize=11)
ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
ax.xaxis.tick_top()
ax.set_title('test MAE [K] — 표본 5,000 (seed 42), 80/20, 튜닝 없음 · 굵게 = 열 최저 · × = 기준선보다 나쁨',
             fontsize=10.5, color=TXT2, pad=46, loc='left')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_S4_results_table.png'); plt.close(fig)

# ---------------------------------------------------------------- 2. parity: XGBoost + 기술자, random vs scaffold (+ GIN)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
panels = [('XGBoost + 기술자 · random 80/20', pd.read_csv(f'{NB}/pred_test_XGBoost_Descriptor_random.csv'), BLUE),
          ('XGBoost + 기술자 · scaffold 80/20', pd.read_csv(f'{NB}/pred_test_XGBoost_Descriptor_scaffold.csv'), ORANGE),
          ('GIN · random 80/10/10 (test 500)', pd.read_csv(f'{RES}/mp_GNN-GIN_mean_random_test_pred.csv').rename(columns={'y': 'y_true', 'pred': 'y_pred'}), AQUA)]
for ax, (title, d, c) in zip(axes, panels):
    yt, yp = d.y_true.values, d.y_pred.values
    mae = np.mean(np.abs(yt - yp)); rmse = np.sqrt(np.mean((yt - yp) ** 2))
    ax.plot([60, 800], [60, 800], color=GREY, lw=1)
    ax.scatter(yt, yp, s=9, alpha=0.45, color=c, edgecolors='none')
    ax.set_xlim(60, 800); ax.set_ylim(60, 800); ax.set_aspect('equal')
    ax.set_xlabel('측정 mp [K]'); ax.set_ylabel('예측 mp [K]')
    ax.set_title(title, fontsize=11, loc='left')
    ax.text(0.04, 0.96, f'MAE {mae:.1f} K\nRMSE {rmse:.1f} K\nn = {len(yt):,}', transform=ax.transAxes, va='top', fontsize=10.5, color=TXT)
    ax.axvspan(500, 800, color='#f1f0ec', zorder=0); ax.axvspan(60, 250, color='#f1f0ec', zorder=0)
    ax.set_xticks([100, 300, 500, 700]); ax.set_yticks([100, 300, 500, 700])
axes[0].text(640, 90, '> 500 K\n과소 예측', fontsize=9, color=TXT2, ha='center')
axes[0].text(155, 430, '< 250 K\n과대 예측', fontsize=9, color=TXT2, ha='center')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_S5_parity_xgb_gin.png'); plt.close(fig)

# ---------------------------------------------------------------- 3. 구간별 MAE (XGB random / scaffold / GIN)
def by_range(path):
    d = pd.read_csv(path, index_col=0)
    d.index = [float(str(i).split(',')[1].strip(' ]')) for i in d.index]   # 구간 상한
    return d.sort_index()
tabs = {'XGBoost + 기술자 random': (by_range(f'{OUT5}/xgb_descriptor_random/error_by_range.csv'), BLUE),
        'XGBoost + 기술자 scaffold': (by_range(f'{OUT5}/xgb_descriptor_scaffold/error_by_range.csv'), ORANGE),
        'GIN random': (by_range(f'{OUT5}/gin_random/error_by_range.csv'), AQUA)}
bins = [250, 300, 350, 400, 450, 500, 550, 600]
labels = ['≤250', '250–300', '300–350', '350–400', '400–450', '450–500', '500–550', '550–600']
fig, ax = plt.subplots(figsize=(10, 4.2))
w = 0.27
for k, (name, (d, c)) in enumerate(tabs.items()):
    vals, ns = [], []
    for b in bins:
        sub = d[d.index <= b] if b == 250 else d[(d.index == b)]
        n = sub.n.sum(); mae = (sub.MAE * sub.n).sum() / n if n else np.nan
        vals.append(mae); ns.append(int(n))
    x = np.arange(len(bins)) + (k - 1) * w
    ax.bar(x, vals, w * 0.92, color=c, label=name, edgecolor='white', lw=1)
    for xi, v, n in zip(x, vals, ns):
        if not np.isnan(v):
            ax.text(xi, v + 1.2, f'{v:.0f}', ha='center', fontsize=8.5, color=TXT)
            ax.text(xi, -5, f'{n}', ha='center', va='top', fontsize=7.5, color=c)
ax.set_xticks(np.arange(len(bins))); ax.set_xticklabels(labels); ax.tick_params(axis='x', pad=20)
ax.set_xlabel('측정 mp 구간 [K]   (막대 아래 작은 숫자 = 그 구간의 test 분자 수)', labelpad=8)
ax.set_ylabel('test MAE [K]'); ax.set_ylim(0, 95)
ax.legend(frameon=False, loc='upper center', ncol=3, fontsize=10)
ax.axhline(0, color=GREY, lw=0.8)
ax.set_title('양 끝 구간일수록, 데이터가 적을수록 오차가 크다 — 세 모델 모두', fontsize=11.5, loc='left', color=TXT)
fig.tight_layout(); fig.savefig(f'{OUT}/fig_S6_error_by_range.png'); plt.close(fig)

# ---------------------------------------------------------------- 4. SHAP top 10 (scaffold, 방향 색)
sh = pd.read_csv(f'{NB}/shap_top20_XGBoost_Descriptor_scaffold.csv').head(10)
sh_r = pd.read_csv(f'{NB}/shap_top20_XGBoost_Descriptor_random.csv').set_index('feature')
fig, ax = plt.subplots(figsize=(8.4, 4.4))
yy = np.arange(len(sh))[::-1]
colors = [BLUE if c > 0 else ORANGE for c in sh.corr_value_vs_shap]
ax.barh(yy, sh.mean_abs_shap_K, color=colors, height=0.62)
for y_, v, f in zip(yy, sh.mean_abs_shap_K, sh.feature):
    r = sh_r.mean_abs_shap_K.get(f, np.nan)
    ax.text(v + 0.3, y_, f'{v:.1f} K' + (f'  (random {r:.1f})' if not np.isnan(r) else ''), va='center', fontsize=9.5, color=TXT2)
ax.set_yticks(yy); ax.set_yticklabels(sh.feature, fontsize=11)
ax.set_xlabel('mean |SHAP| [K] — 그 기술자가 예측을 평균 몇 K 움직였나 (XGBoost + 기술자, scaffold test 1,000)')
ax.set_xlim(0, 27)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=BLUE, label='값이 커지면 mp ↑'), Patch(color=ORANGE, label='값이 커지면 mp ↓')], frameon=False, loc='lower right')
fig.tight_layout(); fig.savefig(f'{OUT}/fig_S6_shap_top10.png'); plt.close(fig)

# ---------------------------------------------------------------- 5. 틀린 분자 (XGB scaffold 3 + GIN 3) — RDKit 그림 + matplotlib 한글 캡션
from rdkit import Chem
from rdkit.Chem import Draw
wx = pd.read_csv(f'{OUT5}/xgb_descriptor_scaffold/worst_molecules.csv').head(3)
wg = pd.read_csv(f'{OUT5}/gin_random/worst_molecules.csv').head(3)
names_x = ['Mirex (C10Cl12) — 값은 문헌과 일치 → 모델 한계(대칭 케이지, 학습 범위 밖)',
           '다환 방향족 디케톤 — 평면·강직 → 패킹 효과를 기술자가 못 담음',
           '헥사페닐다이실레인 — 대칭 큰 분자, 과소 예측']
names_g = ['엔다이아민 (비고리, N 4개) — 극성 원자 많음 → 과대 예측',
           '2-클로로-N,N-다이에틸에틸아민 — 484 K는 염산염 값 의심',
           '바이닐 아세테이트 — 값은 맞음(−96 °C), 원자 6개 분자를 388 K로']
fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4))
for row, (tag, w, names, c) in enumerate((('XGBoost + 기술자 · scaffold', wx, names_x, ORANGE), ('GIN · random', wg, names_g, AQUA))):
    for col, ((_, r), nm) in enumerate(zip(w.iterrows(), names)):
        ax = axes[row, col]
        im = Draw.MolToImage(Chem.MolFromSmiles(r.smiles), size=(420, 260))
        ax.imshow(im); ax.axis('off')
        ax.set_title((f'[{tag}]   ' if col == 0 else '') + f'측정 {r.y_true:.0f} → 예측 {r.y_pred:.0f} K  ({r.err:+.0f} K)', fontsize=10.5, color=c, loc='left', fontweight='bold')
        ax.text(0.0, -0.04, nm, transform=ax.transAxes, fontsize=9, color=TXT2, va='top', wrap=True)
fig.tight_layout(h_pad=2.2); fig.savefig(f'{OUT}/fig_S6_worst_molecules.png'); plt.close(fig)

# ---------------------------------------------------------------- 6. 같은 test 500 에서 순위 (fair compare)
if fair is not None:
    f = fair[fair.model != 'baseline(train mean)'].copy()
    f['label'] = [f'{m} + {"기술자" if ft == "Descriptor" else "Morgan" if ft == "MorganFP" else "그래프"}' if m != 'GIN (train.py)' else 'GIN (그래프)' for m, ft in zip(f.model, f.feature)]
    f = f.sort_values('MAE')
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    yy = np.arange(len(f))[::-1]
    cols_ = [AQUA if 'GIN' in l else BLUE if '기술자' in l else ORANGE for l in f.label]
    ax.barh(yy, f.MAE, color=cols_, height=0.62)
    for y_, v in zip(yy, f.MAE):
        ax.text(v + 0.8, y_, f'{v:.1f}', va='center', fontsize=9.5, color=TXT2)
    ax.axvline(fair[fair.model == 'baseline(train mean)'].MAE.iloc[0], color=GREY, lw=1, ls='--')
    ax.text(fair[fair.model == 'baseline(train mean)'].MAE.iloc[0] + 0.8, yy[0] + 0.1, f'기준선 (train 평균) {fair[fair.model == "baseline(train mean)"].MAE.iloc[0]:.1f} K', fontsize=9, color=TXT2, va='center')
    ax.set_yticks(yy); ax.set_yticklabels(f.label, fontsize=10)
    ax.set_xlabel('test MAE [K] — 같은 train 4,000 / 같은 test 500 (train.py random 80/10/10 분할 재현)')
    ax.set_xlim(0, 100)
    ax.legend(handles=[Patch(color=BLUE, label='기술자 217'), Patch(color=ORANGE, label='Morgan 2048'), Patch(color=AQUA, label='그래프 (GIN)')], frameon=False, loc='center right')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig_S7_same_split_ranking.png'); plt.close(fig)

print('figs written to', OUT, os.listdir(OUT))
