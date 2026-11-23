"""랩미팅용 그림 — mp(녹는점) MAE 개선 실험 (Claude Science 2026-11-20~22 결과).
입력: PC mp_mae_push 폴더에서 가져온 예측·표 파일 (재학습 없음). 출력: OUT/*.png, figs_all.pdf
"""
import os, sys, re
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages

ROOT = "/mnt/user-data/uploads/CEPL ML/mp_mae_push"
WK2 = "/mnt/user-data/uploads/CEPL ML/261109_Week2_Practice/results"
OUT = "/mnt/user-data/outputs/figs_labmeeting"
os.makedirs(OUT, exist_ok=True)

# ---------- 스타일 (dataviz reference palette, light) ----------
KR = [f for f in font_manager.findSystemFonts() if "NotoSansCJK-Regular" in f or "NotoSansCJK-Medium" in f]
for f in KR:
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": "Noto Sans CJK JP", "axes.unicode_minus": False,
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.labelsize": 11,
    "axes.edgecolor": "#8a8985", "axes.linewidth": 0.8, "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b",
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "legend.frameon": False, "lines.linewidth": 2, "axes.axisbelow": True,
})
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GRAY, GRAY_L, INK2, INK3 = "#8a8985", "#c3c2b7", "#52514e", "#8a8985"
DARKBLUE = "#1c5cab"
# 엔티티 → 고정 색 (모든 그림에서 같음)
ENT = {"LGB F0+F5": BLUE, "GIN 하이브리드": ORANGE, "Yalkowsky LGB": AQUA,
       "XGB + 특허 데이터": YELLOW, "XGB F0+F1+O3": MAGENTA}
rng = np.random.default_rng(0)
FIGS = []


def save(fig, name, caption):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    FIGS.append((name, caption, fig))
    print("saved", name)


def load(rel):
    return pd.read_csv(os.path.join(ROOT, rel))


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def boot_idx(n, B=2000):
    return rng.integers(0, n, size=(B, n))


# ---------- 데이터 ----------
P3 = "phase3"
names = {  # 표시 이름: 파일 접두어
    "기준선 XGB (F0)": "baseline_xgb600_F0",
    "C: XGB + F5(LLM 규칙)": "C_xgb600_F0F5_M2",
    "GIN 하이브리드": "gin_hybrid_seedavg",
    "Yalkowsky LGB": "h06_yalk_F0F3F5",
    "LGB F0+F3+F5": "lgb_r1_F0F3F5",
    "B: LGB + GIN 50:50": "B_fixed_mean_best_GIN",
    "A: 가중 앙상블": "A_wavg_mse_noO2",
}
test = {}
for split in ["random", "scaffold"]:
    base = load(f"{P3}/baseline_xgb600_F0_{split}_test.csv")[["row", "y"]]
    for disp, pre in names.items():
        d = load(f"{P3}/{pre}_{split}_test.csv")[["row", "y", "pred"]]
        m = base.merge(d, on=["row"], suffixes=("", "_d"))
        assert len(m) == 1995 and np.allclose(m.y, m.y_d)
        test[(split, disp)] = m[["row", "y", "pred"]].reset_index(drop=True)
# random 기준선은 원본 예측 파일(27.4995 K) 사용: 순서·라벨이 test 와 같음
orig = pd.read_csv(os.path.join(WK2, "mp_XGBoost-Descriptor_random_test_pred.csv")).tail(1995).reset_index(drop=True)
t0 = test[("random", "기준선 XGB (F0)")]
assert np.allclose(orig.y.values, t0.y.values)
test[("random", "기준선 XGB (F0)")] = pd.DataFrame({"row": t0.row, "y": orig.y.values, "pred": orig.pred.values})
for split in ["random", "scaffold"]:
    print(split, {k[1]: round(mae(v.y, v.pred), 3) for k, v in test.items() if k[0] == split})

# =====================================================================
# F01 전체 파이프라인 도식
# =====================================================================
def box(ax, x, y, w, h, text, fc="#f2f1ed", ec=GRAY, fs=10.5, bold=False, color="#0b0b0b"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=color, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, color=INK2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(13, 6.6))
ax.set_xlim(0, 13); ax.set_ylim(0.5, 7.2); ax.axis("off")
box(ax, 0.2, 5.6, 2.4, 1.2, "[0] 데이터\nSMILES + 녹는점(K)\n19,945개", bold=False)
box(ax, 0.2, 3.9, 2.4, 1.2, "고정 분할 80:10:10\ntrain 15,955\nval 1,995 · test 1,995")
arrow(ax, 1.4, 5.6, 1.4, 5.1)
box(ax, 3.2, 4.4, 2.6, 2.4,
    "[1] 입력 만들기\n\nF0  RDKit 기술자 217\nF3  물리·결정 39 (3D 대칭수)\nF5  LLM 규칙 101\n그래프  원자·결합", fs=10)
arrow(ax, 2.6, 4.5, 3.2, 5.4)
box(ax, 3.2, 2.5, 2.6, 1.4, "[2] 튜닝 (train만)\nOptuna → LightGBM 설정\nlr 0.013 · 잎 68 · Huber", fc="#eef4fc", ec=BLUE)
arrow(ax, 4.5, 4.4, 4.5, 3.9)
members = [("① Yalkowsky LGB\nF0+F3+F5 → ΔHm ÷ ΔSm", AQUA),
           ("② GIN 하이브리드\n그래프 + F0", ORANGE),
           ("③ LGB  F0+F5", BLUE),
           ("④ XGBoost  F0+F1+O3", MAGENTA)]
ys = [5.9, 4.6, 3.3, 2.0]
for y in (ys[0], ys[2]):
    arrow(ax, 5.8, 3.2, 6.6, y + 0.3, color=BLUE)
ax.text(9.7, 2.3, "파랑 화살표 = 튜닝 설정 사용 (①·③)\n회색 화살표 = 입력 특징", fontsize=9.2, color=INK2)
ax.text(7.8, 6.95, "[3] 멤버 4개를 각각 따로 학습 (seed 3개 평균)", ha="center", fontsize=11, fontweight="bold")
for (t, c), y in zip(members, ys):
    box(ax, 6.6, y, 2.4, 0.95, t, fc="#ffffff", ec=c, fs=10)
    arrow(ax, 5.8, 5.6, 6.6, y + 0.47, color=GRAY_L)
    arrow(ax, 9.0, y + 0.47, 9.7, 4.2)
box(ax, 9.7, 3.4, 1.6, 1.6, "[4]–[5]\nval 예측 4개\n→ 가중치\n0.47 · 0.39\n0.12 · 0.02", fc="#f4f1fc", ec=VIOLET, fs=9.8)
arrow(ax, 11.3, 4.2, 11.6, 4.2)
box(ax, 11.6, 3.5, 1.3, 1.4, "[6] test\n가중 평균\n→ Tm\n24.75 K", fc="#eef4fc", ec=DARKBLUE, fs=10, bold=True)
ax.text(0.2, 0.9, "· 4개 모델은 서로 연결되지 않음: 각자 Tm을 예측하고 마지막에 숫자만 가중 평균\n"
        "· 튜닝 결과(설정값)는 ①·③에 그대로 사용, ② GIN은 수업 설정, ④는 XGBoost용 튜닝 따로\n"
        "· scaffold 분할은 멤버가 다름: GIN 0.37 · XGB+특허 데이터 0.32 · Yalkowsky 0.31",
        fontsize=10, color=INK2, va="bottom", linespacing=1.6)
ax.set_title("mp 예측 파이프라인 — 입력 → 튜닝 → 멤버 4개 → 가중 앙상블", loc="left", fontsize=14)
save(fig, "F01_pipeline.png", "전체 순서. 멤버 4개는 독립적으로 학습되고, validation에서 정한 가중치로 예측값만 평균한다 (random 분할 기준).")

# =====================================================================
# F02 GIN 하이브리드 구조
# =====================================================================
fig, ax = plt.subplots(figsize=(13, 5.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")
box(ax, 0.2, 3.0, 2.0, 1.3, "분자 그래프\n원자 147특징\n결합 16특징")
box(ax, 2.8, 3.0, 2.3, 1.3, "GIN 5층\n(이웃 정보 섞기 ×5)\nhidden 128", fc="#fdf0ea", ec=ORANGE)
box(ax, 5.7, 3.0, 1.9, 1.3, "mean readout\n→ 분자 벡터\n64개", fc="#fdf0ea", ec=ORANGE)
box(ax, 0.2, 0.9, 4.9, 1.1, "RDKit 기술자 217개 (F0, train 기준 표준화)", fc="#eef4fc", ec=BLUE)
box(ax, 8.2, 1.9, 1.6, 1.4, "이어 붙임\n64 + 217\n= 281", fc="#ffffff", ec=GRAY)
box(ax, 10.3, 1.9, 1.5, 1.4, "MLP\n281→256\n→128→1", fc="#ffffff", ec=GRAY)
box(ax, 12.1, 2.1, 0.8, 1.0, "Tm\n(K)", bold=True)
arrow(ax, 2.2, 3.65, 2.8, 3.65); arrow(ax, 5.1, 3.65, 5.7, 3.65)
arrow(ax, 7.6, 3.65, 8.2, 2.9); arrow(ax, 5.1, 1.45, 8.2, 2.3)
arrow(ax, 9.8, 2.6, 10.3, 2.6); arrow(ax, 11.8, 2.6, 12.1, 2.6)
ax.text(2.8, 4.65, "수업 GIN: 여기서 바로 → Tm  (그래프만, random 30.5 / scaffold 45.0 K, val seed 1)",
        fontsize=10, color=INK2)
ax.text(8.2, 4.1, "하이브리드: 벡터 옆에 기술자를 붙여 함께 학습 (모델 1개)\n"
        "dropout 0.3 · AdamW 1e-3 · 조기 종료 · seed 3개 평균",
        fontsize=10, color=INK2, linespacing=1.5)
ax.set_title("GIN 하이브리드 — 그래프가 배운 벡터 + 사람이 계산한 기술자", loc="left", fontsize=14)
save(fig, "F02_gin_hybrid_architecture.png", "GIN 하이브리드 구조. 앙상블이 아니라 입력이 두 종류인 모델 하나다.")

# =====================================================================
# F03 GIN 학습 곡선 (T2 트랙 원본 그림을 그대로 복사)
# =====================================================================
import shutil
shutil.copy(os.path.join(ROOT, "phase2/T2_gin/T2_learning_curves.png"), os.path.join(OUT, "F03_gin_learning_curves.png"))
img = plt.imread(os.path.join(OUT, "F03_gin_learning_curves.png"))
fig, ax = plt.subplots(figsize=(11, 8.1)); ax.imshow(img); ax.axis("off")
FIGS.append(("F03_gin_learning_curves.png",
             "GIN 학습 곡선 (Claude Science T2 트랙 원본 그림). 회색 점선 = 그래프만, 색 = 하이브리드 seed 0–2, 점선 = train 손실, 원 = 복원된 최적 epoch. 세로축 = train 내부 검증 손실(위)·MAE(아래).", fig))
print("saved F03 (copy)")

# =====================================================================
# F04 그래프만 vs 하이브리드 parity (validation, seed 1)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 9.6), sharex=True, sharey=True)
for i, split in enumerate(["random", "scaffold"]):
    for j, (lab, f, c) in enumerate([("그래프만 GIN", f"phase2/T2_gin/T2_graphonly_{split}_s1_val.csv", GRAY),
                                      ("GIN 하이브리드", f"phase2/T2_gin/T2_hyb_{split}_s1_val.csv", ORANGE)]):
        d = load(f); ax = axes[i, j]
        ax.scatter(d.y, d.pred, s=9, alpha=0.35, color=c, linewidths=0)
        ax.plot([50, 820], [50, 820], color="#0b0b0b", lw=1)
        ax.set_xlim(50, 820); ax.set_ylim(50, 820); ax.set_aspect("equal")
        ax.text(0.04, 0.95, f"MAE {mae(d.y, d.pred):.1f} K · n {len(d):,}", transform=ax.transAxes,
                va="top", fontsize=10.5)
        if i == 0: ax.set_title(lab, loc="left")
        if j == 0: ax.set_ylabel(f"{split}\n예측 녹는점 (K)")
        if i == 1: ax.set_xlabel("측정 녹는점 (K)")
fig.suptitle("기술자를 붙이면 GIN이 좋아진다 — validation, seed 1", x=0.06, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "F04_gin_graphonly_vs_hybrid_parity.png", "같은 GIN(seed 1)에 기술자 217개를 붙였을 때의 차이. validation 예측.")

# =====================================================================
# F05 test MAE 막대 + 95 % CI (모델별)
# =====================================================================
order = list(names.keys())
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
for k, split in enumerate(["random", "scaffold"]):
    ax = axes[k]; vals, lo, hi = [], [], []
    for disp in order:
        d = test[(split, disp)]; e = np.abs(d.y.values - d.pred.values)
        bs = e[boot_idx(len(e))].mean(1)
        vals.append(e.mean()); lo.append(np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5))
    vals, lo, hi = map(np.array, (vals, lo, hi))
    cols = [GRAY] + [BLUE] * 4 + [DARKBLUE] * 2
    yv = np.arange(len(order))[::-1]
    ax.axvline(vals[0], color=GRAY, lw=1, ls="--")
    for y_, v, l, h, c in zip(yv, vals, lo, hi, cols):
        ax.plot([l, h], [y_, y_], color=c, lw=2.4, solid_capstyle="round")
        ax.plot(v, y_, "o", ms=10, color=c, mec="#fcfcfb", mew=2)
        ax.text(hi.max() + 0.35, y_, f"{v:.2f}", va="center", fontsize=10.5)
    ax.set_yticks(yv); ax.set_yticklabels(order)
    ax.set_xlim(lo.min() - 0.8, hi.max() + 1.6); ax.grid(axis="y", visible=False)
    ax.set_title(f"{split} test (n 1,995)", loc="left"); ax.set_xlabel("MAE (K), 낮을수록 좋음")
fig.suptitle("test MAE — 기준선 대비 앙상블 A·B가 가장 낮다 (선 = 부트스트랩 95 % CI, 점선 = 기준선)", x=0.01, ha="left",
             fontsize=14, fontweight="bold")
fig.text(0.01, -0.02, "random 기준선 = 원본 예측 파일 27.50 K, scaffold 기준선 = 재적합 35.85 K.",
         fontsize=9.5, color=INK2)
fig.tight_layout()
save(fig, "F05_test_mae_bars.png", "모델별 test MAE와 95 % CI. 진한 파랑 = 앙상블, 회색 = 기준선.")

# =====================================================================
# F06 ΔMAE vs 기준선 (짝지은 부트스트랩)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
for k, split in enumerate(["random", "scaffold"]):
    ax = axes[k]; b = test[(split, "기준선 XGB (F0)")]; eb = np.abs(b.y.values - b.pred.values)
    idx = boot_idx(len(eb)); rows = []
    for disp in order[1:]:
        d = test[(split, disp)]; e = np.abs(d.y.values - d.pred.values); diff = e - eb
        bs = diff[idx].mean(1); rows.append((disp, diff.mean(), *np.percentile(bs, [2.5, 97.5])))
    yv = np.arange(len(rows))[::-1]
    for y_, (disp, m, l, h) in zip(yv, rows):
        c = DARKBLUE if disp.startswith(("A", "B")) else BLUE
        ax.plot([l, h], [y_, y_], color=c, lw=2.4, solid_capstyle="round")
        ax.plot(m, y_, "o", ms=9, color=c, mec="#fcfcfb", mew=2)
        ax.text(0.95, y_, f"{m:+.2f} [{l:+.2f}, {h:+.2f}]", va="center", fontsize=9.8, color=INK2)
    ax.axvline(0, color="#0b0b0b", lw=1)
    ax.set_yticks(yv); ax.set_yticklabels([r[0] for r in rows]); ax.grid(axis="y", visible=False)
    ax.set_xlim(-3.8, 3.2); ax.set_xlabel("ΔMAE vs 기준선 (K), 왼쪽 = 개선")
    ax.set_title(f"{split} test", loc="left")
fig.suptitle("같은 분자끼리 비교한 개선폭 — scaffold의 GIN 단독만 0을 넘는다 (짝지은 부트스트랩 2,000회)", x=0.01, ha="left",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "F06_delta_mae_forest.png", "기준선 대비 ΔMAE와 95 % CI. 같은 test 분자에서 짝지어 비교.")

# =====================================================================
# F07 parity: 기준선 vs A (test)
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 9.6), sharex=True, sharey=True)
for i, split in enumerate(["random", "scaffold"]):
    for j, (disp, c) in enumerate([("기준선 XGB (F0)", GRAY), ("A: 가중 앙상블", BLUE)]):
        d = test[(split, disp)]; ax = axes[i, j]
        ax.scatter(d.y, d.pred, s=9, alpha=0.35, color=c, linewidths=0)
        ax.plot([50, 820], [50, 820], color="#0b0b0b", lw=1)
        ax.set_xlim(50, 820); ax.set_ylim(50, 820); ax.set_aspect("equal")
        r2 = 1 - np.sum((d.y - d.pred) ** 2) / np.sum((d.y - d.y.mean()) ** 2)
        ax.text(0.04, 0.95, f"MAE {mae(d.y, d.pred):.2f} K · R² {r2:.3f}", transform=ax.transAxes,
                va="top", fontsize=10.5)
        if i == 0: ax.set_title(disp, loc="left")
        if j == 0: ax.set_ylabel(f"{split}\n예측 녹는점 (K)")
        if i == 1: ax.set_xlabel("측정 녹는점 (K)")
fig.suptitle("parity plot — test 1,995개, 기준선 vs 앙상블 A", x=0.06, ha="left", fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "F07_parity_baseline_vs_A.png", "test parity. 대각선 = 완벽한 예측. 고융점 쪽이 대각선 아래로 처지는 경향(평균 쪽 쏠림)이 남아 있다.")

# =====================================================================
# F08 MP 구간별 평균 잔차
# =====================================================================
bins = [0, 250, 350, 450, 550, 2000]; blab = ["<250", "250–350", "350–450", "450–550", ">550"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
for k, split in enumerate(["random", "scaffold"]):
    ax = axes[k]; ns = []
    for off, (disp, c) in zip([-0.14, 0.14], [("기준선 XGB (F0)", GRAY), ("A: 가중 앙상블", BLUE)]):
        d = test[(split, disp)]; r = (d.pred - d.y).values; cat = pd.cut(d.y, bins, labels=blab).values
        for xi, bl in enumerate(blab):
            rr = r[cat == bl]
            bs = rr[boot_idx(len(rr), 1000)].mean(1)
            m, l, h = rr.mean(), *np.percentile(bs, [2.5, 97.5])
            ax.plot([xi + off] * 2, [l, h], color=c, lw=2.2, solid_capstyle="round")
            ax.plot(xi + off, m, "o", ms=8.5, color=c, mec="#fcfcfb", mew=2,
                    label=disp if xi == 0 else None)
            if disp.startswith("A"):
                ns.append(len(rr))
    ax.axhline(0, color="#0b0b0b", lw=1)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{b}\nn {n}" for b, n in zip(blab, ns)]); ax.set_xlabel("측정 녹는점 구간 (K)")
    ax.set_ylim(-95, 80); ax.set_title(f"{split} test", loc="left"); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("평균 잔차 = 예측 − 측정 (K)")
axes[0].legend(loc="upper right")
fig.suptitle("끝 구간 쏠림은 줄었지만 남아 있다 — 저융점은 과대, 고융점은 과소 예측", x=0.01, ha="left",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "F08_residual_by_mp_bin.png", "측정 MP 구간별 평균 잔차와 95 % CI. 0 위 = 과대예측, 0 아래 = 과소예측.")

# =====================================================================
# F09 멤버 간 오차 상관 히트맵 (validation)
# =====================================================================
keep = {"baseline_xgb600_F0": "기준선 XGB", "lgb_r1_F0": "LGB F0", "xgb_r1_F0": "XGB 튜닝 F0",
        "lgb_r1_F0F5": "LGB F0+F5", "lgb_r1_F0F3F5": "LGB F0+F3+F5", "h06_yalk_F0F3F5": "Yalkowsky LGB",
        "xgb600_F0F5_M2": "XGB + F5", "gin_hybrid_seedavg": "GIN 하이브리드"}
from matplotlib.colors import LinearSegmentedColormap
seq = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#104281"])
fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), gridspec_kw={"wspace": 0.05})
for k, split in enumerate(["random", "scaffold"]):
    C = load(f"phase2/ENS_ensemble/error_corr_{split}.csv").set_index(load(f"phase2/ENS_ensemble/error_corr_{split}.csv").columns[0])
    ks = [c for c in keep if c in C.columns]
    M = C.loc[ks, ks].values; ax = axes[k]
    im = ax.imshow(M, cmap=seq, vmin=0.85, vmax=1.0)
    for a in range(len(ks)):
        for b in range(len(ks)):
            ax.text(b, a, f"{M[a, b]:.2f}", ha="center", va="center", fontsize=8.8,
                    color="#ffffff" if M[a, b] > 0.95 else "#0b0b0b")
    ax.set_xticks(range(len(ks))); ax.set_yticks(range(len(ks)))
    ax.set_xticklabels([keep[c] for c in ks], rotation=40, ha="right")
    ax.set_yticklabels([keep[c] for c in ks] if k == 0 else [""] * len(ks))
    ax.grid(False); ax.set_title(f"{split} validation", loc="left")
    for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.02); cb.set_label("분자별 오차의 상관")
fig.suptitle("GIN만 다른 분자에서 틀린다 — 트리 모델끼리 0.94–0.99, GIN과는 0.87–0.90", x=0.01, ha="left",
             fontsize=14, fontweight="bold")
save(fig, "F09_error_correlation_heatmap.png", "멤버 모델 간 오차(예측 − 측정) 상관. 상관이 낮을수록 섞었을 때 오차가 상쇄된다. 색 하한 0.85.")

# =====================================================================
# F10 LGB vs GIN 오차 산점도 (test random)
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
for k, split in enumerate(["random", "scaffold"]):
    a = test[(split, "LGB F0+F3+F5")]; g = test[(split, "GIN 하이브리드")]
    ea, eg = (a.pred - a.y).values, (g.pred - g.y).values; ax = axes[k]
    hb = ax.hexbin(ea, eg, gridsize=45, extent=(-200, 200, -200, 200), cmap=seq, mincnt=1, bins="log", linewidths=0.2)
    ax.plot([-200, 200], [-200, 200], color="#0b0b0b", lw=1)
    ax.axhline(0, color=GRAY_L, lw=0.8); ax.axvline(0, color=GRAY_L, lw=0.8)
    r = np.corrcoef(ea, eg)[0, 1]
    ax.text(0.04, 0.95, f"상관 r = {r:.2f}\n대각선에서 먼 분자 = 한 모델만 크게 틀림", transform=ax.transAxes,
            va="top", fontsize=10.5)
    ax.set_xlabel("LGB F0+F3+F5 오차 (K)"); ax.set_ylabel("GIN 하이브리드 오차 (K)")
    ax.set_title(f"{split} test", loc="left"); ax.set_aspect("equal")
cb = fig.colorbar(hb, ax=axes, shrink=0.8, pad=0.02); cb.set_label("분자 수 (로그)")
fig.suptitle("왜 섞으면 좋아지나 — 두 모델이 틀리는 분자가 다르다", x=0.01, ha="left", fontsize=14, fontweight="bold")
save(fig, "F10_lgb_vs_gin_error_scatter.png", "같은 test 분자에서 두 모델의 오차. 대각선 위에 모이면 같이 틀린 것, 흩어질수록 평균으로 상쇄될 여지가 크다.")

# =====================================================================
# F11 앙상블 가중치 (분할별 누적 막대)
# =====================================================================
W = {"random": [("LGB F0+F5", 0.123188), ("GIN 하이브리드", 0.387661), ("Yalkowsky LGB", 0.469553), ("XGB F0+F1+O3", 0.019598)],
     "scaffold": [("GIN 하이브리드", 0.3722), ("Yalkowsky LGB", 0.3100), ("XGB + 특허 데이터", 0.3178)]}
fig, ax = plt.subplots(figsize=(12.5, 3.9))
for yi, split in enumerate(["scaffold", "random"]):
    left = 0
    for name, w in W[split]:
        ax.barh(yi, w, left=left, color=ENT[name], height=0.55, edgecolor="#fcfcfb", linewidth=2)
        if w > 0.06:
            ax.text(left + w / 2, yi, f"{name}\n{w:.2f}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if ENT[name] in (BLUE, ORANGE) else "#0b0b0b")
        else:
            ax.text(left + w, yi - 0.38, f"끝의 얇은 칸: {name} {w:.2f}", fontsize=9.5, ha="right", va="top", color=INK2)
        left += w
ax.set_yticks([0, 1]); ax.set_yticklabels(["scaffold", "random"]); ax.set_xlim(0, 1); ax.set_ylim(-0.45, 1.45)
ax.set_xlabel("가중치 (합 = 1)"); ax.grid(axis="y", visible=False)
ax.set_title("앙상블 A의 멤버와 가중치 — validation MSE가 최소가 되도록 분할마다 따로 정함", loc="left")
save(fig, "F11_ensemble_weights.png", "앙상블 A의 가중치. 가중치는 음수 불가, 합 1, validation에서만 결정.")

# =====================================================================
# F12 Optuna 수렴
# =====================================================================
tr = load("phase2/T1_boosting/optuna_trials.csv")
fig, ax = plt.subplots(figsize=(11, 5.2))
for study, lab, c in [("lgb_random_F0", "LightGBM (35회)", BLUE), ("xgb_random_F0_v2", "XGBoost (42회)", ORANGE)]:
    d = tr[(tr.study == study) & (tr.state == "COMPLETE")].sort_values("number")
    v = d.value.values; n = d.number.values
    ax.scatter(n, np.clip(v, None, 31.5), s=36, color=c, alpha=0.45, linewidths=0)
    ax.step(n, np.minimum.accumulate(v), where="post", color=c, lw=2.2, label=f"{lab} — 지금까지 최선")
    ax.text(n[-1] + 0.6, np.minimum.accumulate(v)[-1], f"{np.min(v):.2f}", va="center", fontsize=10, color=INK2)
ax.axhline(27.607, color=GRAY, lw=1.2, ls="--")
ax.text(12, 27.68, "기준선 XGB validation 27.61 K (참고)", fontsize=9.5, color=INK2)
ax.set_ylim(26.4, 31.6); ax.set_xlabel("시도 번호 (trial)"); ax.set_ylabel("train 내부 검증 MAE (K)")
ax.legend(loc="upper right")
ax.set_title("Optuna 튜닝 — 10번째 시도 안에 거의 수렴 (점 = 각 시도, 31.5 K 위는 잘라 표시)", loc="left")
save(fig, "F12_optuna_convergence.png", "Optuna 튜닝 경과. 세로축은 train 안 내부 분할 MAE라 validation 값과 직접 비교하지 않는다.")

# =====================================================================
# F13 학습 곡선 (데이터 양)
# =====================================================================
lc = load("phase1/tables/learning_curve.csv")
fig, ax = plt.subplots(figsize=(10.5, 5.6))
for split, c in [("random", BLUE), ("scaffold", ORANGE)]:
    for col, ls, lab in [("val_MAE", "-", "validation"), ("test_MAE", "--", "test")]:
        g = lc[lc.split == split].groupby("n_train")[col].agg(["mean", "min", "max"]).reset_index()
        ax.plot(g.n_train, g["mean"], ls=ls, color=c, marker="o", ms=7, mec="#fcfcfb", mew=1.5,
                label=f"{split} · {lab}")
        ax.fill_between(g.n_train, g["min"], g["max"], color=c, alpha=0.12, lw=0)
ax.set_xscale("log", base=2)
ax.set_xticks([1596, 3989, 7978, 15955]); ax.set_xticklabels(["1,596\n(10 %)", "3,989\n(25 %)", "7,978\n(50 %)", "15,955\n(100 %)"])
ax.set_xlabel("학습 분자 수 (로그 축)"); ax.set_ylabel("MAE (K)"); ax.legend(ncol=2, loc="upper right")
ax.set_title("데이터 2배당 random −1.4 ~ −2.2 K, scaffold −0.6 ~ −1.4 K — 기준선 XGB, seed 3개", loc="left")
save(fig, "F13_learning_curve.png", "고정 test에서 학습 데이터 양만 바꾼 학습 곡선. 띠 = seed 3개의 최소–최대.")

# =====================================================================
# F14 표현 비교: 자연어·LLM vs 기술자 (validation, XGBoost head)
# =====================================================================
reps = [("RDKit 217 + F5 (LLM 규칙)", -0.70, -1.06, -0.37, -0.85, -1.25, -0.41),
        ("F5 단독", 5.13, 4.15, 6.14, 4.15, 3.12, 5.22),
        ("템플릿 서술 → bge-small 임베딩", 8.04, 6.99, 9.06, 7.65, 6.57, 8.71),
        ("템플릿 서술 → MiniLM 임베딩", 9.43, 8.32, 10.58, 8.36, 7.20, 9.50),
        ("IUPAC 이름 → MiniLM 임베딩", 11.43, 10.09, 12.78, 13.41, 11.77, 14.91)]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for k, split in enumerate(["random", "scaffold"]):
    ax = axes[k]; yv = np.arange(len(reps))[::-1]
    for y_, r in zip(yv, reps):
        m, l, h = r[1 + 3 * k: 4 + 3 * k]; c = BLUE if m < 0 else GRAY
        ax.barh(y_, m, color=c, height=0.58, edgecolor="#fcfcfb", linewidth=2)
        ax.plot([l, h], [y_, y_], color="#0b0b0b", lw=1.2)
        ax.text(h + 0.3 if m > 0 else l - 0.3, y_, f"{m:+.2f}", va="center", ha="left" if m > 0 else "right", fontsize=10)
    ax.axvline(0, color="#0b0b0b", lw=1); ax.set_xlim(-3.5, 16.5)
    ax.set_yticks(yv); ax.set_yticklabels([r[0] for r in reps]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("ΔMAE vs RDKit 217 (K), 왼쪽 = 개선"); ax.set_title(f"{split} validation", loc="left")
fig.suptitle("자연어 표현은 모두 기술자보다 나쁨 — LLM은 '규칙 설계자'로 쓸 때만 도움 (XGBoost head)", x=0.01, ha="left",
             fontsize=14, fontweight="bold")
fig.text(0.01, -0.03, "LLM이 쓴 구조 서술(N3)은 템플릿보다도 +3.7 K 나쁨 (random, bge). 막대 끝 선 = 95 % CI.", fontsize=9.5, color=INK2)
fig.tight_layout()
save(fig, "F14_text_llm_representations.png", "같은 XGBoost에서 입력 표현만 바꿨을 때 RDKit 기술자 대비 ΔMAE (validation).")

# =====================================================================
# F15 위험–커버리지 (목표 C)
# =====================================================================
gc = load("phase3/goalC_curve_test.csv")
fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
for k, split in enumerate(["random", "scaffold"]):
    ax = axes[k]
    for cand, c, lab in [("A", BLUE, "A 가중 앙상블"), ("C", AQUA, "C XGB + F5")]:
        d = gc[(gc.split == split) & (gc.cand == cand)].sort_values("coverage")
        d = d[d.coverage >= 0.02]
        ax.plot(d.coverage * 100, d.MAE, color=c, label=lab)
    ax.axhline(10, color=RED, lw=1.2, ls="--"); ax.text(99, 10.6, "목표 10 K", ha="right", fontsize=10, color=INK2)
    if split == "random":
        ax.plot(6.8, 12.64, "o", ms=9, color=BLUE, mec="#fcfcfb", mew=2)
        ax.annotate("validation으로 고정한 기준\n상위 6.8 % → 12.6 K", xy=(6.8, 12.64), xytext=(18, 6),
                    fontsize=10, color=INK2, arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.8))
    ax.set_xlabel("예측을 내놓는 분자 비율 (%) — 신뢰도 높은 순"); ax.set_title(f"{split} test", loc="left")
    ax.set_xlim(0, 100); ax.set_ylim(0, 37)
axes[0].set_ylabel("선택된 분자의 MAE (K)"); axes[0].legend(loc="lower right")
fig.suptitle("10 K는 전체에선 불가 — 가장 자신 있는 분자만 골라도 random 12.6 K (목표 C 미달)", x=0.01, ha="left",
             fontsize=14, fontweight="bold")
fig.tight_layout()
save(fig, "F15_risk_coverage.png", "신뢰도 점수(오차 예측 메타모델) 순으로 분자를 고를 때 커버리지별 MAE. 2 % 미만 구간은 표본이 적어 생략.")

# =====================================================================
# F16 비용 대비 정확도 (validation)
# =====================================================================
cc = load("phase2/ENS_ensemble/candidate_ranking_costs.csv")
cc = cc[~cc.candidate.str.endswith("@T3")].copy()
fig, ax = plt.subplots(figsize=(11, 6))
ax.scatter(cc.total_usd, cc.MAE_random, s=26, color=GRAY_L, linewidths=0, label="다른 후보 (단일·앙상블)")
hl = [("baseline_xgb600_F0", "기준선 XGB", GRAY, (8, 6)), ("xgb600_F0F5_M2", "C: XGB + F5", AQUA, (-120, 18)),
      ("gin_hybrid_seedavg", "GIN 하이브리드", ORANGE, (10, -6)), ("lgb_r1_F0F3F5", "LGB F0+F3+F5", BLUE, (-150, 10)),
      ("fixed_mean|best+GIN", "B: LGB + GIN 50:50", DARKBLUE, (-175, -36)), ("wavg_mse|noO2", "A: 가중 앙상블", VIOLET, (-20, -48))]
for cand, lab, c, off in hl:
    r = cc[cc.candidate == cand].iloc[0]
    ax.plot(r.total_usd, r.MAE_random, "o", ms=10, color=c, mec="#fcfcfb", mew=2)
    ax.annotate(f"{lab}\n{r.MAE_random:.2f} K · {r.total_usd:.2f} USD", xy=(r.total_usd, r.MAE_random),
                xytext=off, textcoords="offset points", fontsize=9.8,
                arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.7))
ax.set_xscale("log"); ax.set_xlabel("총비용 (USD 환산, 탐색 + 최종 학습, 상한) — 로그 축")
ax.set_ylabel("random validation MAE (K)"); ax.set_ylim(24.2, 30)
ax.legend(loc="upper right")
ax.set_title("비용 대비 정확도 — C는 A 비용의 약 1/6로 개선폭의 약 40 %, A·B는 GIN·3D 계산 때문에 비쌈", loc="left")
save(fig, "F16_cost_vs_mae.png", "후보별 총비용(클라우드 환산 USD + LLM API, 상한)과 validation MAE. 30 K 위 후보는 잘림.")

# ---------- 묶음 PDF ----------
with PdfPages(os.path.join(OUT, "figs_all.pdf")) as pdf:
    for name, cap, f in FIGS:
        pdf.savefig(f, bbox_inches="tight")
with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as fh:
    fh.write("# 랩미팅 그림 — mp(녹는점) MAE 개선 (2026-11-23)\n\n")
    fh.write("원자료: `mp_mae_push/phase1–3` 의 저장된 예측·표 (재학습 없음). 단위 K, 분할 seed 42, test n 1,995.\n")
    fh.write("random 기준선 = 원본 예측 파일(27.50 K), scaffold 기준선 = 재적합(35.85 K). CI = 부트스트랩 95 %.\n\n")
    fh.write("| 파일 | 내용 |\n|---|---|\n")
    for name, cap, f in FIGS:
        fh.write(f"| `{name}` | {cap} |\n")
    fh.write("\n생성 스크립트: `make_figs.py`\n")
print("done", len(FIGS))
