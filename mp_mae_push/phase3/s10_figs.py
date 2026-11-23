"""s10_figs.py : 3단계 그림 (s9_evaluate.py 결과 표·P3/preds/test 를 읽음).
 fig_parity_random_AC.png   후보 A·C random test parity (n 1,995)
 fig_mpbin_residuals.png    MP 구간별 평균 잔차(예측 − 관측), 두 분할, 후보 A·B·C + 기준선
 fig_risk_coverage.png      test 위험–커버리지 곡선(실선)과 validation 곡선(점선; ENS 정직한 교차 적합 예측 + 저장 점수)
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p3lib as P  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

FIG = os.path.join(P.P3, 'figs')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False, 'font.size': 8, 'axes.titlesize': 8,
                     'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6,
                     'xtick.direction': 'out', 'ytick.direction': 'out', 'legend.frameon': False, 'savefig.dpi': 300,
                     'axes.spines.top': False, 'axes.spines.right': False})
COL = {'A': '#1f5fa8', 'B': '#d9822b', 'C': '#2a9d6f', 'base_sbx': '#7f7f7f', 'base_orig': '#b0b0b0'}
LAB = {'A': 'A 가중 앙상블', 'B': 'B LGB+GIN 50:50', 'C': 'C XGB F0+F5', 'base_sbx': '기준선(sandbox 재적합)',
       'base_orig': '기준선(원본 예측 파일)'}
VALCOL = {'A': 'wavg_mse|noO2', 'B': 'fixed_mean|best+GIN', 'C': 'xgb600_F0F5_M2'}
FILE = {'A': 'A_wavg_mse_noO2', 'B': 'B_fixed_mean_best_GIN', 'C': 'C_xgb600_F0F5_M2'}


def parity():
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.3), sharex=True, sharey=True)
    R = pd.read_csv(os.path.join(P.TAB, 'test_results.csv'))
    for ax, c in zip(axs, ['A', 'C']):
        d = pd.read_csv(os.path.join(P.P3, 'preds', 'test', f'{FILE[c]}_random_test.csv'))
        ax.scatter(d['y'], d['pred'], s=3, alpha=0.35, color=COL[c], linewidths=0)
        lo, hi = 50, 750
        ax.plot([lo, hi], [lo, hi], color='k', lw=0.6)
        r = R[(R.cand == c) & (R.split == 'random') & (R.basis == '전체')].iloc[0]
        ax.text(0.04, 0.96, f'MAE {r.MAE:.2f} K\nRMSE {r.RMSE:.2f} K\nR² {r.R2:.3f}\nn {int(r.n):,}', transform=ax.transAxes,
                va='top', fontsize=7)
        ax.set_title(LAB[c], loc='left')
        ax.set_xlabel('관측 녹는점 [K]')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    axs[0].set_ylabel('예측 녹는점 [K]')
    fig.suptitle('random test: 예측 대 관측 (train 전용 학습)', x=0.01, ha='left', fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, 'fig_parity_random_AC.png'); fig.savefig(p); plt.close(fig)
    return p


def mpbin():
    B = pd.read_csv(os.path.join(P.TAB, 'mp_bin_residuals.csv'))
    bins = ['<250', '250-350', '350-450', '450-550', '>550']
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=True)
    for ax, how in zip(axs, P.SPLITS):
        sub = B[B.split == how]
        cs = [c for c in ['base_orig', 'base_sbx', 'C', 'B', 'A'] if c in set(sub.cand)]
        off = np.linspace(-0.24, 0.24, len(cs))
        for o, c in zip(off, cs):
            s = sub[sub.cand == c].set_index('bin').loc[bins]
            ax.plot(np.arange(5) + o, s['mean_resid'], 'o-', ms=3, lw=0.8, color=COL[c], label=LAB[c],
                    zorder=3 if c in 'ABC' else 2)
        n = sub[sub.cand == cs[-1]].set_index('bin').loc[bins, 'n']
        ax.set_xticks(range(5)); ax.set_xticklabels([f'{b}\n(n {int(v)})' for b, v in zip(bins, n)])
        ax.axhline(0, color='k', lw=0.5)
        ax.set_title(f'{how} test', loc='left')
        ax.set_xlabel('관측 MP 구간 [K]')
    axs[0].set_ylabel('평균 잔차 (예측 - 관측) [K]')
    axs[0].legend(loc='lower left', fontsize=6)
    fig.suptitle('MP 구간별 평균 잔차: 끝 구간 쪽으로 예측이 평균에 끌림', x=0.01, ha='left', fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, 'fig_mpbin_residuals.png'); fig.savefig(p); plt.close(fig)
    return p


def risk():
    GC = pd.read_csv(os.path.join(P.TAB, 'goalC_curve_test.csv'))
    G = pd.read_csv(os.path.join(P.TAB, 'goalC_test.csv'))
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.0), sharey=False)
    for ax, how in zip(axs, P.SPLITS):
        cv = pd.read_csv(os.path.join(P.INP, 'ens', f'cand_val_{how}.csv')).set_index('row')
        sc = pd.read_csv(os.path.join(P.INP, 'ens', f'scores_A_{how}.csv')).set_index('row').loc[cv.index]
        s = sc['score_metaA_trainOOF_self'].to_numpy()
        for c in ['C', 'B', 'A']:
            err = np.abs(cv[VALCOL[c]].to_numpy() - cv['y'].to_numpy())
            o = np.argsort(s, kind='stable'); cum = np.cumsum(err[o]) / np.arange(1, len(err) + 1)
            ax.plot(np.arange(1, len(err) + 1) / len(err), cum, ls=':', lw=0.9, color=COL[c])
            t = GC[(GC.split == how) & (GC.cand == c)]
            ax.plot(t['coverage'], t['MAE'], lw=1.1, color=COL[c], label=f'{LAB[c]} (test)')
            if how == 'random':
                g = G[(G.split == how) & (G.cand == c)].iloc[0]
                ax.plot([g['coverage']], [g['MAE_sel']], 'o', ms=4, color=COL[c], mec='k', mew=0.4, zorder=4)
        ax.axhline(10, color='#c0392b', lw=0.7, ls='--')
        ax.set_xlim(0, 1); ax.set_ylim(0, (27 if how == 'random' else 37))
        ax.set_title(f'{how}: 실선 test, 점선 validation' + (' (점 = 고정 τ)' if how == 'random' else ' (τ 미고정)'), loc='left')
        ax.set_xlabel('커버리지 (위험 점수 낮은 순 비율)')
    axs[0].set_ylabel('누적 MAE [K]'); axs[1].set_ylabel('누적 MAE [K]')
    axs[0].text(0.98, 10.6, '목표 C: 10 K', ha='right', fontsize=6, color='#c0392b')
    axs[1].legend(loc='lower right', fontsize=6)
    fig.suptitle('위험–커버리지: 메타모델 점수(metaA_trainOOF_self)로 거른 부분집합의 MAE', x=0.01, ha='left', fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIG, 'fig_risk_coverage.png'); fig.savefig(p); plt.close(fig)
    return p


if __name__ == '__main__':
    for f in (parity, mpbin, risk):
        print(f())
