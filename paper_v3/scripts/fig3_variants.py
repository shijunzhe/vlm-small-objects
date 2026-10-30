#!/usr/bin/env python3
"""Supplement H: robustness of Figure 3.
usage: python3 fig3_variants.py <unified law_data.json> <registered law_data.json>
Writes figures_x/figH_models.pdf, figH_data.pdf, figH_registered.pdf and generated/figH_ci.tex."""
import json, sys, collections
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use('Agg'); matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
PAPER = Path(__file__).resolve().parent.parent
SB = [0, 0.5, 1, 1.5, 3, np.inf]; NB = [0, 4, 16, 32, 96, np.inf]
SL = ['<0.5', '[0.5, 1)', '[1, 1.5)', '[1.5, 3)', '≥3']; NL = ['≤3', '4 to 15', '16 to 31', '32 to 95', '≥96']
cmap = LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6']); norm = TwoSlopeNorm(vmin=-0.25, vcenter=0, vmax=0.5)
plt.rcParams.update({'font.size': 7, 'font.family': 'serif'})
def b(x, B): return next(i for i in range(len(B) - 1) if B[i] <= x < B[i + 1])
def pairs(path):
    pair = collections.defaultdict(dict)
    for r in json.load(open(path)):
        if r['ds'] in ('redp', 'fpc_bench', 'fpc_sheets') and r['config'] in ('A1', 'D'): pair[(r['ds'], r['item'], r['model'])][r['config']] = r
        if r['ds'] == 'fsc' and r['config'] in ('U', 'T'): pair[(r['ds'], r['item'], r['model'])]['A1' if r['config'] == 'U' else 'D'] = r
    return pair
def clus(k):
    return (k[0], k[1].rsplit('_k', 1)[0] if k[0] == 'fpc_sheets' else k[1])
def cboot(v, rng, B=4000):
    cl = collections.defaultdict(list)
    for c, d in v: cl[c].append(d)
    sums = np.array([sum(x) for x in cl.values()]); cnt = np.array([len(x) for x in cl.values()])
    bs = [sums[idx].sum() / cnt[idx].sum() for idx in (rng.integers(0, len(sums), len(sums)) for _ in range(B))]
    return np.percentile(bs, [2.5, 97.5])
def grid(pair, keep, filt=lambda k: True, minn=10):
    cell = collections.defaultdict(list)
    for k, v in pair.items():
        if 'A1' in v and 'D' in v and filt(k):
            S, n, _ = v['A1']['calls'][0]; safe = min(c[0] for c in v['D']['calls']) >= S * 0.999
            if safe == keep: cell[(b(S, SB), b(n, NB))].append((clus(k), v['D']['obs'] - v['A1']['obs']))
    M = np.full((5, 5), np.nan); N = np.zeros((5, 5), int); SIG = np.zeros((5, 5), int); CI = {}
    rng = np.random.default_rng(1)
    for (i, j), v in cell.items():
        N[j, i] = len(v)
        if len(v) >= minn:
            M[j, i] = np.mean([d for _, d in v]); lo, hi = cboot(v, rng)
            CI[(j, i)] = (lo, hi); SIG[j, i] = 1 if lo > 0 else (-1 if hi < 0 else 0)
    return M, N, SIG, CI
def panel(ax, M, N, SIG, title, ylab=True, xlab=True, fs=5.6):
    ax.imshow(M, origin='lower', cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
    for j in range(5):
        for i in range(5):
            if not np.isnan(M[j, i]):
                ax.text(i, j, f'{M[j, i]:+.2f}' + ('*' if SIG[j, i] else '') + f'\n({N[j, i]})', ha='center', va='center', fontsize=fs,
                        fontweight='bold' if SIG[j, i] else 'normal')
    ax.axvline(2.5, color='#55554f', lw=0.8, ls='--')
    ax.set_xticks(range(5)); ax.set_xticklabels(SL if xlab else [''] * 5, fontsize=5.5)
    ax.set_yticks(range(5)); ax.set_yticklabels(NL if ylab else [''] * 5, fontsize=5.5); ax.set_title(title, fontsize=7)
U = pairs(sys.argv[1]); RG = pairs(sys.argv[2])
NAMES = {'gpt54': 'GPT-5.4', 'gpt56': 'GPT-5.6', 'sonnet46': 'Claude Sonnet 4.6', 'sonnet55': 'Claude Sonnet 5.5',
         'gemini25': 'Gemini 2.5', 'gem38': 'Gemini 3.8', 'gemma4': 'Gemma 4', 'qwen3vl': 'Qwen3-VL'}
fig, axs = plt.subplots(2, 4, figsize=(7.0, 3.9))
for ax, (m, nm) in zip(axs.flat, NAMES.items()):
    M, N, SIG, _ = grid(U, True, lambda k, m=m: k[2] == m)
    panel(ax, M, N, SIG, nm, ylab=ax in axs[:, 0], xlab=ax in axs[1])
fig.supxlabel('S of the whole-image call', fontsize=7); fig.supylabel('targets in the whole-image call', fontsize=7)
fig.tight_layout(); fig.savefig(PAPER / 'figures_x/figH_models.pdf', dpi=600); plt.close(fig)
DS = {'redp': 'LegendSpot', 'fpc_bench': 'Floor-plan blocks', 'fpc_sheets': 'Floor-plan mosaics', 'fsc': 'Natural images (FSC-147)'}
fig, axs = plt.subplots(1, 4, figsize=(7.0, 2.1))
for ax, (d, nm) in zip(axs, DS.items()):
    M, N, SIG, _ = grid(U, True, lambda k, d=d: k[0] == d)
    panel(ax, M, N, SIG, nm, ylab=ax is axs[0])
fig.supxlabel('S of the whole-image call', fontsize=7); fig.supylabel('targets', fontsize=7)
fig.tight_layout(); fig.savefig(PAPER / 'figures_x/figH_data.pdf', dpi=600); plt.close(fig)
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
for ax, keep, nm in zip(axs, (True, False), ('(a) Splits that never zoom out', '(b) Splits that zoom out')):
    M, N, SIG, _ = grid(RG, keep); panel(ax, M, N, SIG, nm, ylab=ax is axs[0], fs=6)
fig.supxlabel('S of the whole-image call', fontsize=7); fig.supylabel('targets in the whole-image call', fontsize=7)
fig.tight_layout(); fig.savefig(PAPER / 'figures_x/figH_cxcy.pdf', dpi=600); plt.close(fig)
rows = []
for keep, nm in ((True, 'never zoom out'), (False, 'zoom out')):
    M, N, SIG, CI = grid(U, keep)
    for (j, i), (lo, hi) in sorted(CI.items()):
        rows.append(f"{nm} & {NL[j]} & {SL[i]} & {N[j, i]} & {M[j, i]:+.2f} & [{lo:+.2f}, {hi:+.2f}] \\\\".replace('-', '$-$').replace('$-$0', '$-$0'))
open(PAPER / 'generated/figH_ci.tex', 'w').write('\n'.join(rows).replace('≤', '$\\le$').replace('≥', '$\\ge$').replace('<', '$<$').replace('>', '$>$') + '\n')
print(len(rows), 'cells')
