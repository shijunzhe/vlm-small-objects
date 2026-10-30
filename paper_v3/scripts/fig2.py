#!/usr/bin/env python3
"""Figure 2 (single column): (a) recognition curve, (b) recall as content grows at fixed S.
Reads syn/analysis_robust.json of the unified-reader copy. Run from that copy's experiments/."""
import json, numpy as np, matplotlib
from pathlib import Path as _P
_PAPER = str(_P(__file__).resolve().parent.parent); _UNI = str(_P.cwd().parent)   # run from the release's experiments/
matplotlib.use('Agg'); import matplotlib.pyplot as plt
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.family': 'serif', 'font.size': 7, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42})
STY = {'gpt54': ('#2a78d6', 'o', '--', 'GPT-5.4'), 'gpt56': ('#2a78d6', 'o', '-', 'GPT-5.6'),
       'sonnet46': ('#eb6834', 's', '--', 'Sonnet 4.6'), 'sonnet55': ('#eb6834', 's', '-', 'Sonnet 5.5'),
       'gemini25': ('#1baf7a', '^', '--', 'Gemini 2.5'), 'gem38': ('#1baf7a', '^', '-', 'Gemini 3.8'),
       'gemma4': ('#8a5cc2', 'D', ':', 'Gemma 4'), 'qwen3vl': ('#b08a1e', 'v', ':', 'Qwen3-VL')}
rows = json.load(open('syn/analysis_robust.json'))['rows']
fig, axs = plt.subplots(1, 2, figsize=(3.35, 1.75))
for v, (c, mk, ls, lab) in STY.items():
    for ax, exp in zip(axs, ('res', 'cap')):
        rs = [r for r in rows if r['v'] == v and r['exp'] == exp]
        if not rs: continue
        lv = sorted({r['level'] for r in rs})
        xs = [np.mean([r['S'] for r in rs if r['level'] == l]) for l in lv] if exp == 'res' else lv
        ys = [np.mean([r['recall'] for r in rs if r['level'] == l]) for l in lv]
        ax.plot(xs, ys, color=c, linestyle=ls, marker=mk, markersize=2.6, linewidth=0.9, markerfacecolor=c if ls == '-' else SURF, label=lab)
a, b = axs
a.set_xscale('log'); a.set_xticks([0.25, 0.5, 1, 2]); a.set_xticklabels(['0.25', '0.5', '1', '2']); a.minorticks_off()
a.axvline(1.5, color=INK2, lw=0.8, ls=(0, (2, 2))); a.text(1.5, 0.05, ' $S^*$', fontsize=6.5, color=INK2)
a.set_xlabel('Tokens per target side $S$'); a.set_ylabel('Recall'); a.set_title('(a) More $S$: 8 targets', fontsize=7)
b.set_xscale('log', base=2); b.set_xlabel('Targets per call, same $S$'); b.set_title('(b) More content', fontsize=7)
for ax in axs:
    ax.set_ylim(-0.03, 1.05); ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
b.set_yticklabels([])
h, l = a.get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=6.2, handlelength=2.0, columnspacing=0.8)
fig.tight_layout(pad=0.2, w_pad=0.6, rect=(0, 0.13, 1, 1))
fig.savefig(_PAPER + '/figures/fig2.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(_PAPER + '/figures/fig2.png', bbox_inches='tight', pad_inches=0.02, dpi=220)
