#!/usr/bin/env python3
"""Figures for paper 2 from generated/numbers.json (static PDFs, print-safe, palette slots 1-3, shapes as
secondary encoding, legends always present for >= 2 series)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

N = json.load(open('../paper/generated/numbers.json'))
FIG = Path('../paper/figures'); FIG.mkdir(exist_ok=True)
C1, C2, C3 = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'legend.frameon': False,
                     'pdf.fonttype': 42})


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


# Figure 1: per-drawing dose-response, whole drawing, observed vs a priori ceiling
fig, ax = plt.subplots(figsize=(3.0, 2.9))
for key, lab, col, mk in (('gpt54', 'GPT-5.4', C1, 'o'), ('sonnet46', 'Claude Sonnet 4.6', C2, 's'), ('sonnet55', 'Claude Sonnet 5.5', C3, '^')):
    d = N['dose'][key]; xs, ys = zip(*d['points'])
    ax.scatter(xs, ys, s=16, color=col, marker=mk, edgecolors=SURF, linewidths=0.8,
               label=f"{lab} ($\\rho$={d['rho']:.2f})", zorder=3)
ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.set_xlabel('A priori ceiling from token geometry'); ax.set_ylabel('Observed strict F1')
ax.legend(loc='upper center', bbox_to_anchor=(0.42, -0.17), ncol=1, fontsize=6.5, handletextpad=0.3)
style(ax); fig.tight_layout(pad=0.3); fig.savefig(FIG / 'fig_dose.pdf', bbox_inches='tight'); plt.close(fig)

# Figure 2: two generations x three configurations, per vendor family, template matching band
fam = [('OpenAI', 'gpt54', 'gpt56', 'GPT-5.4', 'GPT-5.6'), ('Anthropic', 'sonnet46', 'sonnet55', 'Sonnet 4.6', 'Sonnet 5.5'),
       ('Google', 'gemini25', 'gem38', 'Gemini 2.5 Flash', 'Gemini 3.8 Flash')]
conds = [('A2', 'Whole\ndrawing'), ('D', 'Tiles'), ('H', 'Propose +\nverify')]
fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.2), sharey=True)
tm = N['cells']['TM|dev']['all']; tml = N['cells']['TM|loo']['all']
for ax, (name, g1, g2, l1, l2) in zip(axes, fam):
    ax.axhspan(min(tm, tml), max(tm, tml), color=GRID, zorder=0)
    ax.text(-0.55, max(tm, tml) + 0.015, 'template matching', color=INK2, fontsize=6, ha='left', va='bottom')
    for gk, lab, col, mk, off in ((g1, l1, C1, 'o', -0.08), (g2, l2, C2, 's', 0.08)):
        ys = [N['cells'][f'{c}|{gk}']['all'] for c, _ in conds]
        xs = [i + off for i in range(3)]
        ax.plot(xs, ys, color=col, linewidth=1.2, zorder=2)
        ax.scatter(xs, ys, color=col, marker=mk, s=22, edgecolors=SURF, linewidths=0.8, zorder=3, label=lab)
        for x, y in zip(xs, ys):
            ax.text(x + (0.1 if off > 0 else -0.1), y, f'{y:.2f}', fontsize=6, color=INK, ha='left' if off > 0 else 'right', va='center')
    ax.set_xticks(range(3)); ax.set_xticklabels([c[1] for c in conds], fontsize=6.5)
    ax.set_title(name, fontsize=8, color=INK); ax.set_xlim(-0.6, 2.6); ax.set_ylim(0, 1.0)
    ax.legend(loc='lower right', fontsize=6, handletextpad=0.2, borderaxespad=0.1)
    style(ax)
axes[0].set_ylabel('Strict F1 (34 test drawings)')
fig.tight_layout(pad=0.3, w_pad=0.6); fig.savefig(FIG / 'fig_generations.pdf'); plt.close(fig)

# Figure 3: capacity test, GPT-5.6 on full sheets: ceiling vs observed for three ways of sending the sheet
fig, ax = plt.subplots(figsize=(3.1, 2.1))
rows = [('Whole sheet,\ndefault budget', 'A2|gpt56', 0.60), ('Whole sheet,\nnative resolution', 'A1nat|gpt56orig', 1.00), ('Tiles\n(1024 px)', 'D|gpt56', 1.00)]
for i, (lab, key, ce) in enumerate(rows):
    y = N['cells'][key]['sheets']
    ax.bar(i, y, width=0.45, color=C1, zorder=2)
    ax.text(i, y + 0.02, f'{y:.2f}', ha='center', va='bottom', fontsize=7, color=INK)
    ax.plot([i - 0.3, i + 0.3], [ce, ce], color=C2, linewidth=2, zorder=3, label='a priori ceiling' if i == 0 else None)
ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in rows], fontsize=6.5)
ax.set_ylim(0, 1.25); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_ylabel('Strict F1 (14 full sheets)')
ax.bar([], [], color=C1, label='observed'); ax.legend(loc='upper left', fontsize=6.5, ncol=2, bbox_to_anchor=(0.0, 1.02))
style(ax); ax.grid(axis='x', visible=False)
fig.tight_layout(pad=0.3); fig.savefig(FIG / 'fig_capacity.pdf'); plt.close(fig)
print('figures written to', FIG)
