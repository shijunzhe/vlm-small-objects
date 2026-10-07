#!/usr/bin/env python3
"""Figures for the expansion experiments: fig_syn.pdf (X1) and fig_agent.pdf (X3/X4/X4b)."""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

FIG = Path('../paper/figures')
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({'font.family': 'serif', 'font.size': 7.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.linewidth': 0.6, 'legend.frameon': False, 'pdf.fonttype': 42})
# two generations per vendor: older = dashed + hollow, newer = solid + filled; vendor = hue
STY = {'gpt54': ('#2a78d6', 'o', '--', 'GPT-5.4'), 'gpt56': ('#2a78d6', 'o', '-', 'GPT-5.6'),
       'sonnet46': ('#eb6834', 's', '--', 'Sonnet 4.6'), 'sonnet55': ('#eb6834', 's', '-', 'Sonnet 5.5'),
       'gemini25': ('#1baf7a', '^', '--', 'Gemini 2.5'), 'gem38': ('#1baf7a', '^', '-', 'Gemini 3.8'),
       'gemma4': ('#8a5cc2', 'D', ':', 'Gemma 4'), 'qwen3vl': ('#b08a1e', 'v', ':', 'Qwen3-VL')}


def style(ax):
    ax.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)


def fig_syn():
    Z = json.load(open('syn/analysis_robust.json')); rows = Z['rows']
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.3))
    for v, (c, mk, ls, lab) in STY.items():
        rs = [r for r in rows if r['v'] == v and r['exp'] == 'res']
        if rs:
            lv = sorted({r['level'] for r in rs})
            xs = [np.mean([r['S'] for r in rs if r['level'] == l]) for l in lv]; ys = [np.mean([r['recall'] for r in rs if r['level'] == l]) for l in lv]
            axs[0].plot(xs, ys, color=c, linestyle=ls, marker=mk, markersize=3.2, linewidth=1.0, markerfacecolor=c if ls == '-' else SURF, label=lab)
        rs = [r for r in rows if r['v'] == v and r['exp'] == 'cap']
        if rs:
            lv = sorted({r['level'] for r in rs})
            axs[1].plot(lv, [np.mean([r['recall'] for r in rs if r['level'] == l]) for l in lv], color=c, linestyle=ls, marker=mk, markersize=3.2, linewidth=1.0, markerfacecolor=c if ls == '-' else SURF)
        rs = [r for r in rows if r['v'] == v and r['exp'] == 'area']
        if rs:
            lv = sorted({r['level'] for r in rs}, key=lambda s: int(s.split('x')[0]) * int(s.split('x')[1]))
            mp = [int(s.split('x')[0]) * int(s.split('x')[1]) / 1e6 for s in lv]
            axs[2].plot(mp, [np.mean([r['recall'] for r in rs if r['level'] == l]) for l in lv], color=c, linestyle=ls, marker=mk, markersize=3.2, linewidth=1.0, markerfacecolor=c if ls == '-' else SURF)
    axs[0].set_xscale('log'); axs[0].set_xticks([0.25, 0.5, 1, 2]); axs[0].set_xticklabels(['0.25', '0.5', '1', '2']); axs[0].minorticks_off(); axs[0].set_xlabel('Tokens per symbol side $S$ (log)'); axs[0].set_ylabel('Recall')
    axs[0].set_title('(a) Resolution: 8 targets, 1024$^2$', fontsize=7.5)
    axs[1].set_xscale('log', base=2); axs[1].set_xlabel('Targets per image $n$ (log$_2$)'); axs[1].set_title('(b) Capacity: symbol 40 px, 1024$^2$', fontsize=7.5)
    axs[2].set_xscale('log', base=2); axs[2].set_xlabel('Image area (MP, log$_2$)'); axs[2].set_title('(c) Area: 16 targets, symbol 40 px', fontsize=7.5)
    for ax in axs: ax.set_ylim(-0.03, 1.05); style(ax)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=8, bbox_to_anchor=(0.5, -0.06), fontsize=6.5, handlelength=2.2, columnspacing=1.0)
    fig.tight_layout(pad=0.3, rect=(0, 0.07, 1, 1)); fig.savefig(FIG / 'fig_syn.pdf', bbox_inches='tight'); plt.close(fig)


def fig_agent():
    X = json.load(open('results_x/x_new_scores_robust.json'))
    N = json.load(open('results_x/x9_scores.json'))
    fig = plt.figure(figsize=(7.0, 2.45)); gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1])
    ax = fig.add_subplot(gs[0]); ax2 = fig.add_subplot(gs[1])
    # (a) the matched-reasoning ladder: each step changes one thing (registered decomposition)
    vend = [('gpt56', 'GPT-5.6'), ('sonnet55', 'Claude Sonnet 5.5')]
    steps = [('A1R', 'Whole drawing'), ('OVC', 'Agent: overview crops\n(lower $L$)'), ('OVZ', 'Agent: enlarged crops\n(higher $S$)'),
             ('AG', 'Agent: native crops\n(new information)'), ('DR', 'Tiles (reference)')]
    cols = ['#c9c8c3', '#f2b38f', '#eb8a55', '#c4501f', '#2a78d6']
    w = 0.16
    for i, (v, lab) in enumerate(vend):
        for j, (k, nm_) in enumerate(steps):
            val = N.get(f'{k}|{v}', {}).get('sheets')
            if val is None: continue
            x = i + (j - 2) * w + (0.04 if k == 'DR' else 0)
            ax.bar(x, val, w * 0.92, color=cols[j], edgecolor=SURF, linewidth=0.5, label=nm_ if i == 0 else None,
                   hatch='//' if k == 'DR' else None)
            ax.text(x, val + 0.015, f'{val:.2f}', ha='center', va='bottom', fontsize=5.6)
    ax.set_xticks(range(2)); ax.set_xticklabels([v[1] for v in vend], fontsize=7); ax.set_ylim(0, 1.1); ax.set_ylabel('Strict F1, 14 full sheets', fontsize=7)
    ax.axhspan(X.get('TMsheets', [0.85, 0.85])[0], X.get('TMsheets', [0.85, 0.85])[1], color=GRID, zorder=0)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=6.0, handlelength=1.2, columnspacing=0.8)
    style(ax); ax.set_title('(a) One change at a time, reasoning matched', fontsize=7.5)
    # (b) zoom footprint of GPT-5.6 on one sheet
    import mech as M
    did = X.get('agent_example', 'd018')
    rec = [json.loads(l) for l in open('results_x/agent/gpt56_run0.jsonl') if json.loads(l)['did'] == did][0]
    ann, img, tpl = M.load(did); s = rec['scale']; ov = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS).convert('L')
    ax2.imshow(np.asarray(ov), cmap='gray', vmin=0, vmax=255)
    for z in rec['log']['zooms']:
        x0, y0, x1, y1 = z['overview_box']
        ax2.add_patch(matplotlib.patches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#eb6834', linewidth=0.7))
    gx = [p[0] * s for p in ann['points']]; gy = [p[1] * s for p in ann['points']]
    ax2.scatter(gx, gy, s=14, color='#2a78d6', edgecolors='white', linewidths=0.4, zorder=3)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_title(f'(b) GPT-5.6 zoom calls on one sheet ({rec["log"]["zooms_used"]} zooms)', fontsize=7.5)
    fig.tight_layout(pad=0.3); fig.savefig(FIG / 'fig_agent.pdf', bbox_inches='tight', dpi=300); plt.close(fig)


if __name__ == '__main__':
    fig_syn()
    try:
        fig_agent()
    except Exception as e:
        print('fig_agent skipped:', e)
