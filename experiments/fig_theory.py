#!/usr/bin/env python3
"""Figure: every ordering implied by Assumption 1, checked on recall (results_law/theory_check.json)."""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

O = json.load(open('results_law/theory_check_final.json'))
BLUE, PALE, ORANGE, GREY, INK, MUTED = '#2a78d6', '#9ec3ee', '#eb6834', '#9a9893', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6})
NICE = {'gpt54': 'GPT-5.4', 'sonnet46': 'Sonnet 4.6', 'gemini25': 'Gemini 2.5', 'gemma4': 'Gemma 4', 'qwen3vl': 'Qwen3-VL',
        'gpt56': 'GPT-5.6', 'sonnet55': 'Sonnet 5.5', 'gem38': 'Gemini 3.8', 'gem38lo': 'Gemini 3.8 low', 'gem38uh': 'Gemini 3.8 ultra'}
def nice(label):
    label = label.replace('interior targets, hit recall', '').strip()
    if ' ' not in label: return label
    ds, v = label.rsplit(' ', 1)
    return f'{ds}, {NICE[v]}' if v in NICE else label
SHORT = {'overview crop - overview (agent)': 'crop vs overview (agent)', 'enlarged crop - crop (agent)': 'enlarged vs plain crop (agent)',
         'native zoom - enlarged crop (agent)': 'native vs enlarged crop (agent)', 'tiles of upsampled - whole upsampled': 'tiles vs whole, upsampled',
         'upsampled - native': 'upsampled vs native', 'upsampled - original': 'upsampled vs original'}
def short(rule):
    r = rule.split(' (')[0] if not rule.endswith('(agent)') else rule
    if r.startswith('single tiles'): return '1 vs ' + r.split(' - ')[1].replace(' per image', '/image')
    if r.startswith('Gemini 3.8 budget'): return 'budget ' + r.split('budget ')[1].replace(' - ', ' vs ')
    if r.startswith('first block alone'): return 'block alone vs in ' + r.split('inside ')[1]
    return SHORT.get(r, r)
def style(o):
    return {'supported': (BLUE, 'o', BLUE), 'consistent': (BLUE, 'o', 'white'), 'violated': (ORANGE, 'X', ORANGE)}[o['verdict']]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 5.0), gridspec_kw={'width_ratios': [1, 1], 'wspace': 1.05})
# (a) Theorem 1: per dataset x model, condition met (blue) and not met (grey)
thm = [o for o in O if o['family'] == 'Thm1']; out = {o['label']: o for o in O if o['family'] == 'outside'}
for y, o in enumerate(thm):
    c, mk, fc = style(o)
    ax1.plot([o['lo'], o['hi']], [y, y], color=c, lw=1.2, solid_capstyle='round')
    ax1.plot(o['diff'], y, marker=mk, ms=4.5, mfc=fc, mec=c, mew=1.0, ls='none')
    if o['label'] in out:
        q = out[o['label']]; yy = y + 0.32
        ax1.plot([q['lo'], q['hi']], [yy, yy], color=GREY, lw=1.0)
        ax1.plot(q['diff'], yy, marker='v', ms=4, mfc=GREY, mec=GREY, ls='none')
ax1.set_yticks(range(len(thm))); ax1.set_yticklabels([nice(o['label']) for o in thm]); ax1.invert_yaxis()
ax1.axvline(0, color=MUTED, lw=0.6)
ax1.set_xlabel('recall, tiles minus whole image')
ax1.set_title('(a) Theorem 1: tiles versus the whole image', loc='left', fontsize=7.5, color=INK, pad=24)
ax1.legend(handles=[Line2D([], [], color=BLUE, marker='o', ls='-', ms=4, label=r'$S_{\rm tile} \geq S_{\rm whole}$ (theorem applies)'),
                    Line2D([], [], color=GREY, marker='v', ls='-', ms=4, label=r'$S_{\rm tile} < S_{\rm whole}$ (no guarantee)')],
           loc='upper center', bbox_to_anchor=(0.45, -0.1), ncol=2, fontsize=6, frameon=False, handlelength=1.5)
for s in ('top', 'right'): ax1.spines[s].set_visible(False)
# (b) all other orderings, grouped by the assumption they test
rest = [o for o in O if o['family'] not in ('Thm1', 'outside') and not o['family'].startswith('policy')]
order = {'M': 0, 'R': 1, 'R+M': 2, 'I': 3}; rest.sort(key=lambda o: (order[o['family']], o['rule'], o['label']))
labels = []
for y, o in enumerate(rest):
    c, mk, fc = style(o)
    ax2.plot([o['lo'], o['hi']], [y, y], color=c, lw=1.2)
    ax2.plot(o['diff'], y, marker=mk, ms=4.5 if mk != 'X' else 6, mfc=fc, mec=c, mew=1.0, ls='none')
    labels.append(f'{o["family"]}  {short(o["rule"])}, {nice(o["label"]).replace("REDP-X40 sheets", "REDP sheets").replace("REDP-X40", "REDP")}')
ax2.set_yticks(range(len(rest))); ax2.set_yticklabels(labels, fontsize=5.0); ax2.invert_yaxis()
ax2.axvline(0, color=MUTED, lw=0.6)
ax2.set_xlabel('recall, predicted better minus predicted worse')
ax2.set_title('(b) Other orderings implied by (R), (M), (I)', loc='left', fontsize=7.5, color=INK, pad=24)
ax2.legend(handles=[Line2D([], [], color=BLUE, marker='o', ls='-', ms=4, label='interval above 0'),
                    Line2D([], [], color=BLUE, marker='o', mfc='white', ls='-', ms=4, label='interval includes 0'),
                    Line2D([], [], color=ORANGE, marker='X', ls='-', ms=5, label='interval below 0 (violation)')],
           loc='upper center', bbox_to_anchor=(0.3, -0.1), ncol=3, fontsize=5.5, frameon=False, handlelength=1.5, columnspacing=0.8)
for s in ('top', 'right'): ax2.spines[s].set_visible(False)
fig.savefig('../paper/figures/fig_theory.pdf', bbox_inches='tight'); fig.savefig('../paper/figures/fig_theory.png', dpi=200, bbox_inches='tight')
