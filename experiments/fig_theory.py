#!/usr/bin/env python3
"""Figure 2: every single-call ordering implied by (R), (M), (I) that our data contain (results_law/theory_check_final.json).
Rows are grouped by the relation they test; labels name only the data set and model."""
import json
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

O = json.load(open('results_law/theory_check_final.json'))
BLUE, ORANGE, GREY, INK, MUTED = '#2a78d6', '#eb6834', '#9a9893', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 7, 'font.family': 'DejaVu Sans', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': INK, 'axes.linewidth': 0.6})
NICE = {'gpt54': 'GPT-5.4', 'sonnet46': 'Claude Sonnet 4.6', 'gemini25': 'Gemini 2.5', 'gemma4': 'Gemma 4', 'qwen3vl': 'Qwen3-VL',
        'gpt56': 'GPT-5.6', 'sonnet55': 'Claude Sonnet 5.5', 'gem38': 'Gemini 3.8', 'gem38lo': 'Gemini 3.8 (low budget)',
        'gem38uh': 'Gemini 3.8 (ultra-high budget)', 'gpt61': 'GPT-6.1', 'opus55': 'Claude Opus 5.5'}


def model_of(label, ds):
    rest = label[len(ds):].strip()
    for k in sorted(NICE, key=len, reverse=True):
        if rest.startswith(k):
            tail = rest[len(k):].strip()
            if tail == 'fixed tiles': tail = 'fixed-grid tiles'
            if tail == '+reasoning': tail = 'with reasoning'
            return NICE[k] + (f', {tail}' if tail else '')
    return rest or label


def style(o):
    return {'supported': (BLUE, 'o', BLUE), 'consistent': (BLUE, 'o', 'white'), 'violated': (ORANGE, 'X', ORANGE)}[o['verdict']]


def draw(ax, groups, xlabel, title, outside=None):
    y = 0; ticks, labs, heads = [], [], []
    for head, rows in groups:
        if not rows: continue
        heads.append((y, head)); y += 1
        for lab, o in rows:
            c, mk, fc = style(o)
            ax.plot([o['lo'], o['hi']], [y, y], color=c, lw=1.1, solid_capstyle='round')
            ax.plot(o['diff'], y, marker=mk, ms=4.2 if mk != 'X' else 5.5, mfc=fc, mec=c, mew=0.9, ls='none')
            if outside and lab in outside.get(head, {}):
                q = outside[head][lab]; yy = y + 0.35
                ax.plot([q['lo'], q['hi']], [yy, yy], color=GREY, lw=0.9)
                ax.plot(q['diff'], yy, marker='v', ms=3.6, mfc=GREY, mec=GREY, ls='none')
            ticks.append(y); labs.append(lab); y += 1
        y += 0.4
    ax.set_yticks(ticks); ax.set_yticklabels(labs, fontsize=6.3)
    for yh, h in heads:
        ax.text(-0.02, yh, h, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=6.6, fontweight='bold', color=INK)
    ax.set_ylim(y - 0.4, -1); ax.axvline(0, color=MUTED, lw=0.6)
    ax.set_xlabel(xlabel, fontsize=7); ax.set_title(title, loc='left', fontsize=7.8, color=INK, pad=6)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)


# (a) Theorem 1, by data set
DS = ['REDP-X40', 'REDP-10', 'FPC-300', 'FPC-sheets', 'FSC-147 drawing prompt', 'FSC-147']
thm = [o for o in O if o['family'] == 'Thm1']; outs = [o for o in O if o['family'] == 'outside']
def ds_of(label):
    for d in DS:
        if label.startswith(d + ' '): return d
ga, outside = [], {}
for d in DS:
    rows = [(model_of(o['label'], d), o) for o in thm if ds_of(o['label']) == d]
    head = d.replace('FSC-147 drawing prompt', 'FSC-147, first prompt')
    ga.append((head, rows))
    outside[head] = {model_of(o['label'], d): o for o in outs if ds_of(o['label']) == d}
# (b) the other orderings, by relation
rest = [o for o in O if o['family'] not in ('Thm1', 'outside') and not o['family'].startswith('policy')]
def pick(f): return [o for o in rest if f(o)]
def lab_ds_model(o):
    l = o['label']
    for d in ['REDP-X40 sheets', 'REDP-X40', 'FPC-sheets', 'FPC-300', 'FSC-147 drawing prompt', 'FSC-147', 'synthetic']:
        if l.startswith(d):
            m = model_of(l, d) if l != d else 'Gemini 3.8'
            return f"{d.replace('FSC-147 drawing prompt', 'FSC-147 (first prompt)').replace('REDP-X40 sheets', 'REDP-X40')}, {m}"
    return l
gb = [('(M) single tiles vs tiles packed into one image',
       [(f"{lab_ds_model(o)}, {o['rule'].split(' - ')[1].split(' per')[0]}", o) for o in pick(lambda o: o['rule'].startswith('single tiles') and o['family'] == 'M')]),
      ('(M) tiles vs whole image, both upsampled',
       [(lab_ds_model(o), o) for o in pick(lambda o: o['rule'].startswith('tiles of upsampled'))]),
      ('(R) more tokens on the same pixels',
       [(lab_ds_model(o) + (', ' + o['rule'].split('budget ')[1].split(' (')[0].replace(' - ', ' vs ') if 'budget' in o['rule'] else ', enlarged'), o)
        for o in pick(lambda o: o['family'] == 'R' and 'agent' not in o['rule'])]),
      ('(R+M) single tiles vs a resized packed image',
       [(f"{lab_ds_model(o)}, {o['rule'].split(' - ')[1].split(' per')[0]}" if o['rule'].startswith('single') else lab_ds_model(o) + ', upsampled tiles', o)
        for o in pick(lambda o: o['family'] == 'R+M' and not o['rule'].startswith('first block'))]),
      ('(R+M) a block alone vs inside a larger sheet',
       [(f"{lab_ds_model(o).replace('FPC-sheets, ', '')}, {o['rule'].split('inside ')[1].replace(' sheet', '').replace('x', chr(215))}", o)
        for o in pick(lambda o: o['rule'].startswith('first block'))]),
      ('(R) more pixels or tokens through a vendor setting',
       [(lab_ds_model(o) + (', reasoning' if 'reasoning' in o['rule'] else (', ultra-high budget' if 'ultra-high' in o['rule'] else ', original detail')), o) for o in pick(lambda o: o['family'] == 'R+I' or ('ultra-high' in o['rule'] and 'REDP' in o['label']))])]
# the REDP ultra-high Gemini pair is shown with the vendor modes, not twice
seen = {id(o) for _, rows in gb[5:] for _, o in rows}
gb[2] = (gb[2][0], [(l, o) for l, o in gb[2][1] if id(o) not in seen])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 8.6), gridspec_kw={'width_ratios': [1, 1], 'wspace': 1.25})
draw(ax1, ga, 'hit recall, tiles minus whole image', '(a) Theorem 1: tiles versus the whole image', outside)
draw(ax2, gb, 'recall, predicted better minus predicted worse', '(b) The assumptions on paired single calls')
fig.legend(handles=[Line2D([], [], color=BLUE, marker='o', ls='-', ms=4, label='interval above 0'),
                    Line2D([], [], color=BLUE, marker='o', mfc='white', ls='-', ms=4, label='interval includes 0'),
                    Line2D([], [], color=ORANGE, marker='X', ls='-', ms=5, label='interval below 0 (violation)'),
                    Line2D([], [], color=GREY, marker='v', ls='-', ms=4, label=r'grey: items with $S_{\rm tile} < S_{\rm whole}$ (no prediction)')],
           loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=4, fontsize=6.3, frameon=False, handlelength=1.5, columnspacing=1.0)
fig.subplots_adjust(bottom=0.06, top=0.97, left=0.2, right=0.99)
fig.savefig('../paper/figures/fig_theory.pdf', bbox_inches='tight'); fig.savefig('../paper/figures/fig_theory.png', dpi=200, bbox_inches='tight')
print(sum(len(r) for _, r in ga), 'Thm1 rows;', sum(len(r) for _, r in gb), 'other rows;', len(rest), 'other orderings')
