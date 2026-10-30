#!/usr/bin/env python3
"""Supplement tables computed from released outputs and recorded results (no model calls).
usage (from the release's experiments/ directory): python3 gen_supp_tables.py
Writes generated/tabG_zoomout.tex, generated/tabI_rule.tex, generated/tabI_step3.tex."""
import json, math
from pathlib import Path
PAPER = Path(__file__).resolve().parent.parent
GEN = PAPER / 'generated'
NICE = {'gpt54': 'GPT-5.4', 'sonnet46': 'Claude Sonnet 4.6', 'qwen3vl': 'Qwen3-VL', 'gpt56': 'GPT-5.6', 'sonnet55': 'Claude Sonnet 5.5',
        'gpt61': 'GPT-6.1', 'opus55': 'Claude Opus 5.5', 'gem38': 'Gemini 3.8', 'gem4': 'Gemini 4'}
DSN = {'REDP-X40': 'LegendSpot', 'REDP-10': 'Library-10', 'FPC-300': 'Floor-plan blocks', 'FPC-sheets': 'Floor-plan mosaics'}
def num(x, d=2):
    s = f'{x:+.{d}f}'
    if s in ('+0.00', '-0.00'): s = '0.00' if abs(x) < 0.0005 else f'{x:+.3f}'
    return s.replace('-', '$-$')

# G: the 21 cells in which the 44 px recipe zooms out (theory_check_final.json, family 'outside')
rows = []
for o in json.load(open('results_law/theory_check_final.json')):
    if o['family'] != 'outside': continue
    ds, rest = next((k, o['label'][len(k):].strip()) for k in DSN if o['label'].startswith(k))
    mk, _, tail = rest.partition(' ')
    name = NICE[mk] + (', with reasoning' if '+reasoning' in tail else '')
    sig = o['hi'] < 0
    d = num(o['diff']); d = f'\\textbf{{{d}}}' if sig else d
    rows.append(f"{DSN[ds]} & {name} & {o['n']} & {d} & [{num(o['lo'])}, {num(o['hi'])}] \\\\")
open(GEN / 'tabG_zoomout.tex', 'w').write('\n'.join(rows) + '\n')
print(len(rows), 'zoom-out cells')

# I: Table 2 with intervals (tab1.json)
T = json.load(open(GEN / 'tab1.json'))
DS = {'redp': 'LegendSpot', 'fpc': 'Floor plans'}
out = []
for key, c in T.items():
    b, v = key.split('|')
    ci = lambda a: f"{num(a[0])} [{num(a[1])}, {num(a[2])}]"
    sigb = lambda a, s: f'\\textbf{{{s}}}' if (a[1] > 0 or a[2] < 0) else s
    dd = c['dRecD']
    out.append(f"{DS[b]} & {NICE[v]} & {c['n']} & {sigb(c['dRec'], ci(c['dRec']))} & {sigb(c['dPrec'], ci(c['dPrec']))} & {sigb(c['dF1'], ci(c['dF1']))} & "
               f"{c['n_both']} & " + (sigb(dd, ci(dd)) if dd else 'n/a') + ' & ' + (f"{c['tokD']:.2f}" if c['tokD'] else 'n/a') + f" & {c['cmin']:.2f} \\\\")
    if key == 'redp|qwen3vl': out.append('\\midrule')
open(GEN / 'tabI_rule.tex', 'w').write('\n'.join(out) + '\n')

# I: geometric token cost of step 3 (same-scale subdivision of one full view into q x q views overlapping by one target).
# r_q: image tokens of the q x q sub-views over the parent view. Replacing the view costs r_q; appending the sub-views
# to the view's answer costs 1 + r_q in all.
S = 1.5
res, rq2 = [], {}
for name, N in (('GPT-6.1', 8192), ('Claude Opus 5.5', 4784), ('Qwen3-VL', 2500)):
    side = math.floor(math.sqrt(N))           # a full square view of side sqrt(N) patches
    cells = []
    for q in (2, 3, 4):
        sub = math.ceil((side + (q - 1) * S) / q)   # each sub-view covers its share plus one target (S patches) of overlap
        r = q * q * sub * sub / (side * side)
        if q == 2: rq2[name] = r
        cells.append(f'{r:.2f} & {1 + r:.2f}')
    res.append(f"{name} & {N:,} & ".replace(',', '{,}') + ' & '.join(cells) + ' \\\\')
res.append('Gemini 3.8, Gemini 4 (fixed grid) & about 1{,}090 & 4.00 & 5.00 & 9.00 & 10.00 & 16.00 & 17.00 \\\\')
open(GEN / 'tabI_step3.tex', 'w').write('\n'.join(res) + '\n')
lo, hi = min(rq2.values()), max(rq2.values())
open(GEN / 'macros_step3.tex', 'w').write(
    f"\\newcommand{{\\CstepRepMin}}{{{lo:.2f}}}\n\\newcommand{{\\CstepRepMax}}{{{hi:.2f}}}\n"
    f"\\newcommand{{\\CstepAppMin}}{{{1 + lo:.2f}}}\n\\newcommand{{\\CstepAppMax}}{{{1 + hi:.2f}}}\n"
    f"\\newcommand{{\\CstepGridRep}}{{4.00}}\n\\newcommand{{\\CstepGridApp}}{{5.00}}\n")
print('\n'.join(res))
