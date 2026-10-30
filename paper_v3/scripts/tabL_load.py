#!/usr/bin/env python3
"""Supplement L: table of the S = 1.5 load sweep and value macros, from results_ls/ls_summary.json (ls_analyze.py).
usage (from the release's experiments/ directory): python3 tabL_load.py
Writes generated/tabL_load.tex (rows) and generated/macros_ls.tex (\\lsv{model}{load}{field})."""
import json
from pathlib import Path
G = Path(__file__).resolve().parent.parent / 'generated'
S = json.load(open('results_ls/ls_summary.json'))
NAME = {'gpt61': 'GPT-6.1', 'opus55': 'Claude Opus 5.5', 'gem4': 'Gemini 4', 'gpt56': 'GPT-5.6', 'gem38': 'Gemini 3.8', 'qwen3vl': 'Qwen3-VL'}
STD = [8, 16, 32, 64, 128, 256]


def cell(v):
    if not v or not v.get('k'): return ''
    s = f"{v['R']:.2f}; {v['joint']}" + ('' if v['k'] == v['planned'] else f" of {v['k']}")
    return s


rows, mac = [], []
for m in NAME:
    cells = {v['n']: v for v in S.values() if v['model'] == m}
    loads = sorted(cells)
    extra = [n for n in loads if n not in STD]
    row = [NAME[m]] + [cell(cells.get(n)) for n in STD]
    if extra:
        v = cells[extra[-1]]
        last = f"{extra[-1]}: {cell(v)}" if v.get('k') else ''
        if v.get('k') and v.get('empty'): last += f" ({v['empty']} empty)"
        row.append(last)
    else:
        row.append('')
    rows.append(' & '.join(row) + ' \\\\')
    for n, v in cells.items():
        if not v.get('k'): continue
        for f in ('R', 'P'): mac.append(f"\\expandafter\\def\\csname ls-{m}-{n}-{f}\\endcsname{{{v[f]:.2f}}}")
        for f in ('joint', 'empty', 'k', 'planned', 'under', 'over'): mac.append(f"\\expandafter\\def\\csname ls-{m}-{n}-{f}\\endcsname{{{v[f]}}}")
(G / 'tabL_load.tex').write_text('\n'.join(rows) + '\n')
(G / 'macros_ls.tex').write_text('\\providecommand{\\lsv}[3]{\\csname ls-#1-#2-#3\\endcsname}\n' + '\n'.join(mac) + '\n')
print('\n'.join(rows))
