#!/usr/bin/env python3
"""Score step 2 (FOV x upscale). Per drawing and cell: F1 (tau_sym), F1 (tau_1600),
recall, precision, |count error|/gt. Contrasts are paired over drawings with a
4000-resample bootstrap; predicted S_tok per cell is printed next to the results."""
import json, math, sys
from collections import defaultdict
import numpy as np
import mech as M

TOKEN = {'gpt': 29.2, 'claude': 27.4}
CELLS = ['L1', 'L2', 'S1', 'S2', 'L2n']


def pred_stok(model, cell):
    s0 = 0.75 * M.GPT_TOKEN                       # symbol px in the z0-scaled drawing (22 px)
    fov = {'L': 768, 'S': 384}[cell[0]]; up = 2 if cell[1] == '2' else 1
    if model in ('gemini', 'gemma4'):             # fixed 16x16 grid over the delivered tile
        return s0 / (fov / 16)
    tile = fov * up; sym = s0 * up
    if model == 'claude':                          # API downsizes to <= 1.15 MP
        f = min(1.0, math.sqrt(1.15e6 / tile ** 2)); sym *= f
    return sym / TOKEN[model]


def load(model, run=0):
    recs = {}
    for l in open(M.OUT / f'step2_{model}_run{run}.jsonl'):
        r = json.loads(l)
        k = (r['did'], r['cell'], r['i'], r['j'])
        if r['status'] in ('success', 'parse_error') or k not in recs:
            recs[k] = r
    return list(recs.values())


def score(model, run=0):
    recs = load(model, run)
    anns = {}
    pts = defaultdict(list); bad = defaultdict(int); ntile = defaultdict(int)
    for r in recs:
        k = (r['did'], r['cell']); ntile[k] += 1
        if r['status'] != 'success':
            bad[k] += 1; continue
        pts[k] += M.to_native(r)
    res = {}
    for (did, cell) in ntile:
        a = anns.setdefault(did, json.loads((M.BENCH / 'annotations' / f'{did}.json').read_text()))
        tsym, t16 = M.taus(a); g = a['count']; p = pts[(did, cell)]
        tp, n_p, n_g = M.match(p, a['points'], tsym)
        tp16, _, _ = M.match(p, a['points'], t16)
        res[(did, cell)] = {'f1': 2 * tp / max(n_p + n_g, 1), 'f1_30': 2 * tp16 / max(n_p + n_g, 1),
                            'rec': tp / max(n_g, 1), 'prec': tp / max(n_p, 1),
                            'relerr': abs(n_p - g) / g, 'n': n_p, 'gt': g,
                            'tiles': ntile[(did, cell)], 'bad': bad[(did, cell)]}
    return res


def boot(diffs, n=4000, seed=0):
    d = np.asarray(diffs, float)
    if len(d) < 3:
        return float(d.mean()), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def report(model, run=0):
    res = score(model, run)
    dids = sorted({d for d, _ in res})
    cells = [c for c in CELLS if any((d, c) in res for d in dids)]
    print(f'\n=== {model} (run {run}) ===')
    print('cell  pred_S_tok   F1_sym  F1_30  recall  prec  |relerr|  complete')
    for c in cells:
        v = [res[(d, c)] for d in dids if (d, c) in res]
        print(f"{c:4s}  {pred_stok(model, c):5.2f}       {np.mean([x['f1'] for x in v]):.2f}    "
              f"{np.mean([x['f1_30'] for x in v]):.2f}   {np.mean([x['rec'] for x in v]):.2f}    "
              f"{np.mean([x['prec'] for x in v]):.2f}  {np.mean([x['relerr'] for x in v]):.2f}     "
              f"{len(v)}/{len(dids)}")
    def contrast(a, b, key='f1'):
        ds = [res[(d, b)][key] - res[(d, a)][key] for d in dids if (d, a) in res and (d, b) in res]
        m, lo, hi = boot(ds)
        return f'{m:+.2f} [{lo:+.2f}, {hi:+.2f}] (n={len(ds)})'
    print('contrasts on F1_sym (paired over drawings):')
    print('  upscale, large FOV  L1->L2 :', contrast('L1', 'L2'))
    print('  upscale, small FOV  S1->S2 :', contrast('S1', 'S2'))
    print('  smaller FOV, x1     L1->S1 :', contrast('L1', 'S1'))
    print('  smaller FOV, x2     L2->S2 :', contrast('L2', 'S2'))
    if 'L2n' in cells:
        print('  native info         L2->L2n:', contrast('L2', 'L2n'))
    print('per drawing F1_sym:', {d: [round(res[(d, c)]['f1'], 2) if (d, c) in res else None for c in cells] for d in dids})
    return res


if __name__ == '__main__':
    for m in (sys.argv[1:] or ['gpt', 'claude', 'gemini']):
        report(m)


def report_runs(model, runs):
    per = [score(model, r) for r in runs]
    keys = set.intersection(*[set(p) for p in per])
    res = {k: {m: float(np.mean([p[k][m] for p in per])) for m in ('f1', 'f1_30', 'rec', 'prec', 'relerr')} for k in keys}
    dids = sorted({d for d, _ in res}); cells = [c for c in CELLS if any((d, c) in res for d in dids)]
    print(f'\n=== {model} (mean of runs {runs}) ===')
    print('cell  pred_S_tok   F1_sym  F1_30  recall  prec  |relerr|')
    for c in cells:
        v = [res[(d, c)] for d in dids if (d, c) in res]
        print(f"{c:4s}  {pred_stok(model, c):5.2f}       {np.mean([x['f1'] for x in v]):.2f}    {np.mean([x['f1_30'] for x in v]):.2f}   "
              f"{np.mean([x['rec'] for x in v]):.2f}    {np.mean([x['prec'] for x in v]):.2f}  {np.mean([x['relerr'] for x in v]):.2f}")
    for key in ('f1', 'f1_30', 'relerr'):
        def ct(a, b):
            ds = [res[(d, b)][key] - res[(d, a)][key] for d in dids if (d, a) in res and (d, b) in res]
            m, lo, hi = boot(ds); return f'{m:+.2f} [{lo:+.2f},{hi:+.2f}]'
        line = f'  {key:6s} L1->L2 {ct("L1","L2")} | S1->S2 {ct("S1","S2")} | L1->S1 {ct("L1","S1")} | L2->S2 {ct("L2","S2")}'
        if 'L2n' in cells:
            line += f' | L2->L2n {ct("L2","L2n")} | L1->L2n {ct("L1","L2n")}'
        print(line)
    if len(runs) > 1:
        rr = [abs(per[0][k]['f1'] - per[1][k]['f1']) for k in keys]
        print(f'  run-to-run |dF1_sym| per drawing-cell: median {np.median(rr):.2f}, max {max(rr):.2f}')
    return res
