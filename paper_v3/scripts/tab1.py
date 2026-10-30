#!/usr/bin/env python3
"""Table of the rule: the rule (S* = 1.5, no optional subdivision) vs the whole image and the 44 px recipe, 10 cells.
Uses cs_analyze.py (format-aware reader; Gemini 4 whole image from A1n_) and the precision logic of item8_precision.py.
Writes paper_v3/generated/tab1.tex and tab1.json."""
import contextlib, io, json, glob, os, sys
sys.path.insert(0, os.getcwd())
from pathlib import Path
import numpy as np
from pathlib import Path as _P
_PAPER = str(_P(__file__).resolve().parent.parent); _UNI = str(_P.cwd().parent)   # run from the release's experiments/
with contextlib.redirect_stdout(io.StringIO()):
    import cs_analyze as C
S = json.load(open('results_cs/cs_summary.json'))   # written by cs_analyze on import
def boot(rows, stat, B=4000, seed=1):
    rng = np.random.default_rng(seed); m = len(rows)
    est = stat(rows); bs = [stat([rows[i] for i in rng.integers(0, m, m)]) for _ in range(B)]
    return [float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
NAME = {'gpt61': 'GPT-6.1', 'opus55': 'Claude Opus 5.5', 'gem38': 'Gemini 3.8', 'gem4': 'Gemini 4', 'qwen3vl': 'Qwen3-VL'}
DS = {'redp': 'LegendSpot', 'fpc': 'Floor plans'}
out = {}; lines = []
for bench in ('redp', 'fpc'):
    if bench == 'fpc':
        C.CR.BENCH_ARG = 'fpc'; C.M.BENCH = Path('../data/fpc/sheets')
        ids = sorted(Path(p).stem for p in glob.glob('../data/fpc/sheets/annotations/*_k3.json'))
    else:
        C.CR.BENCH_ARG = 'redp'; C.M.BENCH = Path('../data/redp40/redp_x40')
        import ceilings as K; ids = list(K.TEST)
    metric = C.F.metrics if bench == 'fpc' else C.A.metrics
    for v in C.MODELS:
        W = C.whole(bench, v); CS = C.tiled(C.RC / bench / f'CS_{v}_run0.jsonl')
        rows = []
        for did in ids:
            if did not in W or did not in CS: continue
            mW = metric(did, W[did][0])
            if np.isnan(mW[0]): continue
            mC = metric(did, CS[did][0])
            rows.append({'W': mW, 'CS': mC})
        both = [r for r in rows if r['W'][4] > 0 and r['CS'][4] > 0]
        pW = float(np.mean([r['W'][2] for r in both])); pC = float(np.mean([r['CS'][2] for r in both]))
        dP = boot(both, lambda rs: float(np.mean([r['CS'][2] - r['W'][2] for r in rs])))
        s = S[f'{bench}|{v}']['summary']
        cell = {'n': s['n'], 'recW': s['W']['recall'], 'recR': s['CS']['recall'], 'dRec': s['CS-W']['recall'],
                'precW': pW, 'precR': pC, 'dPrec': dP, 'n_both': len(both), 'empty': len(rows) - len(both),
                'tokW': s['tok_CS_over_W'], 'tokD': s.get('tok_CS_over_D'), 'dRecD': s.get('CS-D', {}).get('recall'),
                'cmin': s['ratio_cmin_median'], 'f1W': s['W']['F1'], 'f1R': s['CS']['F1'], 'dF1': s['CS-W']['F1']}
        out[f'{bench}|{v}'] = cell
        sig = lambda d: d is not None and (d[1] > 0 or d[2] < 0)
        star = lambda d: '$^{*}$' if sig(d) else ''
        dd = cell['dRecD']
        lines.append(f"{DS[bench]} & {NAME[v]} & {cell['recW']:.2f} $\\to$ {cell['recR']:.2f}{star(cell['dRec'])} & "
                     f"{pW:.2f} $\\to$ {pC:.2f}{star(dP)} & {cell['f1W']:.2f} $\\to$ {cell['f1R']:.2f}{star(cell['dF1'])} & {cell['tokW']:.1f} & "
                     + (f"{dd[0]:+.2f}{star(dd)}".replace('-', '$-$') if dd else 'n/a') + f" & {cell['cmin']:.1f} \\\\")
    lines.append('\\midrule')
lines = lines[:-1]
tex = ("\\begin{tabular}{llcccccc}\n\\toprule\nData & Model & Recall & Precision & F1 & Tokens & Recall & Cost \\\\\n"
       " & & whole $\\to$ rule & whole $\\to$ rule & whole $\\to$ rule & rule/whole & rule $-$ 44 px & $/C_{\\min}$ \\\\\n\\midrule\n"
       + '\n'.join(lines) + "\n\\bottomrule\n\\end{tabular}\n")
open(_PAPER + '/generated/tab1.tex', 'w').write(tex)
json.dump(out, open(_PAPER + '/generated/tab1.json', 'w'), indent=1)
print(tex)
for k, c in out.items(): print(k, 'empty', c['empty'], 'dPrec', [round(x, 3) for x in c['dPrec']], 'tokD', c['tokD'])
