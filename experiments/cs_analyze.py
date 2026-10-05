#!/usr/bin/env python3
"""Prereg v8 analysis: CS (Cor. cor:safe) vs W (A1) vs D44 (D), run0 of each, same scoring for all conditions.
Writes results_cs/cs_summary.json."""
import json, glob, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import mech as M
import analyze_test as A
import fpc_score as F   # noqa: F401  installs FPC metrics for s* ids and loads FPC annotations
sys.argv = ['x', 'redp']
import cs_rule as CR

M.FORMAT_AWARE = True
RG, RT, RF, RC = Path('results_gen'), Path('results_test'), Path('results_fpc/sheets'), Path('results_cs')
DEC = {'gpt61': 'px', 'opus55': 'px', 'qwen3vl': 'n1000', 'gem38': 'n1000'}
MODELS = ['gpt61', 'opus55', 'gem38', 'qwen3vl']


def load_last(f, key):
    last = {}
    if not Path(f).exists(): return last
    for l in open(f):
        r = json.loads(l)
        if r['status'] in ('success', 'parse_error'): last[key(r)] = r
    return last


def whole(bench, v):
    f = (RT if v == 'qwen3vl' else RG) / f'A1_{v}_run0.jsonl' if bench == 'redp' else RF / f'A1_{v}_run0.jsonl'
    out = {}
    for did, r in load_last(f, lambda r: r['did']).items():
        s = r['scale']; w, h = r['sent_size']; pts = []
        for q in (r.get('shapes') or []) if r['status'] == 'success' else []:
            pt = M.point_of(q)
            if pt is None: continue
            x, y = pt
            if DEC[v] == 'n1000': x, y = x / 1000 * w, y / 1000 * h
            pts.append((x / s, y / s))
        out[did] = (pts, (r.get('usage') or {}).get('input_tokens', 0))
    return out


def tiled(f):
    out = defaultdict(lambda: [[], 0])
    for k, r in load_last(f, lambda r: (r['did'], r['i'], r['j'], r.get('half', ''))).items():
        out[r['did']][1] += (r.get('usage') or {}).get('input_tokens', 0)
        if r['status'] == 'success': out[r['did']][0] += M.to_native(r)
    return {d: tuple(x) for d, x in out.items()}


def dedup(pts, tau):
    keep = []
    for p in pts:
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) > tau for q in keep): keep.append(p)
    return keep


def boot(d, B=4000, seed=1):
    d = np.array(d, float); rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


res = {}
for bench in ('redp', 'fpc'):
    if bench == 'fpc':
        CR.BENCH_ARG = 'fpc'; M.BENCH = Path('../data/fpc/sheets')
        ids = sorted(Path(p).stem for p in glob.glob('../data/fpc/sheets/annotations/*_k3.json'))
    else:
        ids = list(A.K.TEST) if hasattr(A, 'K') else None
        import ceilings as K; ids = list(K.TEST)
    metric = F.metrics if bench == 'fpc' else A.metrics
    for v in MODELS:
        W = whole(bench, v)
        CS = tiled(RC / bench / f'CS_{v}_run0.jsonl')
        if bench == 'redp': D = tiled((RT if v == 'qwen3vl' else RG) / f'D_{v}_run0.jsonl')
        else: D = tiled(RF / f'D_{v}_run0.jsonl')
        rows = []
        for did in ids:
            if did not in W or did not in CS: continue
            ann = json.loads((M.BENCH / 'annotations' / f'{did}.json').read_text()); p = CR.plan(v, ann)
            mW, mC = metric(did, W[did][0]), metric(did, CS[did][0])
            if np.isnan(mW[0]): continue
            mCd = metric(did, dedup(CS[did][0], M.taus(ann)[0]))
            row = {'did': did, 'W': mW[:3].tolist(), 'CS': mC[:3].tolist(), 'CSdedup': mCd[:3].tolist(), 'tokW': W[did][1], 'tokCS': CS[did][1],
                   'Sw': p['Sw'], 'S': p['S'], 'views': p['n_views'], 'chi': p['chi'], 'cmin': p['cmin'], 'plan_tokens': p['tokens'],
                   'ratio_cmin': p['tokens'] / max(p['cmin'], 1e-9),
                   'bound': p['chi'] * (1 + p['px'] / max(ann['width'] - p['ex'], 1e-9)) * (1 + p['py'] / max(ann['height'] - p['ey'], 1e-9))}
            if did in D:
                mD = metric(did, D[did][0]); row.update(D=mD[:3].tolist(), tokD=D[did][1])
                tw, th = ann['template_size']
                row['SD'] = 2.5 if v == 'gem38' else 44 * p['m'] / (min(tw, th) * (28 if v == 'opus55' else 32))
            rows.append(row)
        if not rows: continue
        out = {'n': len(rows)}
        for name, a, b in (('CS-W', 'CS', 'W'), ('CSdedup-W', 'CSdedup', 'W')):
            out[name] = {'recall': boot([r[a][1] - r[b][1] for r in rows]), 'F1': boot([r[a][0] - r[b][0] for r in rows])}
        out['W'] = {'recall': float(np.mean([r['W'][1] for r in rows])), 'F1': float(np.mean([r['W'][0] for r in rows]))}
        out['CS'] = {'recall': float(np.mean([r['CS'][1] for r in rows])), 'F1': float(np.mean([r['CS'][0] for r in rows]))}
        dr = [r for r in rows if 'D' in r]
        if dr:
            out['D'] = {'recall': float(np.mean([r['D'][1] for r in dr])), 'F1': float(np.mean([r['D'][0] for r in dr])), 'n': len(dr)}
            out['CS-D'] = {'recall': boot([r['CS'][1] - r['D'][1] for r in dr]), 'F1': boot([r['CS'][0] - r['D'][0] for r in dr])}
            out['tok_CS_over_D'] = float(np.sum([r['tokCS'] for r in dr]) / max(1, np.sum([r['tokD'] for r in dr])))
            lo = [r for r in dr if r['SD'] < r['Sw']]
            out['P3_n'] = len(lo)
            if len(lo) >= 3: out['P3'] = {'recall': boot([r['CS'][1] - r['D'][1] for r in lo]), 'F1': boot([r['CS'][0] - r['D'][0] for r in lo])}
        out['tok_CS_over_W'] = float(np.sum([r['tokCS'] for r in rows]) / max(1, np.sum([r['tokW'] for r in rows])))
        out['identity_ok'] = all(r['ratio_cmin'] <= r['bound'] * (1 + 1e-6) for r in rows)
        out['ratio_cmin_median'] = float(np.median([r['ratio_cmin'] for r in rows]))
        out['frac_one_view'] = float(np.mean([r['views'] == 1 for r in rows]))
        res[f'{bench}|{v}'] = {'summary': out, 'rows': rows}
        s = out
        print(f"{bench:4s} {v:8s} n={s['n']:2d} W rec {s['W']['recall']:.2f} CS rec {s['CS']['recall']:.2f}  CS-W rec {s['CS-W']['recall'][0]:+.3f} [{s['CS-W']['recall'][1]:+.3f},{s['CS-W']['recall'][2]:+.3f}]"
              f"  F1 {s['CS-W']['F1'][0]:+.3f} [{s['CS-W']['F1'][1]:+.3f},{s['CS-W']['F1'][2]:+.3f}]"
              + (f"  CS-D rec {s['CS-D']['recall'][0]:+.3f} [{s['CS-D']['recall'][1]:+.3f},{s['CS-D']['recall'][2]:+.3f}] tok CS/D {s['tok_CS_over_D']:.2f}" if 'CS-D' in s else '')
              + (f"  P3(n={s['P3_n']}) {s['P3']['recall'][0]:+.3f} [{s['P3']['recall'][1]:+.3f},{s['P3']['recall'][2]:+.3f}]" if 'P3' in s else f"  P3 n={s.get('P3_n')}")
              + f"  identity {s['identity_ok']} one-view {s['frac_one_view']:.2f} ratio_cmin_med {s['ratio_cmin_median']:.2f}", flush=True)
json.dump(res, open(RC / 'cs_summary.json', 'w'), indent=1)
