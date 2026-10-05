#!/usr/bin/env python3
"""Coupling vs locality: per-instance recall as a function of how many instances share the call.
For every D tile (test set, run 0), count ground-truth centres in the kept core (n) and how many of them
are recovered by that tile's own predictions (one-to-one, tau_sym). Same regions scored for template
matching (dev-chosen threshold), whose decisions are local by construction."""
import json, numpy as np, mech as M, ceilings as K, analyze_test as A
from scipy.optimize import linear_sum_assignment
cands = {d: json.load(open(f'../data/redp40/results/tm_candidates/{d}.json'))['candidates'] for d in K.TEST}
BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 999)]

def hits(pred, gt, tau):
    if not pred or not gt: return 0
    P, G = np.asarray(pred, float), np.asarray(gt, float); D = np.linalg.norm(P[:, None] - G[None], axis=2)
    r, c = linear_sum_assignment(np.where(D <= tau, D, 1e6)); return int((D[r, c] <= tau).sum())

def run(model):
    rows = []
    last = {}
    for l in open(f'results_test/D_{model}_run0.jsonl'):
        r = json.loads(l); last[(r['did'], r['i'], r['j'])] = r
    for (did, i, j), r in last.items():
        if r['status'] != 'success': continue
        a = A.ANN[did]; tau = M.taus(a)[0]
        ca, cb, cc, ce = r['core_px']; x0, y0 = r['origin_z0']; z0, up = r['z0'], r['up']
        inside = lambda x, y: ca <= (x * z0 - x0) * up < cc and cb <= (y * z0 - y0) * up < ce
        gt = [p for p in a['points'] if inside(*p)]
        if not gt: continue
        vlm = M.to_native(r)
        tm = [(x, y) for x, y, s, v in cands[did] if s >= 0.75 and inside(x, y)]
        rows.append((len(gt), hits(vlm, gt, tau), hits(tm, gt, tau), len(vlm)))
    R = np.array(rows)
    out = []
    for lo, hi in BINS:
        s = (R[:, 0] >= lo) & (R[:, 0] <= hi)
        if s.sum() >= 5:
            out.append(f'n={lo}-{hi if hi < 999 else "+"} (tiles {s.sum():3d}): VLM {R[s,1].sum()/R[s,0].sum():.2f} TM {R[s,2].sum()/R[s,0].sum():.2f}')
    # regression of per-tile VLM miss rate on log n (tile-level, weighted)
    x = np.log(R[:, 0]); y = 1 - R[:, 1] / R[:, 0]; yt = 1 - R[:, 2] / R[:, 0]
    b = np.polyfit(x, y, 1, w=np.sqrt(R[:, 0]))[0]; bt = np.polyfit(x, yt, 1, w=np.sqrt(R[:, 0]))[0]
    print(f'{model:8s} ' + ' | '.join(out) + f'  || slope of miss rate per log(n): VLM {b:+.3f}  TM {bt:+.3f}')

for m in A.MODELS:
    run(m)
