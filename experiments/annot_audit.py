#!/usr/bin/env python3
"""X7: annotation audit candidates. Pools detections of independent methods on all 40 REDP-X40 drawings and lists
(a) locations found by >= 3 methods with no annotation within tau (possible missed annotations) and
(b) annotations found by none of the methods (possible spurious or misplaced annotations)."""
import json, glob, math
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import mech as M, ceilings as K
M.FORMAT_AWARE = True
DIDS = K.DEV + K.TEST
ANN = {d: json.loads((M.BENCH / 'annotations' / f'{d}.json').read_text()) for d in DIDS}

def tiles_pts(path, dids):
    out = {d: [] for d in dids}; last = {}
    for l in open(path):
        r = json.loads(l); last[(r['did'], r['i'], r['j'], r.get('half', ''))] = r
    for r in last.values():
        if r['status'] == 'success' and r['did'] in out: out[r['did']] += M.to_native(r)
    return out
def hyb_pts(path):
    out = {}
    for l in open(path):
        r = json.loads(l)
        if r.get('match') is True: out.setdefault(r['did'], []).append((r['x'], r['y']))
    return out
methods = {}
for d in DIDS:
    c = json.load(open(f'../data/redp40/results/tm_candidates/{d}.json'))['candidates']
    methods.setdefault('TM', {})[d] = [(x, y) for x, y, s, v in c if s >= 0.75]
for name, path in [('D_sonnet55', 'results_gen/D_sonnet55_run0.jsonl'), ('D_gem38', 'results_gen/D_gem38_run0.jsonl'),
                   ('D_gpt56', 'results_gen/D_gpt56_run0.jsonl'), ('D_gemma4', 'results_test/D_gemma4_run0.jsonl')]:
    methods[name] = tiles_pts(path, K.TEST)
for name, path in [('H_sonnet55', 'results_gen/hyb_sonnet55.jsonl'), ('H_qwen', 'results_test/hyb_qwen3vl.jsonl')]:
    methods[name] = hyb_pts(path)
items = []
for d in K.TEST:
    a = ANN[d]; tau = M.taus(a)[0]; G = np.array(a['points'], float)
    allp = [(x, y, n) for n, m in methods.items() for (x, y) in m.get(d, [])]
    # cluster detections (greedy, radius tau)
    clusters = []
    for x, y, n in allp:
        for c in clusters:
            if (c['x'] - x) ** 2 + (c['y'] - y) ** 2 <= tau ** 2:
                c['m'].add(n); c['pts'].append((x, y)); break
        else:
            clusters.append({'x': x, 'y': y, 'm': {n}, 'pts': [(x, y)]})
    for c in clusters:
        cx, cy = np.mean(c['pts'], axis=0)
        if len(c['m']) >= 3 and (len(G) == 0 or np.min(np.hypot(G[:, 0] - cx, G[:, 1] - cy)) > tau):
            items.append({'did': d, 'kind': 'possible_missed', 'x': float(cx), 'y': float(cy), 'methods': sorted(c['m'])})
    for g in G:
        found = [n for n, m in methods.items() if any(math.hypot(x - g[0], y - g[1]) <= tau for x, y in m.get(d, []))]
        if not found:
            items.append({'did': d, 'kind': 'possible_spurious', 'x': float(g[0]), 'y': float(g[1]), 'methods': []})
json.dump(items, open('results_x/annot_audit_items.json', 'w'), indent=1)
from collections import Counter
print(Counter(i['kind'] for i in items), 'of', sum(ANN[d]['count'] for d in K.TEST), 'test annotations')
