#!/usr/bin/env python3
"""X5 v2: learned counters with validated wrappers. Label-free detection cap (top 2000 by score per item);
thresholds on tuning items only (REDP-X40: 6 tuning drawings; FPC: 30 dev items). GeCo2 scores relative to the tile max."""
import json, sys
from pathlib import Path
import numpy as np
import mech as M, ceilings as K
import analyze_test as A

R = Path('results_x/learned2'); R1 = Path('results_x/learned')


def nms(pts, r):
    pts = sorted(pts, key=lambda p: -p[2]); keep = []
    for p in pts:
        if all((p[0] - k[0]) ** 2 + (p[1] - k[1]) ** 2 > r * r for k in keep): keep.append(p)
    return keep


def load(f, method):
    out = {}
    for l in open(f):
        r = json.loads(l)
        pts = [[p[0], p[1], p[2] / max(p[5], 1e-9) if method == 'geco2' else p[2]] for p in r['points']]
        out[r['did']] = sorted(pts, key=lambda p: -p[2])[:2000]
    return out


def evaluate(dets, dev, test, metrics):
    def f1(d, t):
        tau = M.taus(A.ANN[d])[0]; return metrics(d, [(x[0], x[1]) for x in nms([p for p in dets[d] if p[2] >= t], tau)])
    sc = [p[2] for d in dev if d in dets for p in dets[d]]
    grid = np.unique(np.quantile(sc, np.linspace(0, 0.99, 40)))
    th = float(max(grid, key=lambda t: np.nanmean([f1(d, t)[0] for d in dev if d in dets])))
    per = {d: [f1(d, th)] for d in test if d in dets}
    return th, per


if __name__ == '__main__':
    out = {}
    print('REDP-X40 test (34): method, mode -> F1 (sheets/views)')
    for f in sorted(list(R.glob('*_redp_x40_*.jsonl')) + list(R1.glob('*_redp_x40.jsonl'))):
        method = f.stem.split('_')[0]; mode = f.stem.split('redp_x40')[1].strip('_') or 'tile-44-legend(v1)'
        th, per = evaluate(load(f, method), K.DEV, K.TEST, A.metrics)
        s = A.summary(per, K.TEST)[0]; sh = A.summary(per, [d for d in K.TEST if A.SUB[d] == 'sheet'])[0]; vw = A.summary(per, [d for d in K.TEST if A.SUB[d] == 'view'])[0]
        print(f'  {method:8s} {mode:24s} F1 {s[0]:.3f} ({sh[0]:.2f}/{vw[0]:.2f})  R {s[1]:.2f} P {s[2]:.2f}  thr {th:.3f}')
        out[f'redp|{method}|{mode}'] = {'f1': [s[0], sh[0], vw[0]], 'recall': s[1], 'precision': s[2], 'thr': th}
    import fpc_score as FS
    print('FPC-300 test (339): method, mode -> F1')
    for f in sorted(R.glob('*_fpc_*.jsonl')):
        method = f.stem.split('_')[0]; mode = f.stem.split('fpc_')[1]
        th, per = evaluate(load(f, method), FS.DEV, FS.TEST, FS.metrics)
        s = FS.mean(per, FS.TEST)
        print(f'  {method:8s} {mode:10s} F1 {s[0]:.3f} n={s[1]} thr {th:.3f}')
        out[f'fpc|{method}|{mode}'] = {'f1': s[0], 'n': s[1], 'thr': th, 'per': {d: v[0].tolist() for d, v in per.items()}}
    json.dump(out, open(R / 'lb2_scores.json', 'w'), indent=1)
