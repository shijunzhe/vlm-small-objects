#!/usr/bin/env python3
"""X6 scoring: REDP-10 (23 drawing-class items, 225 instances). Template-matching threshold is the one chosen on the
six REDP-X40 tuning drawings (no tuning on REDP-10). Paired bootstrap over items; also a cluster bootstrap over the
10 drawings (items of one drawing resampled together)."""
import json
from pathlib import Path
import numpy as np
import mech as M
import analyze_test as A

B10 = Path('../data/redp10/redp10'); R10 = Path('results_redp10')
_, _, THR = A.tm_scores()   # REDP-X40 tuning-set threshold, computed before REDP-10 items are added
ITEMS = sorted(p.stem for p in (B10 / 'annotations').glob('r*.json'))
for d in ITEMS:
    A.ANN[d] = json.loads((B10 / 'annotations' / f'{d}.json').read_text()); A.SUB[d] = 'redp10'
M.FORMAT_AWARE = True
A.TEST = ITEMS
DEC = {'gpt56': 'px', 'sonnet55': 'px', 'gem38': 'n1000'}
rng = np.random.default_rng(3)


def at(fn, *a):
    old = A.R; A.R = R10
    try: return fn(*a)
    finally: A.R = old


def tm(thr):
    per = {}
    for d in ITEMS:
        c = json.load(open(f'../data/redp10/results/tm_candidates/{d}.json'))['candidates']
        per[d] = [A.metrics(d, [(x, y) for x, y, s, *_ in c if s >= thr])]
    return per


def cluster_diff(pa, pb, k=0, B=4000):
    a, b = A.mean_per(pa), A.mean_per(pb); ds = [d for d in ITEMS if d in a and d in b]
    by = {}
    for d in ds: by.setdefault(A.ANN[d]['drawing'], []).append(a[d][k] - b[d][k])
    keys = list(by); est = np.mean([x for v in by.values() for x in v])
    bs = []
    for _ in range(B):
        pick = [keys[i] for i in rng.integers(0, len(keys), len(keys))]; bs.append(np.mean([x for kk in pick for x in by[kk]]))
    return est, *np.percentile(bs, [2.5, 97.5])


if __name__ == '__main__':
    thr = THR
    grid = np.round(np.arange(0.30, 0.95, 0.01), 2)
    best = max(grid, key=lambda t: A.summary(tm(t), ITEMS)[0][0]); print('oracle TM threshold on REDP-10:', best)
    S = {'TM': tm(thr), 'TM_oracle': tm(best)}   # oracle: best threshold on REDP-10 itself, an upper bound for TM
    for v in DEC:
        S[('A1', v)] = at(A.score_whole, v, DEC[v]); S[('D', v)] = at(A.score_tiles, 'D', v); S[('H', v)] = at(A.score_H, v)
    print(f'REDP-10: {len(ITEMS)} items, {sum(A.ANN[d]["count"] for d in ITEMS)} instances; TM threshold {thr} (REDP-X40 tuning set)')
    for v in ('gpt56', 'sonnet55'):
        f = R10 / 'agent' / f'{v}_run0.jsonl'
        if f.exists():
            per = {}
            for l in open(f):
                r = json.loads(l)
                if r['status'] not in ('success', 'parse_error'): continue
                sc = r['scale']; pts = [(p[0] / sc, p[1] / sc) for p in (M.point_of(q) for q in r.get('shapes', [])) if p] if r['status'] == 'success' else []
                per[r['did']] = [A.metrics(r['did'], pts)]
            S[('AG', v)] = per
    out = {}
    for k, per in S.items():
        s, n = A.summary(per, ITEMS)
        if s is None: continue
        mult = [d for d in ITEMS if A.ANN[d]['n_classes_in_legend'] > 1]
        sm, _ = A.summary(per, mult)
        print(f'{str(k):22s} F1 {s[0]:.2f} R {s[1]:.2f} P {s[2]:.2f} relCountErr {s[3]:.2f}  (n={n})  multi-class drawing F1 {sm[0] if sm is not None else float("nan"):.2f}')
        out[str(k)] = {'f1': s[0], 'recall': s[1], 'precision': s[2], 'count_err': s[3], 'n': n}
    print('\npaired differences (item bootstrap | drawing-cluster bootstrap)')
    for v in DEC:
        for a_, b_ in ((('A1', v), 'TM'), (('D', v), ('A1', v)), (('H', v), ('A1', v)), (('D', v), 'TM'), (('H', v), 'TM'), (('H', v), ('D', v)), (('AG', v), ('A1', v)), (('AG', v), ('D', v)), (('AG', v), 'TM')):
            if a_ not in S or b_ not in S: continue
            if not A.mean_per(S[a_]) or not A.mean_per(S[b_]): continue
            d1 = A.boot_diff(S[a_], S[b_], ITEMS); d2 = cluster_diff(S[a_], S[b_])
            print(f'  {str(a_):20s} - {str(b_):20s} {d1[0]:+.2f} [{d1[1]:+.2f}, {d1[2]:+.2f}] | [{d2[1]:+.2f}, {d2[2]:+.2f}]')
            out[f'{a_}-{b_}'] = {'item': list(d1), 'cluster': list(d2)}
    import math
    for v, t in (('gpt56', 32), ('sonnet55', 28), ('gem38', None)):
        ss = []
        for d in ITEMS:
            a = A.ANN[d]; W, H = a['width'], a['height']; m = math.sqrt(a['template_size'][0] * a['template_size'][1]); sc = min(1.0, 1600 / max(W, H))
            ss.append(m * sc / t if t else m / (max(W, H) / math.sqrt(1090)))
        out[f'S_A1|{v}'] = {'median': float(np.median(ss)), 'min': float(min(ss)), 'max': float(max(ss))}
        print('A1 S', v, round(float(np.median(ss)), 2), round(min(ss), 2), round(max(ss), 2))
    json.dump(out, open(R10 / 'x6_scores.json', 'w'), indent=1, default=float)
