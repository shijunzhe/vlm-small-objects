#!/usr/bin/env python3
"""Independent validation (design v3 6.4). usage: phase_val.py freeze | run | analyze"""
import json, sys, math, random
from collections import defaultdict
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import vspec as V, vrun as R, phase2 as P2

ROOT = V.ROOT; OUT = ROOT / 'results'; N0 = 8; B = 2000; K = 32


def c_cons(model, seed=0):
    st = P2.state().get((model, 2.0), {})
    if not st:
        return None, None
    ids = sorted({i for v in st.values() for i, _ in v})
    by = {n: defaultdict(list) for n in st}
    for n, v in st.items():
        for i, r in v:
            by[n][i].append(r)
    rng = random.Random(seed); A = []
    for b in range(B):
        samp = [rng.choice(ids) for _ in ids]
        pooled = {n: R.pooled([r for i in samp for r in by[n][i]]) for n in st}
        a, _ = P2.bracket(pooled)
        A.append(a if a is not None else 0)   # no passing level: below n0
    pooled = {n: R.pooled([r for _, r in v]) for n, v in st.items()}
    return float(np.quantile(A, 0.05, method='lower')), P2.bracket(pooled)


def freeze():
    fr = P2.frozen(); out = {}
    for m in V.ORDER:
        S = P2.scale(m, 2.0)
        cc, br = c_cons(m)
        if cc is None:
            out[m] = dict(status='not testable: x = 2 geometrically unreachable at n0', S=S); continue
        n_safe = N0 if cc <= N0 else int(math.floor(N0 + 0.5 * (cc - N0)))
        out[m] = dict(S=S, C_ref_bracket=br, C_cons=cc, n_safe=n_safe, k=K)
    json.dump(out, open(ROOT / 'frozen_validation.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


def items():
    fz = json.load(open(ROOT / 'frozen_validation.json'))
    plan = {m: v for m, v in fz.items() if 'n_safe' in v}
    by_n = defaultdict(list)
    for m, v in plan.items():
        by_n[v['n_safe']].append((m, v['S']))
    its = []
    for n, renders in by_n.items():
        for i in range(K):
            lay, k = V.frozen_layout('P3', i, n, renders)
            if lay is None:
                print('unreachable layout', i, n); continue
            for m, S in renders:
                it, r = R.make_item('val', lay, m, S, 'safe')
                if it: its.append(it)
                else: print('no item', m, i, n, r)
    json.dump(its, open(ROOT / 'items_val.json', 'w'))
    return its


def analyze():
    fz = json.load(open(ROOT / 'frozen_validation.json'))
    its = json.load(open(ROOT / 'items_val.json')); recs = R.load_results([OUT / 'validation.jsonl'])
    rows = defaultdict(list)
    for it in its:
        if it['id'] in recs:
            rows[it['model']].append((it['i'], R.score_record(it, recs[it['id']])))
    rng = random.Random(1); res = {}
    for m, v in rows.items():
        p = R.pooled([r for _, r in v]); ids = sorted({i for i, _ in v}); by = defaultdict(list)
        for i, r in v:
            by[i].append(r)
        Rs, Ps = [], []
        for b in range(B):
            q = R.pooled([r for i in [rng.choice(ids) for _ in ids] for r in by[i]]); Rs.append(q['R']); Ps.append(q['P'] if q['P'] == q['P'] else 0)
        lbR, lbP = float(np.quantile(Rs, 0.05)), float(np.quantile(Ps, 0.05))
        if not (p['R'] >= 0.9 and p['P'] >= 0.9):
            verdict = 'not validated'
        elif lbR >= 0.9 and lbP >= 0.9:
            verdict = 'validated with one-sided 95% support'
        else:
            verdict = 'meets q on the validation sample'
        res[m] = dict(fz[m], R=p['R'], P=p['P'], lb95_R=lbR, lb95_P=lbP, k=p['k'], trunc=p['trunc'], verdict=verdict)
    json.dump(res, open(OUT / 'validation_analysis.json', 'w'), indent=1)
    for m, r in res.items():
        print(m, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items()})


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'freeze':
        freeze()
    elif cmd == 'run':
        R.run_items(items(), OUT / 'validation.jsonl', workers=12)
    elif cmd == 'analyze':
        analyze()
