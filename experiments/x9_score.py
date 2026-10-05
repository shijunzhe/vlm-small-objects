#!/usr/bin/env python3
"""X9 (prereg v3.3): confound controls on the 34 REDP-X40 test drawings. Paired bootstrap (4000) and sign-flip
permutation p over drawings; Holm across X9-H1 to H3 primary tests."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import mech as M, ceilings as K
import analyze_test as A
import x_new_score as X

M.FORMAT_AWARE = True
GEN = Path('results_gen'); AG = Path('results_x/agent')
T = K.TEST; SH = [d for d in T if A.SUB[d] == 'sheet']
rng = np.random.default_rng(11)
OUT = {}


def agent(v, tags):
    per = defaultdict(list)
    for tag in tags:
        f = AG / f'{v}_{tag}.jsonl'
        if not f.exists(): continue
        for l in open(f):
            r = json.loads(l)
            if r['status'] not in ('success', 'parse_error'): continue
            s = r['scale']; pts = [(p[0] / s, p[1] / s) for p in (M.point_of(q) for q in r.get('shapes', [])) if p] if r['status'] == 'success' else []
            per[r['did']].append(A.metrics(r['did'], pts))
    return per


def native(v, runs):
    per = defaultdict(list)
    for run in runs:
        f = GEN / f'A1nat_{v}_run{run}.jsonl'
        if not f.exists(): continue
        for l in open(f):
            r = json.loads(l)
            if r['status'] not in ('success', 'parse_error'): continue
            s = r['scale']; pts = [(p[0] / s, p[1] / s) for p in (M.point_of(q) for q in r.get('shapes', [])) if p] if r['status'] == 'success' else []
            per[r['did']].append(A.metrics(r['did'], pts))
    return per


def test(pa, pb, ds, k=0):
    a, b = A.mean_per(pa), A.mean_per(pb); ds = [d for d in ds if d in a and d in b]
    x = np.array([a[d][k] - b[d][k] for d in ds])
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
    perm = np.mean([abs((x * rng.choice([-1, 1], len(x))).mean()) >= abs(x.mean()) for _ in range(4000)])
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), float(perm), len(ds)


def f1(per, ds):
    s = A.summary(per, ds)[0]; return float(s[0]) if s is not None else float('nan')


def fmt(t): return f'{t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}] p={t[3]:.3f} (n={t[4]})'


if __name__ == '__main__':
    S = {}
    for v, dr, ar in (('gpt56', 'gpt56m', 'gpt56m'), ('sonnet55', 'sonnet55t', 'sonnet55t'), ('gem38', 'gem38t', 'gem38t')):
        S[('D', v)] = X.at(A.score_tiles, 'D', v); S[('DR', v)] = X.at(A.score_tiles, 'D', dr)
        S[('A1R', v)] = X.at(A.score_whole, ar, X.DEC.get(ar, 'px') if ar != 'gpt56m' else 'px')
        S[('AG', v)] = agent(v, ['run0', 'run1']); S[('OVZ', v)] = agent(v, ['ovzoom_run0', 'ovzoom_run1']); S[('OVC', v)] = agent(v, ['ovcrop_run0'])
    S[('NAT', 'gpt56')] = native('gpt56orig', [0, 1]); S[('NATR', 'gpt56')] = native('gpt56origm', [0])
    print('F1 on full sheets (all drawings)')
    for (c, v), per in sorted(S.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if A.mean_per(per): print(f'  {v:9s} {c:5s} {f1(per, SH):.2f} ({f1(per, T):.2f})')
        OUT[f'{c}|{v}'] = {'sheets': f1(per, SH), 'all': f1(per, T)}
    prim = []
    for v in ('gpt56', 'sonnet55'):
        t = test(S[('AG', v)], S[('DR', v)], SH); prim.append(('X9-H1 ' + v, t)); OUT[f'H1|{v}'] = t
    t = test(S[('NATR', 'gpt56')], S[('DR', 'gpt56')], SH); prim.append(('X9-H2 gpt56', t)); OUT['H2|gpt56'] = t
    ps = sorted(range(len(prim)), key=lambda i: prim[i][1][3]); m = len(prim); holm = {}
    run = 0
    for r, i in enumerate(ps):
        run = max(run, min(1, (m - r) * prim[i][1][3])); holm[i] = run
    print('\nprimary tests (sheets), Holm-adjusted permutation p')
    for i, (lab, t) in enumerate(prim): print(f'  {lab:14s} {fmt(t)} holm p={holm[i]:.3f}'); OUT[lab + '|holm'] = holm[i]
    print('\nX9-H3 decomposition on sheets (each step minus the previous)')
    for v in ('gpt56', 'sonnet55'):
        steps = [('A1R', 'whole + reasoning'), ('OVC', 'crop only'), ('OVZ', 'upsampled crops'), ('AG', 'native zoom')]
        for (a, la), (b, lb) in zip(steps[1:], steps[:-1]):
            t = test(S[(a, v)], S[(b, v)], SH); print(f'  {v:9s} {la:18s} - {lb:18s} {fmt(t)}'); OUT[f'H3|{v}|{a}-{b}'] = t
    print('\nsecondary')
    for v in ('gpt56', 'sonnet55', 'gem38'):
        t = test(S[('DR', v)], S[('D', v)], SH); print(f'  {v:9s} tiles+R - tiles      {fmt(t)}'); OUT[f'DR-D|{v}'] = t
        t = test(S[('AG', v)], S[('DR', v)], T); print(f'  {v:9s} agent - tiles+R (all) {fmt(t)}'); OUT[f'AG-DR-all|{v}'] = t
    t = test(S[('NATR', 'gpt56')], S[('NAT', 'gpt56')], SH); print(f'  gpt56 native+R - native {fmt(t)}'); OUT['NATR-NAT'] = t
    for k, lab in ((1, 'recall'), (3, 'count error')):
        a = A.summary(S[('NAT', 'gpt56')], SH)[0][k]; b = A.summary(S[('NATR', 'gpt56')], SH)[0][k]
        print(f'  gpt56 native {lab}: no reasoning {a:.2f}, reasoning {b:.2f}')
    json.dump(OUT, open('results_x/x9_scores.json', 'w'), indent=1)
