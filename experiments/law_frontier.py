#!/usr/bin/env python3
"""Recall-cost frontier check on REDP-X40 test drawings (plan v5, section 4).
For a policy whose views are chosen without knowledge of target positions, average recall <= Rhat(u), u = B m^2 / |D|,
where B = image tokens = sum_V |V| S_V^2 / m^2 (c = 1, the weakest valid case) and Rhat is the least concave majorant of
g(u) = R(sqrt(u)), R(S) = sigmoid(a_v + delta + b_v log S) (law at n = 1, A = m^2: its upper envelope over content).
Reports, per model and configuration, u, observed recall, frontier, and the share of drawings above the frontier."""
import json, math
from collections import defaultdict
import numpy as np
import analyze_test as A, ceilings as K, x_new_score as X
import law_data as LD

F = json.load(open('results_law/law_fit.json')); D = json.load(open('results_law/law_data.json'))
delta = F['delta']['redp']


def Rhat_fn(v):
    a, b = F['params'][v][0], F['params'][v][1]
    us = np.concatenate([[0.0], np.logspace(-4, 4, 4000)])
    g = np.where(us > 0, 1 / (1 + np.exp(-(a + delta + 0.5 * b * np.log(np.maximum(us, 1e-12))))), 0.0)
    # least concave majorant via upper convex hull of (u, g)
    hull = []
    for p in zip(us, g):
        while len(hull) >= 2 and (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= (p[1] - hull[-2][1]) * (hull[-1][0] - hull[-2][0]):
            hull.pop()
        hull.append(p)
    hx, hy = np.array(hull).T
    return lambda u: float(np.interp(u, hx, hy))


def agent_calls(v, tag='run0'):
    """Overview and zoom views of the agent as calls [S, n(unused), A_units]."""
    out = {}
    f = X.XR / 'agent' / f'{v}_{tag}.jsonl'
    for l in open(f):
        r = json.loads(l)
        if r['status'] not in ('success', 'parse_error'): continue
        a = A.ANN[r['did']]; m = LD.gm(a); s = r['scale']; ow, oh = r['overview']
        calls = [[LD.S_of(v, ow, oh, m * s), 0, a['width'] * a['height'] / m ** 2]]
        for z in r['log']['zooms']:
            x0, y0, x1, y1 = z['native_box']; w, h = z['shown']; zz = w / max(1, x1 - x0)
            calls.append([LD.S_of(v, w, h, m * zz), 0, (x1 - x0) * (y1 - y0) / m ** 2])
        out[r['did']] = calls
    return out


rows = []
for r in D:
    if r['ds'] == 'redp' and r['split'] == 'test' and r['config'] in ('A1', 'D'):
        a = A.ANN[r['item']]; Ad = a['width'] * a['height'] / LD.gm(a) ** 2
        u = sum(c[2] * c[0] ** 2 for c in r['calls']) / Ad
        rows.append({'model': r['model'], 'config': r['config'], 'item': r['item'], 'u': u, 'obs': r['obs']})
for v in ('gpt56', 'sonnet55', 'gem38'):
    per, _ = X.agent_per(v); mp = A.mean_per(per); calls = agent_calls(v)
    for d, cs in calls.items():
        if d not in mp or d not in K.TEST: continue
        a = A.ANN[d]; Ad = a['width'] * a['height'] / LD.gm(a) ** 2
        rows.append({'model': v, 'config': 'AG', 'item': d, 'u': sum(c[2] * c[0] ** 2 for c in cs) / Ad, 'obs': float(mp[d][1])})
fr = {v: Rhat_fn(v) for v in F['params']}
summary = {}
for r in rows:
    r['frontier'] = fr[r['model']](r['u']); r['above'] = r['obs'] > r['frontier'] + 1e-9
g = defaultdict(list)
for r in rows: g[(r['model'], r['config'])].append(r)
print('model     cfg  n   median u   mean obs  mean frontier  share above')
for k, rs in sorted(g.items()):
    summary['|'.join(k)] = {'n': len(rs), 'u_med': float(np.median([r['u'] for r in rs])), 'obs': float(np.mean([r['obs'] for r in rs])),
                           'frontier': float(np.mean([r['frontier'] for r in rs])), 'above': float(np.mean([r['above'] for r in rs]))}
    s = summary['|'.join(k)]
    print(f'{k[0]:9s} {k[1]:3s} {s["n"]:3d}  {s["u_med"]:8.2f}   {s["obs"]:.2f}      {s["frontier"]:.2f}         {s["above"]:.2f}')
json.dump({'summary': summary, 'rows': rows}, open('results_law/law_frontier.json', 'w'), indent=1)
