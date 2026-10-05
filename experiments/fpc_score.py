#!/usr/bin/env python3
"""Scoring for FPC-300 and FPC-sheets (prereg v4/v4.1). Strict F1 within tau = m/2; predictions and labels within m/2
of any block edge are ignored. Paired bootstrap over items (FPC-300) or source drawings (sheets)."""
import json, math, glob, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import mech as M
import analyze_test as A

M.FORMAT_AWARE = True
FPC = Path('../data/fpc'); RF = Path('results_fpc')
DEC = {'gpt': 'px', 'gpt56': 'px', 'claude': 'px_claude', 'sonnet55': 'px', 'gemini': 'n1000', 'gemma4': 'n1000', 'qwen3vl': 'n1000',
       'gem38': 'n1000', 'gem38lo': 'n1000', 'gem38uh': 'n1000', 'gpt61': 'px', 'opus55': 'px'}
NAME = {'gpt': 'gpt54', 'claude': 'sonnet46', 'gemini': 'gemini25'}
ANN = {}
for b in ('bench', 'sheets'):
    for f in glob.glob(str(FPC / b / 'annotations' / '*.json')):
        a = json.load(open(f)); ANN[a['id']] = a; A.ANN[a['id']] = a; A.SUB[a['id']] = a['subset']
MAN = json.load(open(FPC / 'bench' / 'manifest.json')); DEV, TEST = MAN['dev'], MAN['test']
SHEETS = sorted(i for i in ANN if i.startswith('s'))
GEO = json.load(open(FPC / 'geometry.json'))
_orig = A.metrics


def near_edge(x, y, rects, band):
    for x0, y0, x1, y1 in rects:
        if x0 <= x < x1 and y0 <= y < y1:
            return min(x - x0, x1 - x, y - y0, y1 - y) < band
    return True


import os
BLOCK0 = os.environ.get('REGION') == 'block0'   # post hoc diagnostic: score only the first block, identical across k


def metrics(did, pts):
    if did not in ANN: return _orig(did, pts)
    a = ANN[did]; tau = M.taus(a)[0]; rects = a['ignore_rects']
    if BLOCK0 and did.startswith('s'): rects = [[0, 0, 1000, 1000]]
    gt = [p for p in a['points'] if not near_edge(p[0], p[1], rects, tau)]
    pr = [p for p in pts if not near_edge(p[0], p[1], rects, tau)]
    g = len(gt)
    if g == 0: return np.array([np.nan] * 5)
    tp, n, _ = M.match(pr, gt, tau)
    return np.array([2 * tp / (n + g), tp / g, tp / max(n, 1), abs(n - g) / g, n / g])


A.metrics = metrics


def at(bench, fn, *a):
    old = A.R; A.R = RF / bench
    try: return fn(*a)
    finally: A.R = old


def cond(bench, c, v):
    if c == 'A1': return at(bench, A.score_whole, v, DEC[v])
    if c == 'D': return at(bench, A.score_tiles, 'D', v)
    if c == 'AG':
        per = defaultdict(list); f = RF / bench / 'agent' / f'{v}_run0.jsonl'
        if f.exists():
            for l in open(f):
                r = json.loads(l)
                if r['status'] not in ('success', 'parse_error'): continue
                s = r['scale']; pts = [(p[0] / s, p[1] / s) for p in (M.point_of(q) for q in r.get('shapes', [])) if p] if r['status'] == 'success' else []
                per[r['did']].append(metrics(r['did'], pts))
        return per
    if c == 'TM':
        per = {}
        th = TM_THR
        for d in (TEST + DEV if bench == 'bench' else SHEETS):
            fp = FPC / f'tm_{bench}' / f'{d}.json'
            if fp.exists():
                cands = json.load(open(fp))['candidates']; per[d] = [metrics(d, [(x, y) for x, y, s, *_ in cands if s >= th])]
        return per


def mean(per, ds, k=0):
    mp = A.mean_per(per); v = [mp[d][k] for d in ds if d in mp and not np.isnan(mp[d][k])]
    return float(np.mean(v)) if v else float('nan'), len(v)


rng = np.random.default_rng(5)


def diff(pa, pb, ds, k=0, cluster=None):
    a, b = A.mean_per(pa), A.mean_per(pb); ds = [d for d in ds if d in a and d in b and not np.isnan(a[d][k]) and not np.isnan(b[d][k])]
    if not ds: return (np.nan,) * 4
    x = {d: a[d][k] - b[d][k] for d in ds}
    groups = defaultdict(list)
    for d in ds: groups[cluster(d) if cluster else d].append(x[d])
    keys = list(groups); est = np.mean(list(x.values()))
    bs = [np.mean([v for kk in (keys[i] for i in rng.integers(0, len(keys), len(keys))) for v in groups[kk]]) for _ in range(4000)]
    return float(est), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(ds)


def tm_threshold():
    if not all((FPC / 'tm_bench' / f'{d}.json').exists() for d in DEV): return None
    grid = np.round(np.arange(0.3, 0.96, 0.01), 2); best = None
    for t in grid:
        f = []
        for d in DEV:
            cands = json.load(open(FPC / 'tm_bench' / f'{d}.json'))['candidates']; f.append(metrics(d, [(x, y) for x, y, s, *_ in cands if s >= t])[0])
        f = np.nanmean(f)
        if best is None or f > best[0]: best = (f, float(t))
    return best[1]


TM_THR = None
if __name__ == '__main__':
    TM_THR = tm_threshold(); print('TM threshold (FPC dev):', TM_THR)
    out = {'tm_threshold': TM_THR}
    models = ['gpt', 'claude', 'gemini', 'gemma4', 'qwen3vl', 'gpt56', 'sonnet55', 'gem38', 'gem38lo', 'gem38uh']
    S = {}
    for b in ('bench', 'sheets'):
        for v in models:
            for c in ('A1', 'D', 'AG'):
                per = cond(b, c, v)
                if A.mean_per(per): S[(b, c, v)] = per
        if TM_THR is not None: S[(b, 'TM', 'tm')] = cond(b, 'TM', None)
    print('\nFPC-300 test (n=339): F1 A1 / D / AG')
    for v in models + ['tm']:
        row = [S.get(('bench', c, v)) for c in ('A1', 'D', 'AG')] if v != 'tm' else [S.get(('bench', 'TM', 'tm'))]
        print(f'  {NAME.get(v, v):9s} ' + '  '.join(f'{mean(p, TEST)[0]:.2f}(n={mean(p, TEST)[1]})' if p else '   -   ' for p in row))
        out[f'bench|{v}'] = [mean(p, TEST)[0] if p else None for p in row]
    print('\nF-H1/H2: D - A1 on FPC-300 test')
    for v in models:
        if ('bench', 'D', v) in S and ('bench', 'A1', v) in S:
            d = diff(S[('bench', 'D', v)], S[('bench', 'A1', v)], TEST); print(f'  {NAME.get(v, v):9s} {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] n={d[3]}'); out[f'H12|{v}'] = d
    print('\nFPC-sheets: A1 F1 / recall by k, D F1 by k, AG F1 by k')
    clus = lambda d: d.split('_')[0]
    for v in models + ['tm']:
        cells = []
        for c in (('A1', 'D', 'AG') if v != 'tm' else ('TM',)):
            p = S.get(('sheets', c, v))
            if not p: cells.append('-'); continue
            cells.append(c + ' ' + ' '.join(f'{mean(p, [d for d in SHEETS if d.endswith(f"_k{k}")])[0]:.2f}' for k in (1, 2, 3)))
        print(f'  {NAME.get(v, v):9s} ' + ' | '.join(cells))
        for c in ('A1', 'D', 'AG', 'TM'):
            p = S.get(('sheets', c, v))
            if p: out[f'sheets|{c}|{v}'] = [mean(p, [d for d in SHEETS if d.endswith(f'_k{k}')])[0] for k in (1, 2, 3)]
    # k3 - k1 paired by source drawing
    def kpair(per, k, kk=1, metric=0):
        mp = A.mean_per(per); x = {}
        for d in SHEETS:
            if d.endswith(f'_k{k}'):
                b = d[:-3] + f'_k{kk}'
                if d in mp and b in mp and not np.isnan(mp[d][metric]) and not np.isnan(mp[b][metric]): x[d] = mp[d][metric] - mp[b][metric]
        v = np.array(list(x.values())); bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(4000)] if len(v) else [np.nan]
        return float(v.mean()) if len(v) else np.nan, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(v)
    print('\nF-H4/H5/H6: k3 - k1 (F1; recall for H5)')
    for v in models:
        for c in ('A1', 'D', 'AG'):
            p = S.get(('sheets', c, v))
            if not p: continue
            t = kpair(p, 3); tr = kpair(p, 3, metric=1)
            print(f'  {NAME.get(v, v):9s} {c:3s} F1 {t[0]:+.2f} [{t[1]:+.2f}, {t[2]:+.2f}]  recall {tr[0]:+.2f} [{tr[1]:+.2f}, {tr[2]:+.2f}] n={t[3]}')
            out[f'k3k1|{c}|{v}'] = [t, tr]

    # ---- T1 (budget at fixed pixels), F-H3, F-H8 ----
    from scipy.stats import spearmanr
    for b, ids in (('bench', TEST), ('sheets', SHEETS)):
        trio = [S.get((b, 'A1', v)) for v in ('gem38lo', 'gem38', 'gem38uh')]
        if not all(trio): continue
        small = [d for d in ids if GEO[d]['gem38lo']['A1'] < 1.5]; large = [d for d in ids if GEO[d]['gem38lo']['A1'] >= 1.5]
        for lab, ds in (('all', ids), ('small', small), ('large', large)):
            dl = diff(trio[1], trio[0], ds, cluster=(lambda d: d.split('_')[0]) if b == 'sheets' else None)
            ud = diff(trio[2], trio[1], ds, cluster=(lambda d: d.split('_')[0]) if b == 'sheets' else None)
            out[f'T1|{b}|{lab}'] = {'f1': [mean(t, ds)[0] for t in trio], 'def_lo': dl, 'uh_def': ud, 'n': len(ds)}
            print(f'T1 {b:6s} {lab:5s} n={len(ds):3d} F1 lo/def/uh ' + '/'.join(f'{mean(t, ds)[0]:.2f}' for t in trio) + f'  def-lo {dl[0]:+.2f} [{dl[1]:+.2f},{dl[2]:+.2f}]  uh-def {ud[0]:+.2f} [{ud[1]:+.2f},{ud[2]:+.2f}]')
    for v in ('gemini', 'gemma4', 'gem38', 'gpt56', 'sonnet55'):
        gv = {'gemini': 'gemini25'}.get(v, v)
        if ('bench', 'D', v) in S and ('bench', 'A1', v) in S:
            a, d_ = A.mean_per(S[('bench', 'A1', v)]), A.mean_per(S[('bench', 'D', v)])
            ds = [d for d in TEST if d in a and d in d_ and not np.isnan(a[d][0]) and not np.isnan(d_[d][0])]
            r = spearmanr([GEO[d][gv]['A1'] for d in ds], [d_[d][0] - a[d][0] for d in ds])
            out[f'H3|{v}'] = [float(r[0]), float(r[1]), len(ds)]; print(f'F-H3 {v}: rho(S_A1, D-A1) {r[0]:+.2f} p={r[1]:.3f} n={len(ds)}')
    for v in ('gpt', 'claude', 'gemini', 'gemma4', 'qwen3vl', 'gpt56', 'sonnet55', 'gem38'):
        gv = {'gpt': 'gpt54', 'claude': 'sonnet46', 'gemini': 'gemini25'}.get(v, v)
        p = S.get(('sheets', 'A1', v))
        if not p: continue
        a = A.mean_per(p); ds = [d for d in SHEETS if d in a and not np.isnan(a[d][0])]
        r = spearmanr([GEO[d][gv]['A1'] for d in ds], [a[d][0] for d in ds]); out[f'H8|{v}'] = [float(r[0]), float(r[1]), len(ds)]
        print(f'F-H8 {v}: rho(S, F1) sheets {r[0]:+.2f} p={r[1]:.3f}')

    # ---- F-H4 (decline larger for low-S group) and F-H7 (agent >= tiles at k = 3) ----
    def k_items(per, k, metric=0):
        mp = A.mean_per(per); return {d[:-3]: mp[d][metric] for d in SHEETS if d.endswith(f'_k{k}') and d in mp and not np.isnan(mp[d][metric])}
    low = [v for v in ('gemini', 'gemma4', 'gem38') if ('sheets', 'A1', v) in S]; high = [v for v in ('gpt56', 'sonnet55') if ('sheets', 'A1', v) in S]
    if low and high:
        dec = {}
        for v in low + high:
            k1, k3 = k_items(S[('sheets', 'A1', v)], 1), k_items(S[('sheets', 'A1', v)], 3)
            dec[v] = {s_: k1[s_] - k3[s_] for s_ in k1 if s_ in k3}
        srcs = [s_ for s_ in dec[low[0]] if all(s_ in dec[v] for v in low + high)]
        x = np.array([np.mean([dec[v][s_] for v in low]) - np.mean([dec[v][s_] for v in high]) for s_ in srcs])
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
        out['H4'] = [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x)]
        print(f'F-H4 decline(low-S group) - decline(high-S group) {x.mean():+.2f} [{out["H4"][1]:+.2f}, {out["H4"][2]:+.2f}] n={len(x)} low={low} high={high}')
    for v in ('gpt56', 'sonnet55'):
        if ('sheets', 'AG', v) in S and ('sheets', 'D', v) in S:
            a3, d3 = k_items(S[('sheets', 'AG', v)], 3), k_items(S[('sheets', 'D', v)], 3)
            x = np.array([a3[s_] - d3[s_] for s_ in a3 if s_ in d3])
            bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
            out[f'H7|{v}'] = [float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x)]
            print(f'F-H7 {v}: AG - D at k=3 {x.mean():+.2f} [{out[f"H7|{v}"][1]:+.2f}, {out[f"H7|{v}"][2]:+.2f}] n={len(x)}')
    json.dump({k: v for k, v in out.items()}, open(RF / ('fpc_scores_block0.json' if BLOCK0 else 'fpc_scores.json'), 'w'), indent=1, default=float)
    import pickle; pickle.dump({str(k): {d: [x.tolist() for x in v] for d, v in per.items()} for k, per in S.items()}, open(RF / ('fpc_per_block0.pkl' if BLOCK0 else 'fpc_per.pkl'), 'wb'))
