#!/usr/bin/env python3
"""Experiment G: scoring and pre-registered tests (claude/gen_experiment_prereg.md)."""
import json, glob
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import mech as M, ceilings as K
import analyze_test as A

GEN = Path('results_gen'); G1 = A.R
GC = json.load(open('gen_ceilings.json'))
T = K.TEST; SHEETS = [d for d in T if A.SUB[d] == 'sheet']; VIEWS = [d for d in T if A.SUB[d] == 'view']
rng = np.random.default_rng(1)


def at(root, fn, *a):
    A.R = root
    try:
        return fn(*a)
    finally:
        A.R = G1


def native(vendor):
    per = {}
    for f in glob.glob(str(GEN / f'A1nat_{vendor}_run*.jsonl')):
        for l in open(f):
            r = json.loads(l)
            if r['status'] not in ('success', 'parse_error'):
                continue
            s = r['scale']
            pts = [(q['cx'] / s, q['cy'] / s) for q in (r['shapes'] if r['status'] == 'success' else [])
                   if isinstance(q, dict) and A.num(q.get('cx')) and A.num(q.get('cy'))]
            per.setdefault(r['did'], []).append(A.metrics(r['did'], pts))
    return per


S = {}
for v in ('sonnet55', 'opus55', 'gpt55', 'gpt56', 'gem38', 'gem38uh'):
    S[('A1', v)] = at(GEN, A.score_whole, v, 'px')
S[('A1nat', 'gpt56orig')] = native('gpt56orig')
for v in ('sonnet55', 'gpt56', 'gem38'):
    S[('D', v)] = at(GEN, A.score_tiles, 'D', v)
    S[('H', v)] = at(GEN, A.score_H, v)
# generation 1 (claude A1 decoded in its documented resized frame; gemini/gemma/qwen A2 = 0-1000)
S[('A1', 'gpt54')] = A.score_whole('gpt', 'px'); S[('A1', 'sonnet46')] = A.score_whole('claude', 'px_claude')
S[('A1', 'gemini25')] = A.score_whole('gemini', 'px'); S[('A2', 'gemini25')] = A.score_whole('gemini', 'n1000')
for v, k in (('gpt54', 'gpt'), ('sonnet46', 'claude'), ('gemini25', 'gemini'), ('gemma4', 'gemma4'), ('qwen3vl', 'qwen3vl')):
    S[('D', v)] = A.score_tiles('D', k); S[('H', v)] = A.score_H(k)

def f1(per, ds): return A.summary(per, ds)[0][0] if A.summary(per, ds)[0] is not None else float('nan')

print(f'{"cond":6s} {"model":10s} | all   sheets views')
for (c, v), per in S.items():
    print(f'{c:6s} {v:10s} | {f1(per,T):.2f}  {f1(per,SHEETS):.2f}  {f1(per,VIEWS):.2f}   n={len(A.mean_per(per))}')
tm_dev, tm_loo, thr = A.tm_scores()
print(f'TM dev-thr {f1(tm_dev,T):.2f} ({f1(tm_dev,SHEETS):.2f}/{f1(tm_dev,VIEWS):.2f})  TM LOO {f1(tm_loo,T):.2f}')

print('\n=== G-H1 ceilings (strong, quantized observer) vs observed')
cmap = {('A1', 'sonnet55'): 'sonnet55', ('A1', 'opus55'): 'opus55', ('A1', 'gpt55'): 'gpt55', ('A1', 'gpt56'): 'gpt56',
        ('A1', 'gem38'): 'gem38', ('A1', 'gem38uh'): 'gem38uh', ('A1nat', 'gpt56orig'): 'gpt56orig',
        ('D', 'sonnet55'): 'D_sonnet55', ('D', 'gpt56'): 'D_gpt56', ('D', 'gem38'): 'D_gem38'}
viol = 0; tot = 0
for key, ck in cmap.items():
    mp = A.mean_per(S[key])
    for lab, ds in (('sheets', SHEETS), ('views', VIEWS)):
        ds = [d for d in ds if d in mp]
        if not ds: continue
        o = np.mean([mp[d][0] for d in ds]); c = np.mean([GC[f'{d}|{ck}']['ceiling_f1'] for d in ds]); tot += 1
        flag = o > c + 0.05; viol += flag
        print(f'  {key[0]:6s} {key[1]:10s} {lab:6s} obs {o:.2f} ceiling {c:.2f}{"  <-- above" if flag else ""}')
print(f'  violations {viol}/{tot}')

print('\n=== G-H2 same geometry, new generation (A1, paired over 34 drawings)')
for v in ('gpt55', 'gpt56'):
    d = A.boot_diff(S[('A1', v)], S[('A1', 'gpt54')], T); print(f'  {v} - gpt54: {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]')
d = A.boot_diff(S[('A1', 'opus55')], S[('A1', 'sonnet55')], T); print(f'  opus55 - sonnet55 (same geometry): {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]')

print('\n=== G-H3 larger budget, sheets (paired)')
for a_, b_ in ((('A1', 'gem38'), ('A1', 'gemini25')), (('A1', 'gem38'), ('A2', 'gemini25')), (('A1', 'gem38uh'), ('A1', 'gem38')),
               (('A1', 'sonnet55'), ('A1', 'sonnet46'))):
    d = A.boot_diff(S[a_], S[b_], SHEETS); print(f'  {a_[1]}({a_[0]}) - {b_[1]}({b_[0]}): {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]')

print('\n=== G-H4 tiling still needed: D - A1 on sheets')
for v in ('sonnet55', 'gpt56', 'gem38'):
    d = A.boot_diff(S[('D', v)], S[('A1', v)], SHEETS); print(f'  {v}: {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]')

print('\n=== G-H5 capacity: GPT-5.6 native (original) vs auto whole drawing vs tiles, sheets')
d1 = A.boot_diff(S[('A1nat', 'gpt56orig')], S[('A1', 'gpt56')], SHEETS); d2 = A.boot_diff(S[('D', 'gpt56')], S[('A1nat', 'gpt56orig')], SHEETS)
print(f'  native - auto: {d1[0]:+.2f} [{d1[1]:+.2f}, {d1[2]:+.2f}]   tiles - native: {d2[0]:+.2f} [{d2[1]:+.2f}, {d2[2]:+.2f}]  (n={d1[4]})')
mp = A.mean_per(S[('A1nat', 'gpt56orig')])
print('  native recall/precision on sheets: %.2f / %.2f' % tuple(np.mean([mp[d][k] for d in SHEETS if d in mp]) for k in (1, 2)))

print('\n=== G-H6 role: propose-and-verify vs template matching; spread across 8 models')
hs = {}
for v in ('gpt54', 'sonnet46', 'gemini25', 'gemma4', 'qwen3vl', 'sonnet55', 'gpt56', 'gem38'):
    hs[v] = f1(S[('H', v)], T)
    if v in ('sonnet55', 'gpt56', 'gem38'):
        d = A.boot_diff(S[('H', v)], tm_dev, T); print(f'  H {v} - TM(dev thr): {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]')
print('  H by model:', {k: round(v, 2) for k, v in hs.items()}, ' range %.2f' % (max(hs.values()) - min(hs.values())))
ds_ = {v: f1(S[('D', v)], T) for v in ('gpt54', 'sonnet46', 'gemini25', 'gemma4', 'qwen3vl', 'sonnet55', 'gpt56', 'gem38')}
print('  D by model:', {k: round(v, 2) for k, v in ds_.items()}, ' range %.2f' % (max(ds_.values()) - min(ds_.values())))

print('\n=== G-H7 dose-response (A1 observed vs per-drawing ceiling)')
for v in ('sonnet55', 'opus55', 'gpt55', 'gpt56', 'gem38', 'gem38uh'):
    mp = A.mean_per(S[('A1', v)]); ds = [d for d in T if d in mp]
    x = [GC[f'{d}|{v}']['ceiling_f1'] for d in ds]; y = [mp[d][0] for d in ds]
    bs = [spearmanr(np.array(x)[i], np.array(y)[i])[0] for i in (rng.integers(0, len(ds), len(ds)) for _ in range(2000))]
    print(f'  {v:9s} rho {spearmanr(x, y)[0]:+.2f} [{np.nanpercentile(bs,2.5):+.2f}, {np.nanpercentile(bs,97.5):+.2f}]')
