#!/usr/bin/env python3
"""T2 (prereg v4.1): does the quantized-observer ceiling explain per-drawing whole-image F1 beyond log S?
Per model: OLS of F1 on (i) log S, (ii) ceiling, (iii) log S + ceiling + log n; incremental R^2 and nested F test."""
import json, math
import numpy as np
from scipy import stats
import mech as M, ceilings as K
import analyze_test as A
import x_new_score as X

C1 = json.load(open('ceilings.json')); C2 = json.load(open('gen_ceilings.json'))
def ols(Xm, y):
    Xm = np.column_stack([np.ones(len(y))] + Xm); b, *_ = np.linalg.lstsq(Xm, y, rcond=None); r = y - Xm @ b
    return 1 - r @ r / ((y - y.mean()) @ (y - y.mean())), r @ r, Xm.shape[1]
def nested(rss_r, k_r, rss_f, k_f, n):
    F = ((rss_r - rss_f) / (k_f - k_r)) / (rss_f / (n - k_f)); return F, 1 - stats.f.cdf(F, k_f - k_r, n - k_f)
rows = []
G1 = {'gpt54': ('gpt', 'px'), 'sonnet46': ('claude', 'px_claude'), 'gemini25': ('gemini', 'n1000'), 'gemma4': ('gemma4', 'n1000'), 'qwen3vl': ('qwen3vl', 'n1000')}
out = {}
for name, (k, dec) in G1.items():
    per = A.mean_per(A.score_whole(k, dec)); ds = [d for d in K.TEST if d in per]
    S = np.array([C1[f'{d}|A1|{k}']['sym_tokens'] for d in ds]); ce = np.array([C1[f'{d}|A1|{k}']['ceiling_f1'] for d in ds])
    rows.append((name, ds, per, S, ce))
for name in ('gpt56', 'sonnet55', 'gem38'):
    per = A.mean_per(X.at(A.score_whole, name, 'n1000' if name == 'gem38' else 'px')); ds = [d for d in K.TEST if d in per]
    S = np.array([C2[f'{d}|{name}']['sym_tokens'] for d in ds]); ce = np.array([C2[f'{d}|{name}']['ceiling_f1'] for d in ds])
    rows.append((name, ds, per, S, ce))
print('model     n   R2(logS) R2(ceil) R2(both+logn)  dR2 ceil|logS  F p   | dR2 logS|ceil  F p')
for name, ds, per, S, ce in rows:
    y = np.array([per[d][0] for d in ds]); ls = np.log(np.maximum(S, 0.05)); ln = np.log([A.ANN[d]['count'] for d in ds]); n = len(y)
    r1, rss1, k1 = ols([ls], y); r2, rss2, k2 = ols([ce], y); r12, rss12, k12 = ols([ls, ce], y); rf, rssf, kf = ols([ls, ce, ln], y)
    F1, p1 = nested(rss1, k1, rss12, k12, n); F2, p2 = nested(rss2, k2, rss12, k12, n)
    print(f'{name:9s} {n:3d}  {r1:.2f}     {r2:.2f}     {rf:.2f}          {r12 - r1:+.2f}  {F1:.1f} {p1:.3f}    | {r12 - r2:+.2f}  {F2:.1f} {p2:.3f}')
    out[name] = {'n': n, 'r2_logS': r1, 'r2_ceil': r2, 'r2_both': r12, 'r2_full': rf, 'ceil_given_S': [r12 - r1, F1, p1], 'S_given_ceil': [r12 - r2, F2, p2]}
json.dump(out, open('results_x/t2_nested_redp.json', 'w'), indent=1)
