#!/usr/bin/env python3
"""T2 on FloorPlanCAD (prereg v4.1): per model, OLS of per-item whole-image F1 on log S, on the quantized-observer
ceiling, and on both plus log n; incremental R^2 and nested F tests. FPC-300 test items and FPC-sheets (all areas).
Uses per-item scores saved by fpc_score.py (all-block scoring)."""
import json, math, pickle, ast
import numpy as np
from scipy import stats
import ceilings as K
import fpc_score as F

PER = pickle.load(open(F.RF / 'fpc_per.pkl', 'rb'))
GV = {'gpt': 'gpt54', 'claude': 'sonnet46', 'gemini': 'gemini25'}
def ols(X, y):
    X = np.column_stack([np.ones(len(y))] + X); b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
    return 1 - r @ r / ((y - y.mean()) @ (y - y.mean())), r @ r, X.shape[1]
def nested(rss_r, k_r, rss_f, k_f, n):
    Fv = ((rss_r - rss_f) / (k_f - k_r)) / (rss_f / (n - k_f)); return Fv, 1 - stats.f.cdf(Fv, k_f - k_r, n - k_f)
out = {}; ceil_cache = {}
for key, per in PER.items():
    b, c, v = ast.literal_eval(key)
    if c != 'A1' or v == 'tm': continue
    ids = [d for d in (F.TEST if b == 'bench' else F.SHEETS) if d in per and not np.isnan(np.mean([r[0] for r in per[d]]))]
    if len(ids) < 20: continue
    gv = GV.get(v, v); y, S, ce, ln = [], [], [], []
    for d in ids:
        a = F.ANN[d]; m = math.sqrt(a['template_size'][0] * a['template_size'][1]); s_ = F.GEO[d][gv]['A1']; t = m / s_
        if (d, gv) not in ceil_cache: ceil_cache[(d, gv)] = K.ideal_observer(a['points'], t, m / 2)[0]
        y.append(np.mean([r[0] for r in per[d]])); S.append(s_); ce.append(ceil_cache[(d, gv)]); ln.append(math.log(max(1, a['count'])))
    y = np.array(y); ls = np.log(np.maximum(S, 0.05)); ce = np.array(ce); n = len(y)
    r1, rss1, k1 = ols([ls], y); r2, rss2, k2 = ols([ce], y); r12, rss12, k12 = ols([ls, ce], y); rf, _, _ = ols([ls, ce, np.array(ln)], y)
    F1, p1 = nested(rss1, k1, rss12, k12, n); F2, p2 = nested(rss2, k2, rss12, k12, n)
    out[f'{b}|{v}'] = {'n': n, 'r2_logS': r1, 'r2_ceil': r2, 'r2_both': r12, 'r2_full': rf, 'ceil_given_S': [r12 - r1, F1, p1], 'S_given_ceil': [r12 - r2, F2, p2]}
    print(f'{b:6s} {v:9s} n={n:3d} R2 logS {r1:.2f} ceil {r2:.2f} both {r12:.2f} | ceil|S {r12 - r1:+.2f} p={p1:.3f} | S|ceil {r12 - r2:+.2f} p={p2:.3f}')
json.dump(out, open('results_x/t2_nested_fpc.json', 'w'), indent=1)
