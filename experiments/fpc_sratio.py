#!/usr/bin/env python3
"""Post hoc (FPC-300): does the per-item tile effect D - A1 track the change in S that tiling causes?
The D recipe rescales so the reference's shorter side spans 44 px; for large FloorPlanCAD symbols this shrinks the
image, so tiles can lower S. Spearman of (D - A1) with log(S_D / S_A1), and mean D - A1 where tiles raise or lower S."""
import json, pickle
import numpy as np
from scipy.stats import spearmanr
G = json.load(open('../data/fpc/geometry.json')); T = json.load(open('../data/fpc/bench/manifest.json'))['test']
P = pickle.load(open('results_fpc/fpc_per.pkl', 'rb'))
GV = {'gpt': 'gpt54', 'claude': 'sonnet46', 'gemini': 'gemini25'}
rng = np.random.default_rng(3); out = {}
for v in ['gpt', 'claude', 'qwen3vl', 'gpt56', 'sonnet55', 'gem38', 'gemini', 'gemma4']:
    a = P[str(('bench', 'A1', v))]; d = P[str(('bench', 'D', v))]; g = GV.get(v, v); xs, ys = [], []
    for i in T:
        if i in a and i in d:
            fa = np.nanmean([r[0] for r in a[i]]) if a[i] else np.nan; fd = np.nanmean([r[0] for r in d[i]]) if d[i] else np.nan
            if np.isnan(fa) or np.isnan(fd): continue
            xs.append(np.log(G[i][g]['D'] / G[i][g]['A1'])); ys.append(fd - fa)
    xs, ys = np.array(xs), np.array(ys); r = spearmanr(xs, ys)
    def mci(z):
        if len(z) < 5: return [float(np.mean(z)) if len(z) else None, None, None, len(z)]
        bs = [z[rng.integers(0, len(z), len(z))].mean() for _ in range(4000)]
        return [float(z.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(z)]
    out[v] = {'rho': [float(r[0]), float(r[1]), len(xs)], 'up': mci(ys[xs > 0.05]), 'down': mci(ys[xs < -0.05]), 'frac_down': float(np.mean(xs < -0.05))}
    print(v, out[v])
json.dump(out, open('results_fpc/fpc_sratio.json', 'w'), indent=1)
