import json, numpy as np, math
from scipy.optimize import minimize
import syn_analysis as SA
rows = [r for r in SA.per_image() if r['exp'] == 'res' and r['v'] in ('gem38lo', 'gem38', 'gem38uh', 'gemini25')]
print('recall by symbol size (px): lo / default / ultra-high / Gemini 2.5')
for m in SA.SY.RES_SIZES:
    cells = [np.mean([r['recall'] for r in rows if r['v'] == v and r['m'] == m]) for v in ('gem38lo', 'gem38', 'gem38uh', 'gemini25')]
    Ss = [np.mean([r['S'] for r in rows if r['v'] == v and r['m'] == m]) for v in ('gem38lo', 'gem38', 'gem38uh')]
    print(f'  {m:3d}px  S {Ss[0]:.2f}/{Ss[1]:.2f}/{Ss[2]:.2f}   recall ' + ' / '.join(f'{c:.2f}' for c in cells))
# logistic in log S (pooled over the three Gemini 3.8 budgets) with and without budget offsets; binomial over targets
g = [r for r in rows if r['v'] != 'gemini25']
x = np.log([r['S'] for r in g]); k = np.array([round(r['recall'] * r['n']) for r in g]); n = np.array([r['n'] for r in g])
B = np.array([[r['v'] == 'gem38lo', r['v'] == 'gem38uh'] for r in g], float)
def nll(th, use_b):
    z = th[0] + th[1] * x + (B @ th[2:4] if use_b else 0); p = 1 / (1 + np.exp(-z)); p = np.clip(p, 1e-9, 1 - 1e-9)
    return -np.sum(k * np.log(p) + (n - k) * np.log(1 - p))
r0 = minimize(nll, [0, 2], args=(False,)); r1 = minimize(nll, [0, 2, 0, 0], args=(True,))
aic0 = 2 * 2 + 2 * r0.fun; aic1 = 2 * 4 + 2 * r1.fun
print(f'pooled logistic in log S: S50={math.exp(-r0.x[0] / r0.x[1]):.2f}, slope {r0.x[1]:.2f}; AIC {aic0:.1f} vs with budget offsets {aic1:.1f} (offsets lo {r1.x[2]:+.2f}, uh {r1.x[3]:+.2f})')
# T1-H3: same token grid (16x16): Gemini 3.8 low vs Gemini 2.5, paired by image
lo = {r['id']: r['recall'] for r in rows if r['v'] == 'gem38lo'}; g25 = {r['id']: r['recall'] for r in rows if r['v'] == 'gemini25'}
ids = sorted(set(lo) & set(g25)); d = np.array([lo[i] - g25[i] for i in ids]); rng = np.random.default_rng(0)
bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
print(f'T1-H3 Gemini 3.8 low - Gemini 2.5 (same S): {d.mean():+.2f} [{np.percentile(bs, 2.5):+.2f}, {np.percentile(bs, 97.5):+.2f}] n={len(d)}')
json.dump({'S50_pooled': math.exp(-r0.x[0] / r0.x[1]), 'aic': [aic0, aic1], 'offsets': list(r1.x[2:]), 'h3': [d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)]}, open('results_x/t1_syn.json', 'w'))
