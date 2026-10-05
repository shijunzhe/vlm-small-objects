#!/usr/bin/env python3
"""Fit and test the S-L law (plan claude/law_analysis_plan_v5.md).
Per-target detection probability in one call: logit p = a_v + delta_d + b_v log S - c_v log n_call - e_v log(A_call/m^2).
Model parameters from synthetic stimuli only; one domain offset per domain from tuning items only; then frozen.
Writes results_law/law_fit.json."""
import json, math, sys
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr, pearsonr

R = json.load(open('results_law/law_data.json'))
SYN = [r for r in R if r['ds'] == 'syn']
MODELS = ['gpt54', 'sonnet46', 'gemini25', 'gemma4', 'qwen3vl', 'gpt56', 'sonnet55', 'gem38']
PARAM_OF = {m: m for m in MODELS} | {'gem38lo': 'gem38', 'gem38uh': 'gem38'}
PRIOR_SD = 5.0
import os
MONO = os.environ.get('MONO', '1') == '1'
EXPS = tuple(os.environ.get('EXPS', 'res,cap,area,clut,up').split(','))
interaction = False
CLUT = {}   # synthetic clutter density (segments per 1024^2): 40 by default, X1d levels for 'clut' items
for r in SYN: CLUT[r['item']] = float(r['level']) if r['exp'] == 'clut' else 40.0
rng = np.random.default_rng(11)


def design(S, n, A, form, ink=None):
    cols = [np.log(np.maximum(S, 1e-3))]
    if 'n' in form: cols.append(-np.log(np.maximum(n, 1)))
    if 'A' in form: cols.append(-np.log(np.maximum(A, 1)))
    if 'I' in form: cols.append(-np.log1p(ink))
    if 'X' in form: cols.append(np.log(np.maximum(S, 1e-3)) * np.log(np.maximum(n, 1)))
    return np.column_stack(cols)


def fit_binom(X, k, n):
    """Penalized binomial logistic fit: intercept free, slopes N(0, PRIOR_SD^2)."""
    def obj(p):
        z = p[0] + X @ p[1:]; ll = k * (-np.logaddexp(0, -z)) + (n - k) * (-np.logaddexp(0, z))
        return -ll.sum() + (p[1:] ** 2).sum() / (2 * PRIOR_SD ** 2)
    p0 = np.zeros(X.shape[1] + 1)
    # monotone law: detection never falls with S and never rises with content (slopes >= 0); interaction unconstrained
    bounds = [(None, None)] + [(0, None)] * (X.shape[1]) if MONO else None
    if MONO and X.shape[1] and 'interaction' in globals() and globals()['interaction']: bounds[-1] = (None, None)
    res = minimize(obj, p0, method='L-BFGS-B', bounds=bounds)
    z = res.x[0] + X @ res.x[1:]; ll = (k * (-np.logaddexp(0, -z)) + (n - k) * (-np.logaddexp(0, z))).sum()
    return res.x, ll


def syn_rows(v, exps):
    rs = [r for r in SYN if r['model'] == v and r['exp'] in exps]
    S = np.array([r['calls'][0][0] for r in rs]); n = np.array([r['calls'][0][1] for r in rs]); A = np.array([r['calls'][0][2] for r in rs])
    k = np.array([round(r['obs'] * r['n']) for r in rs]); N = np.array([r['n'] for r in rs]); ink = np.array([CLUT[r['item']] for r in rs])
    return S, n, A, ink, k, N


OUT = {'forms': {}, 'params': {}, 'sep': {}}
# ---- 1. L form selection on synthetic only (res, cap, area, clut) ----
FORMS = ['n', 'A', 'nA', 'nAI']
tot = defaultdict(float)
for v in MODELS:
    S, n, A, ink, k, N = syn_rows(v, EXPS)
    for f in FORMS:
        p, ll = fit_binom(design(S, n, A, f, ink), k, N); aic = 2 * (len(p)) - 2 * ll
        OUT['forms'].setdefault(v, {})[f] = aic; tot[f] += aic
OUT['forms']['total'] = dict(tot)
FORM = min(FORMS, key=lambda f: tot[f])
print('AIC by L form (sum over models):', {f: round(tot[f], 1) for f in FORMS}, '-> chosen', FORM)

# ---- 2. separability: add log S x log n ----
for v in MODELS:
    S, n, A, ink, k, N = syn_rows(v, EXPS)
    _, l0 = fit_binom(design(S, n, A, FORM, ink), k, N)
    globals()['interaction'] = True; _, l1 = fit_binom(design(S, n, A, FORM + 'X', ink), k, N); globals()['interaction'] = False
    OUT['sep'][v] = 2 - 2 * (l1 - l0)   # delta AIC of adding the interaction (negative favours interaction)
print('separability dAIC (interaction):', {v: round(x, 1) for v, x in OUT['sep'].items()})

# ---- 3. final per-model parameters on synthetic (the chosen form; ink term only if chosen) ----
for v in MODELS:
    S, n, A, ink, k, N = syn_rows(v, EXPS)
    p, _ = fit_binom(design(S, n, A, FORM, ink), k, N); OUT['params'][v] = p.tolist()
print('params:', {v: [round(x, 2) for x in p] for v, p in OUT['params'].items()})


def p_call(v, calls, delta, form=FORM, params=None):
    pr = np.array((params or OUT['params'])[PARAM_OF[v]])
    c = np.array(calls, float); S, n, A = c[:, 0], c[:, 1], c[:, 2]
    X = design(S, n, A, form.replace('I', ''), np.full(len(S), 40.0))
    if 'I' in form: X = np.column_stack([X, np.full(len(S), -np.log1p(40.0))])
    z = pr[0] + delta + X @ pr[1:]
    return 1 / (1 + np.exp(-z)), n


def item_pred(r, delta, **kw):
    p, n = p_call(r['model'], r['calls'], delta, **kw)
    return float((p * n).sum() / max(n.sum(), 1)) if n.sum() > 0 else float(p.mean())


# ---- 4. domain offsets from tuning items ----
DOM = {'redp': 'redp', 'fpc_bench': 'fpc', 'fpc_sheets': 'fpc', 'fsc': 'fsc'}
REAL = [r for r in R if r['ds'] != 'syn' and r['model'] in PARAM_OF]
OUT['delta'] = {}
for d in ('redp', 'fpc', 'fsc'):
    tune = [r for r in REAL if DOM[r['ds']] == d and r['split'] == 'tune']
    k = np.array([r['obs'] * r['n'] for r in tune]); N = np.array([r['n'] for r in tune])
    def nll(x):
        q = np.clip([item_pred(r, x[0]) for r in tune], 1e-4, 1 - 1e-4)
        return -(k * np.log(q) + (N - k) * np.log(1 - q)).sum()
    OUT['delta'][d] = float(minimize(nll, [0.0], method='Nelder-Mead').x[0]); print(f'delta {d}: {OUT["delta"][d]:+.2f} (n tune items {len(tune)})')

# ---- 5. test predictions (frozen) ----
TEST = [r for r in REAL if r['split'] == 'test']
for r in TEST:
    r['pred'] = item_pred(r, OUT['delta'][DOM[r['ds']]])
    r['pred0'] = item_pred(r, 0.0)
    # baselines: S-only law and n-only law, refit on synthetic with the same procedure
OUT['baselines'] = {}
for bname, form in (('S_only', ''), ('n_only', 'n')):
    params = {}
    for v in MODELS:
        S, n, A, ink, k, N = syn_rows(v, EXPS)
        params[v] = fit_binom(design(S, n, A, form, ink), k, N)[0].tolist()
    deltas = {}
    for d in ('redp', 'fpc', 'fsc'):
        tune = [r for r in REAL if DOM[r['ds']] == d and r['split'] == 'tune']
        k = np.array([r['obs'] * r['n'] for r in tune]); N = np.array([r['n'] for r in tune])
        f = lambda x: -(k * np.log(np.clip([item_pred(r, x[0], form=form, params=params) for r in tune], 1e-4, 1 - 1e-4)) + (N - k) * np.log(1 - np.clip([item_pred(r, x[0], form=form, params=params) for r in tune], 1e-4, 1 - 1e-4))).sum()
        deltas[d] = float(minimize(f, [0.0], method='Nelder-Mead').x[0])
    for r in TEST: r['pred_' + bname] = item_pred(r, deltas[DOM[r['ds']]], form=form, params=params)
    OUT['baselines'][bname] = {'params': params, 'delta': deltas}


def m1(rs, key='pred'):
    o = np.array([r['obs'] for r in rs]); p = np.array([r[key] for r in rs])
    if len(rs) < 5 or o.std() == 0: return None
    return {'n': len(rs), 'spearman': float(spearmanr(p, o)[0]), 'pearson': float(pearsonr(p, o)[0]), 'mae': float(np.abs(p - o).mean()),
            'r2': float(1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum()), 'mean_obs': float(o.mean()), 'mean_pred': float(p.mean())}


OUT['M1'] = {}
groups = defaultdict(list)
for r in TEST: groups[(r['ds'], r['model'], r['config'])].append(r)
for g, rs in sorted(groups.items()):
    OUT['M1']['|'.join(g)] = {k: m1(rs, k) for k in ('pred', 'pred0', 'pred_S_only', 'pred_n_only')}
for ds in sorted({r['ds'] for r in TEST}):
    rs = [r for r in TEST if r['ds'] == ds]
    OUT['M1'][ds + '|ALL'] = {k: m1(rs, k) for k in ('pred', 'pred0', 'pred_S_only', 'pred_n_only')}
OUT['M1']['ALL'] = {k: m1(TEST, k) for k in ('pred', 'pred0', 'pred_S_only', 'pred_n_only')}

# ---- M2 configuration effects and M3 choice ----
alt = {'redp': ('A1', 'D'), 'fpc_bench': ('A1', 'D'), 'fpc_sheets': ('A1', 'D'), 'fsc': ('W', 'U', 'T')}
idx = defaultdict(dict)
for r in TEST: idx[(r['ds'], r['model'], r['item'])][r['config']] = r
OUT['M2'] = {}; OUT['M3'] = {}
for ds, cfgs in alt.items():
    a, b = cfgs[0], cfgs[-1]
    pairs = [(c[a], c[b]) for (d, v, i), c in idx.items() if d == ds and a in c and b in c]
    if pairs:
        po = np.array([y['obs'] - x['obs'] for x, y in pairs]); pp = np.array([y['pred'] - x['pred'] for x, y in pairs])
        big = np.abs(pp) > 0.1
        boot = []
        for _ in range(2000):
            j = rng.integers(0, len(po), len(po)); boot.append(spearmanr(pp[j], po[j])[0])
        OUT['M2'][ds] = {'n': len(pairs), 'spearman': float(spearmanr(pp, po)[0]), 'ci': [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                         'sign_acc_big': float(np.mean(np.sign(pp[big]) == np.sign(po[big]))) if big.any() else None, 'n_big': int(big.sum()),
                         'mean_obs': float(po.mean()), 'mean_pred': float(pp.mean())}
        # by model
        for v in sorted({x['model'] for x, _ in pairs}):
            sel = np.array([x['model'] == v for x, _ in pairs])
            if sel.sum() > 5:
                OUT['M2'][f'{ds}|{v}'] = {'n': int(sel.sum()), 'spearman': float(spearmanr(pp[sel], po[sel])[0]), 'mean_obs': float(po[sel].mean()), 'mean_pred': float(pp[sel].mean())}
    # M3: choose among available configs
    rows = [(c) for (d, v, i), c in idx.items() if d == ds and all(x in c for x in cfgs)]
    if rows:
        law = [max(cfgs, key=lambda x: c[x]['pred']) for c in rows]; best = [max(cfgs, key=lambda x: c[x]['obs']) for c in rows]
        obs = lambda c, x: c[x]['obs']
        reg = lambda choice: float(np.mean([max(obs(c, x) for x in cfgs) - obs(c, ch) for c, ch in zip(rows, choice)]))
        OUT['M3'][ds] = {'n': len(rows), 'regret_law': reg(law), **{f'regret_always_{x}': reg([x] * len(rows)) for x in cfgs},
                         'agree': float(np.mean([l == b_ for l, b_ in zip(law, best)]))}
json.dump(OUT, open(f'results_law/law_fit{"" if MONO else "_unconstrained"}.json', 'w'), indent=1)
json.dump([{k: r[k] for k in r if k != 'calls'} for r in TEST], open('results_law/law_test_preds.json', 'w'))
print('M1 ALL', OUT['M1']['ALL'])
for k, v in OUT['M1'].items():
    if k.endswith('ALL'): print(k, {kk: (round(vv['spearman'], 2), round(vv['mae'], 2), round(vv['r2'], 2)) if vv else None for kk, vv in v.items()})
print('M2', json.dumps(OUT['M2'], indent=0)[:3000])
print('M3', OUT['M3'])
