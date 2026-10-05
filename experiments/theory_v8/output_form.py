"""Output-form check (false outputs and true outputs per region; cell-kept decomposition F1) on the existing syn_assume calls. No new calls."""
import json, sys, numpy as np
sys.path.insert(0, '.')
import syn_assume as A
items = {it['id']: it for it in json.load(open(A.ROOT / 'items.json'))}
SIDE = A.SIDE
def load(v):
    recs = {}
    for l in open(A.OUT / f'{v}.jsonl'):
        r = json.loads(l)
        if r['status'] in ('success', 'parse_error'): recs[(r['pid'], r['kind'])] = r
    return recs
def all_points(rec):  # every returned point mapped to native, no core filter
    L, coord, fixed = A.VEND[rec['vendor']]
    import mech as M
    pts = [p for p in (M.point_of(q) for q in rec.get('shapes', [])) if p is not None] if rec['status'] == 'success' else []
    if not pts and rec['status'] == 'parse_error': pts = M.lenient_points(rec.get('raw'))
    cw, ch = rec['canvas']; out = []
    for x, y in pts:
        if coord == 'n1000': x, y = x / 1000 * cw, y / 1000 * ch
        out.append((rec['origin'][0] + x / rec['scale'], rec['origin'][1] + y / rec['scale']))
    return np.array(out, float).reshape(-1, 2)
def inside(P, U): return (P[:, 0] >= U[0]) & (P[:, 0] < U[2]) & (P[:, 1] >= U[1]) & (P[:, 1] < U[3])
def classify(P, G, tau):
    if len(P) == 0: return np.zeros(0, bool), np.zeros(0, int)
    d = np.hypot(P[:, None, 0] - G[None, :, 0], P[:, None, 1] - G[None, :, 1]); j = d.argmin(1)
    return d[np.arange(len(P)), j] <= tau, j
def counts(P, G, tau):
    """true outputs, false outputs, matched targets (TP), duplicates."""
    tr, j = classify(P, G, tau); T = int(tr.sum()); F = int((~tr).sum()); TP = len(set(j[tr].tolist())); return T, F, TP, T - TP
def boot(xs, rng, B=4000):
    g = {}
    for pid, d in xs: g.setdefault(pid, []).append(d)
    keys = list(g); arr = np.array([d for _, d in xs], float)
    bs = [np.mean([d for kk in (keys[i] for i in rng.integers(0, len(keys), len(keys))) for d in g[kk]]) for _ in range(B)]
    return float(arr.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))
res = {}; rng = np.random.default_rng(5)
for v in ['gpt56', 'gpt54', 'sonnet55', 'qwen3vl', 'gem38', 'gemma4', 'opus55', 'gpt61']:
    recs = load(v); rel = {k: {'T': [], 'F': []} for k in ['M', 'R', 'I', 'C']}
    dec = {'dT': [], 'dF': [], 'dDup': [], 'dTP': [], 'dF1': []}; agg = {'w': np.zeros(4), 'd': np.zeros(4), 'n': 0}
    for pid, it in items.items():
        tau = it['m'] / 2; G = np.array(it['points'], float)
        Pw = all_points(recs[(pid, 'P')]) if (pid, 'P') in recs else None
        if Pw is None: continue
        for c in A.calls_for(it):
            if c['kind'][0] != 'M': continue
            ij = c['kind'][1:]; U = c['core']
            have = {k: (pid, k + ij) in recs for k in 'MRI'}
            if not all(have.values()): continue
            Pm, Pr, Pi = (all_points(recs[(pid, k + ij)]) for k in 'MRI')
            cw = counts(Pw[inside(Pw, U)], G, tau); cm = counts(Pm[inside(Pm, U)], G, tau)
            cr = counts(Pr[inside(Pr, U)], G, tau); ci = counts(Pi[inside(Pi, U)], G, tau)
            for k, (a, b) in {'M': (cm, cw), 'R': (cr, cm), 'I': (cr, ci), 'C': (cr, cw)}.items():
                rel[k]['T'].append((pid, a[0] - b[0])); rel[k]['F'].append((pid, b[1] - a[1]))  # predicted >= 0 for both
        # cell-kept decomposition: 3x3 partition of the drawing, each tile keeps its points in its own cell
        tiles = [c for c in A.calls_for(it) if c['kind'][0] == 'T']
        if not all((pid, c['kind']) in recs for c in tiles): continue
        cell = SIDE / 3; Pd = []
        for c in tiles:
            i, j = int(c['kind'][1]), int(c['kind'][2]); Q = [i * cell, j * cell, (i + 1) * cell, (j + 1) * cell]
            P = all_points(recs[(pid, c['kind'])]); Pd.append(P[inside(P, Q)])
        Pd = np.vstack(Pd) if Pd else np.zeros((0, 2))
        cw = counts(Pw[inside(Pw, [0, 0, SIDE, SIDE])], G, tau); cd = counts(Pd, G, tau); n = len(G)
        f1 = lambda c: 2 * c[2] / (n + c[0] + c[1])
        dec['dT'].append((pid, (cd[0] - cw[0]) / n)); dec['dF'].append((pid, (cw[1] - cd[1]) / n)); dec['dDup'].append((pid, (cd[3] - cw[3]) / n))
        dec['dTP'].append((pid, (cd[2] - cw[2]) / n)); dec['dF1'].append((pid, f1(cd) - f1(cw)))
        agg['w'] += cw; agg['d'] += cd; agg['n'] += n
    out = {}
    for k in rel:
        for q in 'TF':
            m, lo, hi = boot(rel[k][q], rng); out[f'{k}:{q}'] = [m, lo, hi, len(rel[k][q])]
    for k in dec:
        m, lo, hi = boot(dec[k], rng); out[f'dec:{k}'] = [m, lo, hi, len(dec[k])]
    out['agg'] = {'whole': agg['w'].tolist(), 'dec': agg['d'].tolist(), 'n': agg['n']}
    res[v] = out
    print(v); [print(f'   {k:12s} {x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}] n={x[3]}') for k, x in out.items() if k != 'agg']; print('   agg (T,F,TP,Dup) whole', agg['w'], 'dec', agg['d'], 'n', agg['n'])
json.dump(res, open('theory_v8/output_form.json', 'w'), indent=1)
