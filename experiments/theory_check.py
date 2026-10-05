#!/usr/bin/env python3
"""Every ordering implied by Assumption 1 (R, M, I) that the existing data contain, on recall.
Each comparison is 'better - worse' as predicted by the theory (prediction: >= 0). Paired bootstrap (4000) over items,
clustered by source drawing for FPC-sheets. A comparison is a violation if the upper end of its 95% interval is below 0.
Writes results_law/theory_check.json."""
import json, pickle, ast, math
from collections import defaultdict
import numpy as np

rng = np.random.default_rng(2026)
OUT = []


def boot(x, clusters=None):
    x = np.asarray(x, float)
    if clusters is None:
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]
    else:
        g = defaultdict(list)
        for v, c in zip(x, clusters): g[c].append(v)
        keys = list(g); bs = []
        for _ in range(4000):
            sel = [keys[i] for i in rng.integers(0, len(keys), len(keys))]; bs.append(np.mean([v for k in sel for v in g[k]]))
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x)


def add(family, rule, label, est):
    m, lo, hi, n = est
    OUT.append({'family': family, 'rule': rule, 'label': label, 'diff': m, 'lo': lo, 'hi': hi, 'n': n,
                'verdict': 'violated' if hi < 0 else ('supported' if lo > 0 else 'consistent')})


# ---------- Theorem 1: tiles with S_tile >= S_whole versus the whole image (REDP-X40, FPC-300, FPC-sheets, FSC) ----------
D = json.load(open('results_law/law_data.json'))
idx = defaultdict(dict)
for r in D:
    if r['split'] == 'test' and r['ds'] in ('redp', 'fpc_bench', 'fpc_sheets', 'fsc'): idx[(r['ds'], r['model'], r['item'])][r['config']] = r
NAME = {'redp': 'REDP-X40', 'fpc_bench': 'FPC-300', 'fpc_sheets': 'FPC-sheets', 'fsc': 'FSC-147'}
for ds in ('redp', 'fpc_bench', 'fpc_sheets'):
    for v in sorted({k[1] for k in idx if k[0] == ds}):
        xs, cl, xs_low, cl_low = [], [], [], []
        for k, c in idx.items():
            if k[0] != ds or k[1] != v or 'A1' not in c or 'D' not in c: continue
            s0 = c['A1']['calls'][0][0]; smin = min(x[0] for x in c['D']['calls'])
            d = c['D']['obs'] - c['A1']['obs']; src = k[2].split('_')[0] if ds == 'fpc_sheets' else k[2]
            if smin >= s0: xs.append(d); cl.append(src)
            else: xs_low.append(d); cl_low.append(src)
        if len(xs) >= 5: add('Thm1', 'tiles (S_tile >= S_whole) - whole', f'{NAME[ds]} {v}', boot(xs, cl if ds == 'fpc_sheets' else None))
        if len(xs_low) >= 5:
            m = boot(xs_low, cl_low if ds == 'fpc_sheets' else None)
            OUT.append({'family': 'outside', 'rule': 'tiles (S_tile < S_whole) - whole (no prediction)', 'label': f'{NAME[ds]} {v}', 'diff': m[0], 'lo': m[1], 'hi': m[2], 'n': m[3], 'verdict': 'n/a'})
# FSC: 2x2 tiles at the upsampled scale versus the whole image at native scale (S_T >= S_W): Theorem 1
for v in ('gpt56', 'sonnet55', 'gem38'):
    xs = [c['T']['obs'] - c['W']['obs'] for k, c in idx.items() if k[0] == 'fsc' and k[1] == v and 'T' in c and 'W' in c
          and min(x[0] for x in c['T']['calls']) >= c['W']['calls'][0][0]]
    if len(xs) >= 5: add('Thm1', 'tiles (S_tile >= S_whole) - whole', f'FSC-147 {v}', boot(xs))

# ---------- Prop. 2(d): tiles of the upsampled image versus the whole upsampled image (M), FSC ----------
for v in ('gpt56', 'sonnet55', 'gem38'):
    xs = [c['T']['obs'] - c['U']['obs'] for k, c in idx.items() if k[0] == 'fsc' and k[1] == v and 'T' in c and 'U' in c]
    add('M', 'tiles of upsampled - whole upsampled', f'FSC-147 {v}', boot(xs))
# ---------- R: upsampled whole image versus native whole image, resolution-scaled (U refines W), FSC ----------
for v in ('gpt56', 'sonnet55'):
    xs = [c['U']['obs'] - c['W']['obs'] for k, c in idx.items() if k[0] == 'fsc' and k[1] == v and 'U' in c and 'W' in c
          and c['U']['calls'][0][0] > c['W']['calls'][0][0]]
    add('R', 'upsampled - native (same pixels, higher S)', f'FSC-147 {v}', boot(xs))

# ---------- R: synthetic information-fixed upsampling (up vs its res source) where S increases ----------
syn = {(r['model'], r['item']): r for r in D if r['ds'] == 'syn'}
items = json.load(open('syn/items.json')); src = {i['id']: i.get('source') for i in items if i['exp'] == 'up'}
for v in ('gpt54', 'sonnet46', 'qwen3vl', 'gpt56', 'sonnet55'):
    xs = [syn[(v, u)]['obs'] - syn[(v, s)]['obs'] for u, s in src.items() if (v, u) in syn and (v, s) in syn and syn[(v, u)]['calls'][0][0] > syn[(v, s)]['calls'][0][0] * 1.05]
    if len(xs) >= 5: add('R', 'upsampled - original (same pixels, higher S)', f'synthetic {v}', boot(xs))

# ---------- R: Gemini 3.8 token budget at fixed pixels (low < default < ultra-high) ----------
P = pickle.load(open('results_fpc/fpc_per.pkl', 'rb'))
def rec(b, c, v):
    per = P.get(str((b, c, v)), {})
    return {d: float(np.nanmean([x[1] for x in rs])) for d, rs in per.items() if rs and not np.isnan(np.nanmean([x[1] for x in rs]))}
for b, nm in (('bench', 'FPC-300'), ('sheets', 'FPC-sheets')):
    lo, de, uh = rec(b, 'A1', 'gem38lo'), rec(b, 'A1', 'gem38'), rec(b, 'A1', 'gem38uh')
    for (a, x, lab) in ((de, lo, 'default - low'), (uh, de, 'ultra-high - default')):
        ds = [d for d in a if d in x and (b == 'sheets' or not d.startswith('s'))]
        if b == 'bench':
            man = json.load(open('../data/fpc/bench/manifest.json')); ds = [d for d in ds if d in set(man['test'])]
        add('R', f'Gemini 3.8 budget {lab} (same pixels)', nm, boot([a[d] - x[d] for d in ds], [d.split('_')[0] for d in ds] if b == 'sheets' else [d.split('-')[0] for d in ds]))
syn_rows = json.load(open('syn/analysis_robust.json'))['rows']
import syn_analysis as SA
rows = [r for r in SA.per_image() if r['exp'] == 'res' and r['v'] in ('gem38lo', 'gem38', 'gem38uh')]
byid = defaultdict(dict)
for r in rows: byid[r['id']][r['v']] = r['recall']
for a, x, lab in (('gem38', 'gem38lo', 'default - low'), ('gem38uh', 'gem38', 'ultra-high - default')):
    xs = [c[a] - c[x] for c in byid.values() if a in c and x in c]
    add('R', f'Gemini 3.8 budget {lab} (same pixels)', 'synthetic', boot(xs))

# ---------- Prop. 2(a): packing tiles into one image (mosaic) versus single tiles (M), REDP-X40 sheets ----------
X2 = json.load(open('results_x/x_new_scores_robust.json'))
for v in ('gpt56', 'sonnet55'):
    for k in (2, 4, 8):
        d = X2[f'X2|{v}|k{k}']['dR']   # recall(k) - recall(k = 1)
        add('M', f'single tiles - {k} tiles per image', f'REDP-X40 sheets {v}', (-d[0], -d[2], -d[1], d[4]))

# ---------- Prop. 2(b): first block alone versus inside a larger sheet (R then M), FPC-sheets ----------
P0 = pickle.load(open('results_fpc/fpc_per_block0.pkl', 'rb'))
KEY = {'gpt': 'gpt54', 'claude': 'sonnet46', 'gemini': 'gemini25'}
for key, per in P0.items():
    b, c, v = ast.literal_eval(key)
    if b != 'sheets' or c != 'A1' or v == 'tm': continue
    r = {d: float(np.nanmean([x[1] for x in rs])) for d, rs in per.items() if rs and not np.isnan(np.nanmean([x[1] for x in rs]))}
    for kk in (2, 3):
        pairs = [(r[d], r[d[:-1] + str(kk)]) for d in r if d.endswith('_k1') and d[:-1] + str(kk) in r]
        add('R+M', f'first block alone - inside {kk}x{kk} sheet', f'FPC-sheets {KEY.get(v, v)}', boot([a - x for a, x in pairs]))

# ---------- Prop. 2(c): zoom tools (crop-only >= overview call; enlarged >= crop-only; native >= enlarged), REDP-X40 ----------
import analyze_test as A, ceilings as K, x_new_score as X, x9_score as X9
T = K.TEST; SH = [d for d in T if A.SUB[d] == 'sheet']
for v, ar in (('gpt56', 'gpt56m'), ('sonnet55', 'sonnet55t')):
    S = {'A1R': X.at(A.score_whole, ar, 'px'), 'OVC': X9.agent(v, ['ovcrop_run0']), 'OVZ': X9.agent(v, ['ovzoom_run0', 'ovzoom_run1']), 'AG': X9.agent(v, ['run0', 'run1'])}
    mp = {k: A.mean_per(p) for k, p in S.items()}
    for hi, lo, rule, fam in (('OVC', 'A1R', 'overview crop - overview (agent)', 'M'), ('OVZ', 'OVC', 'enlarged crop - crop (agent)', 'R'), ('AG', 'OVZ', 'native zoom - enlarged crop (agent)', 'I')):
        ds = [d for d in T if d in mp[hi] and d in mp[lo]]
        add(fam, rule, f'REDP-X40 {v}', boot([mp[hi][d][1] - mp[lo][d][1] for d in ds]))

json.dump(OUT, open('results_law/theory_check.json', 'w'), indent=1)
tested = [o for o in OUT if o['verdict'] != 'n/a']
print(f'{len(tested)} predicted orderings: supported {sum(o["verdict"] == "supported" for o in tested)}, consistent {sum(o["verdict"] == "consistent" for o in tested)}, violated {sum(o["verdict"] == "violated" for o in tested)}')
for o in OUT:
    print(f'{o["family"]:7s} {o["rule"][:48]:48s} {o["label"][:26]:26s} {o["diff"]:+.3f} [{o["lo"]:+.3f}, {o["hi"]:+.3f}] n={o["n"]:4d} {o["verdict"]}')
