#!/usr/bin/env python3
"""Dev-set check of the geometric ceilings: every observed (model, condition) cell vs. its a priori
ceiling from ceilings.ideal_observer, on the same drawings. The ceiling is a necessary condition, so
the check is: observed <= ceiling in every cell (up to run noise). The gap is the non-geometric part."""
import contextlib, io, json
import numpy as np
import mech as M
import ceilings as K
with contextlib.redirect_stdout(io.StringIO()):
    from x_score import cellstats
    import x_score2 as XS

DEV = K.DEV
ANN = {d: json.loads((M.BENCH / 'annotations' / f'{d}.json').read_text()) for d in DEV}
GPT_Z0 = lambda a: 0.75 * 29.2 / min(a['template_size'])


def ceil_for(side_fn, dids):
    v = []
    for d in dids:
        a = ANN[d]; tau, _ = M.taus(a)
        v.append(K.ideal_observer(a['points'], side_fn(a), tau, n_off=32)[0])
    return float(np.mean(v))


def whole(runs):
    per = {}
    for d in DEV:
        if d not in runs:
            continue
        a = ANN[d]; tau, _ = M.taus(a)
        per[d] = np.mean([2 * M.match(p, a['points'], tau)[0] / (len(p) + a['count']) for p in runs[d]])
    return per


rows = []
def add(label, per, side_fn):
    dids = sorted(per)
    rows.append((label, float(np.mean([per[d] for d in dids])), ceil_for(side_fn, dids), len(dids)))


# whole drawing (A1) and vendor high-res (B)
for mdl in ('gpt', 'claude', 'gemini', 'gemma4'):
    add(f'A1 whole {mdl}', whole(XS.full_metrics(mdl)), lambda a, m=mdl: K.token_side('A1', m, a))
add('B gpt original', whole(XS.x4['gpt_original']), lambda a: K.token_side('B', 'gpt', a))
add('B gemini HIGH', whole(XS.x4['gemini_high']), lambda a: K.token_side('B', 'gemini', a))

# 44-px tiles (x2), GPT/Claude
for mdl, patch in (('gpt', 32), ('claude', 28)):
    s = cellstats([M.OUT / f'x2_{mdl}.jsonl', M.OUT / f'x2_{mdl}_run1.jsonl'], ['T512', 'T768', 'T1024'])
    for c, v in s.items():
        add(f'{c} tiles {mdl}', v['per'], lambda a, p=patch: p * min(a['template_size']) / 44)
# z0-scaled cells (step2): symbol = 0.75 GPT tokens, upsampled x up
for mdl, patch, files in (('gpt', 32, ['step2_gpt_run0.jsonl', 'step2_gpt_run1.jsonl']), ('claude', 28, ['step2_claude_run0.jsonl'])):
    s = cellstats([M.OUT / f for f in files], ['L1', 'L2', 'S1', 'S2', 'L2n'])
    for c, v in s.items():
        up = 2 if c[1] == '2' else 1
        add(f'{c} step2 {mdl}', v['per'], lambda a, p=patch, u=up: p / (GPT_Z0(a) * u))
# fixed-grid FOV sweep (x3)
for mdl, side in (('gemini', 16), ('gemma4', K.GEMMA_SIDE)):
    s = cellstats([M.OUT / f'x3_{mdl}_run0.jsonl'], ['k1.0', 'k1.5', 'k2.5'])
    for c, v in s.items():
        k = float(c[1:])
        add(f'{c} FOV {mdl}', v['per'], lambda a, k=k, sd=side: max(96, int(round(16 * min(a['template_size']) / k))) / sd)
# phase 0 cells
for f, c, mdl, fn in (('p0_qwen_qwen3vl.jsonl', 'T1024', 'qwen', lambda a: 32 * min(a['template_size']) / 44),
                      ('p0_pad_gpt.jsonl', 'S1pad', 'gpt', lambda a: 32 / GPT_Z0(a)),
                      ('p0_pad_claude.jsonl', 'S1pad', 'claude', lambda a: 28 / GPT_Z0(a)),
                      ('p0_ruler_gemini.jsonl', 'k1.5ruler', 'gemini', lambda a: max(96, int(round(16 * min(a['template_size']) / 1.5))) / 16)):
    s = cellstats([M.OUT / f], [c])
    for cc, v in s.items():
        add(f'{cc} p0 {mdl}', v['per'], fn)

# half-masked tiles: two records per tile (key includes the half); keep only points in the visible half
from collections import defaultdict
for mdl, patch in (('gpt', 32), ('claude', 28)):
    acc = defaultdict(lambda: {'ne': [], 'fpe': 0}); seen = {}
    for l in open(M.OUT / f'p0_mask_{mdl}.jsonl'):
        r = json.loads(l); seen[(r['did'], r['i'], r['j'], r['half'])] = r
    for r in seen.values():
        if r['status'] != 'success':
            continue
        a = ANN[r['did']]; ca, cb, cc, ce = r['core_px']; x0, y0 = r['origin_z0']; z0, up = r['z0'], r['up']; lo, hi = r['keep_x']
        tx = lambda gx: (gx * z0 - x0) * up
        ng = sum(ca <= tx(gx) < cc and cb <= (gy * z0 - y0) * up < ce and lo <= tx(gx) < hi for gx, gy in a['points'])
        pts = [p for p in M.to_native(r) if lo <= tx(p[0]) < hi]
        if ng: acc[r['did']]['ne'] += pts
        else: acc[r['did']]['fpe'] += len(pts)
    per = {}
    for d, q in acc.items():
        a = ANN[d]; tau, _ = M.taus(a); tp, n_p, n_g = M.match(q['ne'], a['points'], tau)
        per[d] = 2 * tp / (n_p + q['fpe'] + n_g)
    add(f'T1024mask p0 {mdl}', per, lambda a, p=patch: p * min(a['template_size']) / 44)

print(f'{"cell":24s} observed  ceiling  gap   n')
viol = 0
for label, obs, ceil, n in rows:
    flag = '  <-- above ceiling' if obs > ceil + 0.02 else ''
    viol += bool(flag)
    print(f'{label:24s} {obs:6.2f}   {ceil:6.2f}  {ceil - obs:5.2f}  {n}{flag}')
print(f'\n{len(rows)} cells, {viol} above ceiling (+0.02 tolerance)')
obs = np.array([r[1] for r in rows]); cel = np.array([r[2] for r in rows])
print(f'Spearman(obs, ceiling) over cells: {np.corrcoef(np.argsort(np.argsort(obs)), np.argsort(np.argsort(cel)))[0,1]:.2f}')
json.dump([dict(zip(['cell', 'observed', 'ceiling', 'n'], r)) for r in rows],
          open(M.OUT.parent / 'ceiling_check_dev.json', 'w'), indent=1)
