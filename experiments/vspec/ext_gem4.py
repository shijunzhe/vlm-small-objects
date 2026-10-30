#!/usr/bin/env python3
"""Extension (2026-10-26, user request): Gemini 4 Argon under the frozen v3 protocol (with amendments 1-2).
Separate files so the frozen five-model artefacts are never touched.
usage: ext_gem4.py dry | p1 | freeze | wave | refine | val_freeze | val | status"""
import json, sys, math, random
from collections import defaultdict
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import vspec as V, vrun as R, phase1 as P1, phase2 as P2, phase0 as P0

M = 'gem4'; ROOT = V.ROOT; OUT = ROOT / 'results'
XS, NS, N0 = P2.XS, P2.NS, 8
F_FROZEN = ROOT / 'frozen_s90_gem4.json'; F_PLAN = ROOT / 'plan_p2_gem4.json'
F_IT2 = ROOT / 'items_p2_gem4.json'; F_RES2 = OUT / 'phase2_gem4.jsonl'
F_LAY_NEW = ROOT / 'layouts_p2_gem4.json'


def dry():
    spec = [(8, 0.45), (8, 0.9), (8, 1.9), (32, 1.0), (128, 0.8)]
    its = []
    for j, (n, S) in enumerate(spec):
        lay, k = V.frozen_layout('P0', j, n, [(m, S) for m in V.ORDER])
        it, r = R.make_item('p0', lay, M, S, 'dry')
        if it: its.append(it)
        else: print('dry unreachable', n, S, r)
    R.run_items(its, OUT / 'phase0_dry_gem4.jsonl', workers=5)
    recs = R.load_results([OUT / 'phase0_dry_gem4.jsonl'])
    for it in its:
        r = recs.get(it['id'])
        if r:
            print(it['n'], it['S'], R.score_record(it, r), r['usage'], round(r['cost'], 4))


def p1_items():
    ks = {d['i']: d['k'] for d in json.load(open(ROOT / 'freeze_p1_log.json'))['layout_log']}
    its = []
    for i in range(12):
        lay = V.layout('P1', i, N0, ks[i])
        for S in P1.S_LEVELS:
            it, r = R.make_item('p1', lay, M, S, 'S')
            assert it is not None, (i, S, r)
            its.append(it)
    json.dump(its, open(ROOT / 'items_p1_gem4.json', 'w'), indent=1)
    return its


def p1():
    R.run_items(p1_items(), OUT / 'phase1_gem4.jsonl', workers=12)


def freeze(B=2000):
    its = json.load(open(ROOT / 'items_p1_gem4.json')); recs = R.load_results([OUT / 'phase1_gem4.jsonl'])
    rows = defaultdict(list)
    for it in its:
        if it['id'] in recs:
            rows[it['S']].append((it['i'], R.score_record(it, recs[it['id']])))
    cells = {S: R.pooled([r for _, r in rows[S]]) for S in P1.S_LEVELS}
    s90, st = P1.s90_from_cells(cells)
    rng = random.Random(0); ids = sorted({i for v in rows.values() for i, _ in v}); boots = []
    for b in range(B):
        samp = [rng.choice(ids) for _ in ids]; bc = {}
        for S in P1.S_LEVELS:
            by = defaultdict(list)
            for i, r in rows[S]: by[i].append(r)
            bc[S] = R.pooled([r for i in samp for r in by[i]])
        v, s = P1.s90_from_cells(bc); boots.append(v if s in ('identified', 'left_censored') else float('inf'))
    arr = np.array(boots)
    S_ref = s90 if st == 'identified' else (0.2 if st == 'left_censored' else 2.8)
    fr = dict(S_ref=S_ref, status=st, S90=s90, ci90=[float(np.quantile(arr, .05)), float(np.quantile(arr, .95))],
              cells={str(S): {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for S, c in cells.items()})
    json.dump(fr, open(F_FROZEN, 'w'), indent=1)
    plan = []; feas = {}
    for x in XS + [3.0]:
        S = x * S_ref; nf = P2.n_feas(M, S); feas[str(x)] = nf
        for n in NS:
            if n <= nf:
                plan.append(dict(model=M, x=x, n=n, S=S, k=P2.n_images(x) if x != 3.0 else 8, optional=(x == 3.0)))
    json.dump(dict(frozen=fr, n_feas=feas, plan=plan), open(F_PLAN, 'w'), indent=1)
    print({k: v for k, v in fr.items() if k != 'cells'})
    for S, c in fr['cells'].items():
        print(f"   S={S:>5} R={c['R']:.3f} P={c['P']:.3f} pass={c['pass_']} trunc={c['trunc']}")
    print('n_feas', feas, 'standard calls', sum(c['k'] for c in plan if not c['optional']))


# ---------------- layouts: reuse the frozen P2 layouts; new n get their own frozen layout ----------------
_KS = json.load(open(ROOT / 'layouts_p2.json')) if (ROOT / 'layouts_p2.json').exists() else {}


def layout_for(i, n, S):
    key = f'{i}|{n}'
    if key in _KS and _KS[key] is not None:
        return V.layout('P2', i, n, _KS[key]), 'shared'
    new = json.load(open(F_LAY_NEW)) if F_LAY_NEW.exists() else {}
    if key in new:
        return (V.layout('P2', i, n, new[key]) if new[key] is not None else None), 'own'
    lay, k = V.frozen_layout('P2', i, n, [(M, S)])
    new[key] = k; json.dump(new, open(F_LAY_NEW, 'w'), indent=1)
    return lay, 'own'


def items_for(x, n):
    fr = json.load(open(F_FROZEN)); S = x * fr['S_ref']; k = P2.n_images(x)
    its = []
    for i in range(k):
        lay, src = layout_for(i, n, S)
        if lay is None:
            print('no layout', i, n); continue
        it, r = R.make_item('p2', lay, M, S, f'x{x}')
        if it:
            it['x'] = x; it['layout_src'] = src; its.append(it)
        else:
            print('render fail on shared layout (skipped)', i, n, r)
    return its


def save(new):
    allit = json.load(open(F_IT2)) if F_IT2.exists() else []
    have = {it['id'] for it in allit}; allit += [it for it in new if it['id'] not in have]
    json.dump(allit, open(F_IT2, 'w'))


def state():
    recs = R.load_results([F_RES2]); allit = json.load(open(F_IT2)) if F_IT2.exists() else []
    by = defaultdict(lambda: defaultdict(list))
    for it in allit:
        if it['id'] in recs:
            by[it['x']][it['n']].append((it['i'], R.score_record(it, recs[it['id']])))
    return by


def wave():
    plan = json.load(open(F_PLAN))['plan']; st = state(); todo = []
    for x in XS:
        reach = [c['n'] for c in plan if c['x'] == x and not c['optional']]
        tested = st.get(x, {}); pooled = {n: R.pooled([r for _, r in v]) for n, v in tested.items()}
        if any(pooled[n]['R'] < 0.5 for n in pooled) and len(pooled) >= min(3, len(reach)):
            continue
        nxt = [n for n in reach if n not in tested]
        if nxt:
            todo += items_for(x, nxt[0])
    if not todo:
        print('no standard levels left'); return False
    save(todo); print('wave', len(todo), sorted({(it['x'], it['n']) for it in todo}))
    R.run_items(todo, F_RES2, workers=12); return True


def refine_round():
    plan = json.load(open(F_PLAN)); feas = plan['n_feas']; st = state(); todo = []; added = []
    for x in (2.0, 1.25, 1.5):
        tested = st.get(x, {})
        if not tested: continue
        pooled = {n: R.pooled([r for _, r in v]) for n, v in tested.items()}
        n_extra = sum(1 for n in tested if n not in NS); a, b = P2.bracket(pooled); new_n = None
        if b is not None and a is not None and b / a >= 1.5 and n_extra < 2:
            n = int(round(math.sqrt(a * b) / 2) * 2)
            if a < n < b and n not in tested: new_n = n
        elif b is None and a is not None:
            nf = feas[str(x)]; mx = max(tested); reach_std = [n for n in NS if n <= nf]
            if mx >= max(reach_std) and mx < 256 and nf < 256:
                n = int(0.9 * nf)
                if n >= 1.25 * mx and n not in tested and not any(k not in NS and k > mx for k in tested): new_n = n
        if new_n is not None:
            its = items_for(x, new_n); todo += its; added.append((x, new_n, len(its)))
    if not todo:
        print('refine: nothing to add'); return 0
    save(todo); print('refine round', added); R.run_items(todo, F_RES2, workers=12); return len(todo)


def status():
    st = state()
    for x, lv in sorted(st.items()):
        pooled = {n: R.pooled([r for _, r in v]) for n, v in lv.items()}
        print(M, x, P2.bracket(pooled), ' '.join(f"{n}:{p['R']:.2f}/{p['P']:.2f}{'*' if p['pass_'] else ''}(k{p['k']},t{p['trunc']})" for n, p in sorted(pooled.items())))


def val_freeze(B=2000):
    st = state().get(2.0, {}); fr = json.load(open(F_FROZEN)); S = 2.0 * fr['S_ref']
    if not st:
        out = dict(status='not testable', S=S)
    else:
        ids = sorted({i for v in st.values() for i, _ in v}); by = {n: defaultdict(list) for n in st}
        for n, v in st.items():
            for i, r in v: by[n][i].append(r)
        rng = random.Random(0); A = []
        for b in range(B):
            samp = [rng.choice(ids) for _ in ids]
            a, _ = P2.bracket({n: R.pooled([r for i in samp for r in by[n][i]]) for n in st}); A.append(a if a is not None else 0)
        cc = float(np.quantile(A, 0.05, method='lower'))
        br = P2.bracket({n: R.pooled([r for _, r in v]) for n, v in st.items()})
        out = dict(S=S, C_ref_bracket=br, C_cons=cc, n_safe=N0 if cc <= N0 else int(math.floor(N0 + 0.5 * (cc - N0))), k=32)
    json.dump(out, open(ROOT / 'frozen_validation_gem4.json', 'w'), indent=1); print(out)


def val():
    fz = json.load(open(ROOT / 'frozen_validation_gem4.json')); n, S = fz['n_safe'], fz['S']
    orig = json.load(open(ROOT / 'frozen_validation.json'))
    renders_orig = [(m, v['S']) for m, v in orig.items() if v.get('n_safe') == n]
    its = []
    for i in range(32):
        lay, k = V.frozen_layout('P3', i, n, renders_orig if renders_orig else [(M, S)])
        it, r = R.make_item('val', lay, M, S, 'safe') if lay else (None, 'no layout')
        if it: its.append(it)
        else: print('val skip', i, r)
    json.dump(its, open(ROOT / 'items_val_gem4.json', 'w'))
    R.run_items(its, OUT / 'validation_gem4.jsonl', workers=12)
    recs = R.load_results([OUT / 'validation_gem4.jsonl'])
    rows = [(it['i'], R.score_record(it, recs[it['id']])) for it in its if it['id'] in recs]
    p = R.pooled([r for _, r in rows]); rng = random.Random(1); by = defaultdict(list)
    for i, r in rows: by[i].append(r)
    ids = sorted(by); Rs, Ps = [], []
    for b in range(2000):
        q = R.pooled([r for i in [rng.choice(ids) for _ in ids] for r in by[i]]); Rs.append(q['R']); Ps.append(q['P'] if q['P'] == q['P'] else 0)
    res = dict(fz, R=p['R'], P=p['P'], lb95_R=float(np.quantile(Rs, .05)), lb95_P=float(np.quantile(Ps, .05)), k=p['k'], trunc=p['trunc'])
    json.dump(res, open(OUT / 'validation_gem4_analysis.json', 'w'), indent=1); print(res)


def topup():
    """Items skipped because the shared P2 layout failed the pixel check at the Argon scale get their own layout
    (same template combination, re-drawn positions; first k passing at that render). Recorded as layout_src='own_topup'."""
    fr = json.load(open(F_FROZEN)); st = state(); allit = json.load(open(F_IT2)); have = {(it['x'], it['n'], it['i']) for it in allit}
    new = json.load(open(F_LAY_NEW)) if F_LAY_NEW.exists() else {}; todo = []
    for x, lv in st.items():
        for n in lv:
            S = x * fr['S_ref']
            for i in range(P2.n_images(x)):
                if (x, n, i) in have: continue
                lay, k = V.frozen_layout('P2', i, n, [(M, S)])
                new[f'{i}|{n}|x{x}'] = k
                it, r = R.make_item('p2', lay, M, S, f'x{x}') if lay else (None, 'none')
                if it:
                    it['x'] = x; it['layout_src'] = 'own_topup'; todo.append(it)
                else: print('topup fail', x, n, i, r)
    json.dump(new, open(F_LAY_NEW, 'w'), indent=1)
    save(todo); print('topup', len(todo), sorted({(it['x'], it['n'], it['i']) for it in todo}))
    R.run_items(todo, F_RES2, workers=12)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'wave':
        if F_IT2.exists(): R.run_items(json.load(open(F_IT2)), F_RES2, workers=12)
        while wave(): pass
    elif cmd == 'refine':
        for _ in range(3):
            if not refine_round(): break
    else:
        {'topup': topup, 'dry': dry, 'p1': p1, 'freeze': freeze, 'status': status, 'val_freeze': val_freeze, 'val': val}[cmd]()
