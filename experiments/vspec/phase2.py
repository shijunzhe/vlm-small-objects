#!/usr/bin/env python3
"""Phases 2-3: capacity grid on frozen S90 (design v3 sections 5-6).
usage: phase2.py freeze | wave | refine | status | analyze
  freeze : freeze S90 from phase 1, compute reachability, n_feas, planned standard cells, cost upper bound
  wave   : run standard levels wave by wave with adaptive skipping (recall < 0.5 stops larger n; >= 3 levels)
  refine : bisection levels (x = 2, 1.25, 1.5; b/a >= 1.5; at most 2 per (model, x)) and the geometric-limit level
"""
import json, sys, math, random
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import vspec as V, vrun as R, vcall as VC

ROOT = V.ROOT; OUT = ROOT / 'results'
XS = [1.0, 1.25, 1.5, 2.0]
NS = [8, 16, 32, 64, 128, 256]
N0 = 8
SET = 'P2'
RESID = {'gpt61': 155, 'gpt56': 155, 'opus55': 298, 'gem38': 241, 'qwen3vl': 247}   # text overhead from phase 0


def n_images(x):
    return 16 if x == 2.0 else 12


def frozen():
    return json.load(open(ROOT / 'frozen_s90.json'))


def scale(model, x):
    return x * frozen()[model]['S_ref']


def canvas_ok(model, n, S):
    Wu, Hu = V.extent(n)
    return V.canvas_for(model, Wu, Hu, S * V.MODELS[model]['t']) is not None


def n_feas(model, S, hi=4096):
    lo = 0
    for n in range(1, hi + 1):
        if canvas_ok(model, n, S):
            lo = n
        else:
            break
    return lo


def freeze():
    a = json.load(open(OUT / 'phase1_analysis.json'))
    fr = {}
    for m in V.ORDER:
        o = a[m]
        if o['status'] == 'identified':
            fr[m] = dict(S_ref=o['S90'], basis='S90', status='identified', ci90=o['ci90'])
        elif o['status'] == 'left_censored':
            fr[m] = dict(S_ref=0.2, basis='S90<=0.2 (left-censored; lowest level used)', status='left_censored', ci90=o['ci90'])
        elif o['status'] == 'precision_limited':
            s = o['S90_recall_only'] if o['status_recall_only'] in ('identified', 'left_censored') else 2.8
            fr[m] = dict(S_ref=s, basis='recall-only S90 (precision-limited)', status='precision_limited', ci90=o['ci90'])
        else:
            fr[m] = dict(S_ref=2.8, basis='right-censored; highest level used', status=o['status'], ci90=o['ci90'])
    json.dump(fr, open(ROOT / 'frozen_s90.json', 'w'), indent=1)
    plan = []
    feas = {}
    for m in V.ORDER:
        for x in XS + [3.0]:
            S = x * fr[m]['S_ref']; nf = n_feas(m, S); feas[f'{m}|{x}'] = nf
            for n in NS:
                if n <= nf:
                    plan.append(dict(model=m, x=x, n=n, S=S, k=n_images(x) if x != 3.0 else 8, optional=(x == 3.0)))
    # cost upper bound: every planned standard call at full 16,384 output
    ub = 0.0; exp_in = 0
    for c in plan:
        if c['optional']:
            continue
        Wu, Hu = V.extent(c['n']); W, H = V.canvas_for(c['model'], Wu, Hu, c['S'] * V.MODELS[c['model']]['t'])
        tin = V.expected_tokens(c['model'], W, H) + 100 + RESID[c['model']]
        pi, po = VC.PRICE[c['model']]
        ub += c['k'] * (tin * pi + 16384 * po) / 1e6; exp_in += c['k'] * tin
    json.dump(dict(frozen=fr, n_feas=feas, plan=plan, cost_upper_bound_standard=ub), open(ROOT / 'plan_p2.json', 'w'), indent=1)
    print(json.dumps(fr, indent=1))
    print('n_feas', feas)
    print('standard cells', sum(1 for c in plan if not c['optional']), 'calls', sum(c['k'] for c in plan if not c['optional']),
          '| x=3 cells', sum(1 for c in plan if c['optional']), '| cost UB standard $%.0f' % ub)


# ---------------- layouts and items ----------------
_LAY = {}


def get_layout(i, n, renders):
    key = (i, n)
    if key not in _LAY:
        cache = ROOT / 'layouts_p2.json'
        log = json.load(open(cache)) if cache.exists() else {}
        lay, k = V.frozen_layout(SET, i, n, renders)
        log[f'{i}|{n}'] = k
        json.dump(log, open(cache, 'w'), indent=1)
        _LAY[key] = lay
    return _LAY[key]


def renders_for(n, extra=()):
    """All (model, S) renders planned for this n in the standard grid (shared layout across models and x)."""
    pl = json.load(open(ROOT / 'plan_p2.json'))['plan']
    rr = [(c['model'], c['S']) for c in pl if c['n'] == n]
    return rr + list(extra)


def items_for(model, x, n, extra_renders=()):
    S = scale(model, x); k = n_images(x) if x != 3.0 else 8
    items = []
    for i in range(k):
        lay = get_layout(i, n, renders_for(n, extra_renders) if n in NS else list(extra_renders))
        if lay is None:
            continue
        it, r = R.make_item('p2', lay, model, S, f'x{x}')
        if it:
            it['x'] = x; items.append(it)
    return items


# ---------------- cell evaluation ----------------
def cell_rows(model, x, n, recs=None):
    S = scale(model, x)
    recs = recs if recs is not None else R.load_results([OUT / 'phase2.jsonl'])
    rows = []
    p = ROOT / 'items_p2.json'
    allit = json.load(open(p)) if p.exists() else []
    for it in allit:
        if it['model'] == model and abs(it['x'] - x) < 1e-9 and it['n'] == n and it['id'] in recs:
            rows.append((it['i'], R.score_record(it, recs[it['id']])))
    return rows


def save_items(new):
    p = ROOT / 'items_p2.json'
    allit = json.load(open(p)) if p.exists() else []
    have = {it['id'] for it in allit}
    allit += [it for it in new if it['id'] not in have]
    json.dump(allit, open(p, 'w'))


def state():
    """Per (model, x): tested levels with pooled results."""
    recs = R.load_results([OUT / 'phase2.jsonl'])
    p = ROOT / 'items_p2.json'
    allit = json.load(open(p)) if p.exists() else []
    by = defaultdict(lambda: defaultdict(list))
    for it in allit:
        if it['id'] in recs:
            by[(it['model'], it['x'])][it['n']].append((it['i'], R.score_record(it, recs[it['id']])))
    return by


def bracket(levels):
    """levels: {n: pooled}. a = largest tested n with all tested n0..n passing; b = first failing tested level."""
    a, b = None, None
    for n in sorted(levels):
        if levels[n]['pass_']:
            if b is None:
                a = n
        else:
            if b is None:
                b = n
    return a, b


def wave():
    pl = json.load(open(ROOT / 'plan_p2.json'))['plan']
    st = state()
    todo = []
    for m in V.ORDER:
        for x in XS:
            reach = [c['n'] for c in pl if c['model'] == m and c['x'] == x]
            tested = st.get((m, x), {})
            pooled = {n: R.pooled([r for _, r in v]) for n, v in tested.items()}
            # adaptive skip: stop after a level with recall < 0.5, once >= 3 levels tested
            stopped = any(pooled[n]['R'] < 0.5 for n in pooled) and len(pooled) >= min(3, len(reach))
            if stopped:
                continue
            nxt = [n for n in reach if n not in tested]
            if nxt:
                todo += items_for(m, x, nxt[0])
    if not todo:
        print('no standard levels left'); return False
    save_items(todo)
    print('wave:', len(todo), 'calls', sorted({(it['model'], it['x'], it['n']) for it in todo}))
    R.run_items(todo, OUT / 'phase2.jsonl', workers=12)
    return True


def refine_round():
    """One round of design-v3 5.3 refinement. Returns number of calls launched."""
    pl = json.load(open(ROOT / 'plan_p2.json'))
    feas = pl['n_feas']; st = state(); todo = []; added = []
    for m in V.ORDER:
        for x in (2.0, 1.25, 1.5):
            tested = st.get((m, x), {})
            if not tested:
                continue
            pooled = {n: R.pooled([r for _, r in v]) for n, v in tested.items()}
            n_extra = sum(1 for n in tested if n not in NS)
            a, b = bracket(pooled)
            new_n = None
            if b is not None and a is not None and b / a >= 1.5 and n_extra < 2:
                n = int(round(math.sqrt(a * b) / 2) * 2)
                if a < n < b and n not in tested:
                    new_n = n
            elif b is None and a is not None:
                nf = feas[f'{m}|{x}']; mx = max(tested)
                reach_std = [n for n in NS if n <= nf]
                if mx >= max(reach_std) and mx < 256 and nf < 256:   # geometry, not the 256 cap, stopped the grid
                    n = int(0.9 * nf)
                    if n >= 1.25 * mx and n not in tested and not any(k not in NS and k > mx for k in tested):
                        new_n = n
            if new_n is not None:
                its = items_for(m, x, new_n, extra_renders=[(m, scale(m, x))])
                todo += its; added.append((m, x, new_n, len(its)))
    if not todo:
        print('refine: nothing to add'); return 0
    save_items(todo)
    print('refine round:', added)
    R.run_items(todo, OUT / 'phase2.jsonl', workers=12)
    return len(todo)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'freeze':
        freeze()
    elif cmd == 'wave':
        p = ROOT / 'items_p2.json'
        if p.exists():   # resume any saved items without a result (e.g. after a container restart)
            R.run_items(json.load(open(p)), OUT / 'phase2.jsonl', workers=12)
        while wave():
            pass
    elif cmd == 'refine':
        for _ in range(3):
            if not refine_round():
                break
    elif cmd == 'status':
        st = state()
        for (m, x), lv in sorted(st.items()):
            pooled = {n: R.pooled([r for _, r in v]) for n, v in lv.items()}
            print(m, x, bracket(pooled), ' '.join(f"{n}:{p['R']:.2f}/{p['P']:.2f}{'*' if p['pass_'] else ''}(k{p['k']},t{p['trunc']})" for n, p in sorted(pooled.items())))
