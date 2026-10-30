#!/usr/bin/env python3
"""Amendment 4 follow-up: re-freeze validation working points with the corrected reader; models whose working
point changes are re-validated on the same P3 scenes (32 images). Also reports image-level exact bounds."""
import json, sys, math, random
from collections import defaultdict
import numpy as np
from scipy.stats import beta
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import vspec as V, vrun as R, phase_val as PV, phase2 as P2
ROOT = V.ROOT; OUT = ROOT / 'results'

def exact_lb(k, n, a=0.05):
    return 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))

def freeze():
    old = json.load(open(ROOT / 'frozen_validation.json')); new = {}
    for m in V.ORDER:
        if 'n_safe' not in old[m]:
            new[m] = old[m]; continue
        cc, br = PV.c_cons(m)
        n_safe = 8 if cc <= 8 else int(math.floor(8 + 0.5 * (cc - 8)))
        new[m] = dict(old[m], C_cons=cc, n_safe=n_safe, C_ref_bracket=br, changed=(n_safe != old[m]['n_safe']))
    json.dump(new, open(ROOT / 'frozen_validation_amend4.json', 'w'), indent=1)
    print(json.dumps(new, indent=1))

def run():
    fz = json.load(open(ROOT / 'frozen_validation_amend4.json')); its = []
    for m, v in fz.items():
        if not v.get('changed'): continue
        for i in range(32):
            lay, k = V.frozen_layout('P3', i, v['n_safe'], [(m, v['S'])])
            it, r = R.make_item('val4', lay, m, v['S'], 'safe') if lay else (None, 'no layout')
            if it: its.append(it)
            else: print('skip', m, i, r)
    json.dump(its, open(ROOT / 'items_val_amend4.json', 'w'))
    R.run_items(its, OUT / 'validation_amend4.jsonl', workers=12)

def analyze():
    rows = defaultdict(list)
    for itf, rf in [('items_val.json', 'validation.jsonl'), ('items_val_amend4.json', 'validation_amend4.jsonl'),
                    ('items_val_gem4.json', 'validation_gem4.jsonl')]:
        p = ROOT / itf
        if not p.exists(): continue
        recs = R.load_results([OUT / rf])
        for it in json.load(open(p)):
            if it['id'] in recs:
                rows[(it['model'], it['n'], it['phase'])].append((it['i'], R.score_record(it, recs[it['id']])))
    res = {}
    for (m, n, ph), v in sorted(rows.items()):
        p = R.pooled([r for _, r in v]); k = len(v)
        ok = [r for _, r in v if r['n_pred'] > 0 and r['tp'] / r['n_gt'] >= 0.9 and r['tp'] / r['n_pred'] >= 0.9]
        perfect = sum(1 for _, r in v if r['tp'] == r['n_gt'] == r['n_pred'])
        res[f'{m}|n{n}|{ph}'] = dict(model=m, n=n, phase=ph, k=k, R=round(p['R'], 4), P=round(p['P'], 4),
                                     images_meeting_q=len(ok), images_perfect=perfect,
                                     exact_lb95_image_q=round(exact_lb(len(ok), k), 3),
                                     failures=[(i, r['tp'], r['n_pred'], r['n_gt']) for i, r in v if r not in ok])
    json.dump(res, open(OUT / 'validation_summary_amend4.json', 'w'), indent=1)
    for kk, r in res.items():
        print(kk, {a: b for a, b in r.items() if a not in ('model', 'n', 'phase')})

if __name__ == '__main__':
    {'freeze': freeze, 'run': run, 'analyze': analyze}[sys.argv[1]]()
