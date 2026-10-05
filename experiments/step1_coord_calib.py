#!/usr/bin/env python3
"""Step 1: coordinate-interface calibration. For each model, the same 10 tiles
(2 per drawing: the two cores with the most instances, cell FOV=L, up=2) are
queried with pixel coordinates and with 0-1000 normalized coordinates.
The interface with the higher localization F1 (tau_sym) is fixed for step 2."""
import json, sys
from collections import defaultdict
import numpy as np
import mech as M

DIDS = ['d001', 'd004', 'd010', 'd017', 'd019']
MODELS = ['gpt', 'claude', 'gemini', 'gemma4']
OUTP = M.OUT / 'step1_coord.jsonl'


def pick_tiles():
    sel = []
    for did in DIDS:
        ann, img, tpl = M.load(did)
        ts = M.tiles_for(ann, img, 'L', 2)
        z0 = ts[0]['z0']
        def n_in(t):
            a, b, c, e = t['core_px']; x0, y0 = t['origin_z0']
            cnt = 0
            for gx, gy in ann['points']:
                lx, ly = (gx * z0 - x0) * 2, (gy * z0 - y0) * 2
                cnt += a <= lx < c and b <= ly < e
            return cnt
        ts.sort(key=n_in, reverse=True)
        tplz = M.scaled_template(tpl, z0, 2)
        for t in ts[:2]:
            sel.append((did, ann, t, tplz, n_in(t)))
    return sel


def main():
    M.C.load_env('../config.env')
    sel = pick_tiles()
    print('tiles:', [(d, t['i'], t['j'], n) for d, _, t, _, n in sel])
    jobs = []
    for model in MODELS:
        for coord in ('px', 'n1000'):
            for did, ann, t, tplz, n in sel:
                jobs.append({'model': model, 'coord': coord, 'tpl': tplz, 'tile': t,
                             'meta': {'did': did, 'model': model, 'coord': coord, 'i': t['i'], 'j': t['j'],
                                      'cell': 'L2', 'n_gt_core': n}})
    M.run_jobs(jobs, OUTP, key=lambda r: (r['model'], r['coord'], r['did'], r['i'], r['j']), workers=8)
    summarize(sel)


def summarize(sel=None):
    anns = {d: json.loads((M.BENCH / 'annotations' / f'{d}.json').read_text()) for d in DIDS}
    agg = defaultdict(lambda: {'tp': 0, 'np': 0, 'ng': 0, 'd': [], 'bad': 0})
    for l in open(OUTP):
        r = json.loads(l)
        k = (r['model'], r['coord'])
        if r['status'] != 'success':
            agg[k]['bad'] += 1; continue
        ann = anns[r['did']]
        tsym, _ = M.taus(ann)
        a, b, c, e = r['core_px']; x0, y0 = r['origin_z0']; z0, up = r['z0'], r['up']
        gt = [(gx, gy) for gx, gy in ann['points']
              if a <= (gx * z0 - x0) * up < c and b <= (gy * z0 - y0) * up < e]
        pred = M.to_native(r)
        tp, n_p, n_g = M.match(pred, gt, tsym)
        agg[k]['tp'] += tp; agg[k]['np'] += n_p; agg[k]['ng'] += n_g
        tp3, _, _ = M.match(pred, gt, 3 * tsym)
        agg[k]['d'].append(tp3)
    print('model   coord  F1_sym  recall  precision  (pred/gt)  loose-matches@3tau  failed')
    for (m, c), v in sorted(agg.items()):
        p = v['tp'] / max(v['np'], 1); rc = v['tp'] / max(v['ng'], 1)
        f = 2 * v['tp'] / max(v['np'] + v['ng'], 1)
        print(f"{m:7s} {c:6s} {f:.2f}    {rc:.2f}    {p:.2f}      ({v['np']}/{v['ng']})     {sum(v['d'])}   {v['bad']}")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summarize()
    else:
        main()
