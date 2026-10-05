#!/usr/bin/env python3
"""Validation of the learned-counter wrappers on natural images: CountGD and GeCo2 with three FSC-147 exemplar boxes
on the 150 X8 images (384 px release, whole image). Point-matching F1 as for the VLMs in X8 (tolerance m/2).
Threshold: leave-one-image-out (chosen on the other 149 images), so no image's own labels set its threshold.
Also count MAE. Compared with the VLM whole-image condition W under the domain-neutral prompt.
Writes results_x/learned2/fsc_counters.json."""
import json
from pathlib import Path
import numpy as np
import mech as M

ROOT = Path('../data/fsc147'); L2 = Path('results_x/learned2')
ITEMS = json.load(open(ROOT / 'x8_items.json')); ANN = json.load(open(ROOT / 'x8_ann_subset.json'))
GRID = np.round(np.arange(0.05, 0.96, 0.025), 3)


def load(method):
    out = {}
    for l in open(L2 / f'{method}_fsc_fsc-3.jsonl'):
        r = json.loads(l)
        pts = r['points']
        if method == 'geco2':   # relative score (score / image max), as on drawings
            pts = [(p[0], p[1], p[2] / max(p[5], 1e-9)) for p in pts]
        else:
            pts = [(p[0], p[1], p[2]) for p in pts]
        out[r['did']] = pts
    return out


def f1_at(pts, it, thr):
    gt = [tuple(p) for p in ANN[it['name']]['points']]
    sel = [(x, y) for x, y, s in pts if s >= thr]
    tp, n_p, n_g = M.match(sel, gt, it['m'] / 2)
    return 2 * tp / (n_p + n_g), abs(n_p - n_g)


def main():
    res = {}
    for method in ('countgd', 'geco2'):
        P = load(method); its = [it for it in ITEMS if it['name'] in P]
        tab = np.array([[f1_at(P[it['name']], it, t)[0] for t in GRID] for it in its])
        err = np.array([[f1_at(P[it['name']], it, t)[1] for t in GRID] for it in its])
        loo_f1, loo_err = [], []
        for i in range(len(its)):
            j = np.argmax(np.delete(tab, i, 0).mean(0)); loo_f1.append(tab[i, j]); loo_err.append(err[i, j])
        loo_f1 = np.array(loo_f1); loo_err = np.array(loo_err)
        r = {'n': len(its), 'f1': float(loo_f1.mean()), 'mae': float(loo_err.mean()), 'thr_global': float(GRID[np.argmax(tab.mean(0))])}
        for st in ('small', 'medium', 'large'):
            idx = [i for i, it in enumerate(its) if it['stratum'] == st]
            r[st] = {'f1': float(loo_f1[idx].mean()), 'mae': float(loo_err[idx].mean()), 'n': len(idx)}
        res[method] = r
        print(method, json.dumps(r))
    for v in ('gpt56', 'sonnet55', 'gem38'):
        try:
            s = json.load(open(ROOT / 'x8_scores_obj.json'))
        except FileNotFoundError:
            break
        vals = {st: [x['f1'] for k, x in s.items() if k.startswith(f'{v}|W|') and x['stratum'] == st] for st in ('small', 'medium', 'large')}
        allv = [x['f1'] for k, x in s.items() if k.startswith(f'{v}|W|')]
        res[f'vlm_W|{v}'] = {'f1': float(np.mean(allv)), **{st: float(np.mean(x)) for st, x in vals.items() if x}}
        print(v, res[f'vlm_W|{v}'])
    json.dump(res, open(L2 / 'fsc_counters.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
