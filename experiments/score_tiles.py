#!/usr/bin/env python3
"""Score the S_tok pilot. Per drawing and level: count (programmatic, from kept list),
absolute count error, F1 (Hungarian) at tau_1600 = 30 px in 1600-space and at
tau_sym = 0.5*sqrt(template area), both in native pixels. Level `full` reuses the E2
counts-first responses (list readout, averaged over runs)."""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

BENCH = Path('../data/redp40/redp_x40')
E2 = Path('vlm_api/results/e2')
TOK = {'gpt': 29.2, 'claude': 27.4, 'qwen3vl': 33.0, 'gemini': 100.0, 'gemma4': 100.0}


def f1(pred, gt, tau):
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    P, G = np.array(pred, float), np.array(gt, float)
    D = np.linalg.norm(P[:, None] - G[None], axis=2)
    r, c = linear_sum_assignment(np.where(D <= tau, D, 1e6))
    return 2 * int((D[r, c] <= tau).sum()) / (len(P) + len(G))


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_tiles(model):
    per = defaultdict(lambda: defaultdict(list))   # did -> level -> list of native points (single run)
    stat = defaultdict(lambda: defaultdict(lambda: {'tiles': 0, 'bad': 0, 'outside': 0}))
    for l in open(f'results/{model}.jsonl'):
        r = json.loads(l)
        s = stat[r['did']][r['level']]
        s['tiles'] += 1
        if r['status'] != 'success':
            s['bad'] += 1
            continue
        x0, y0 = r['origin']; a, b, c, d = r['core']; z = r['zoom']
        pts = per[r['did']][r['level']]
        for sh in r['shapes']:
            if not isinstance(sh, dict) or fnum(sh.get('cx')) is None or fnum(sh.get('cy')) is None:
                continue
            cx, cy = fnum(sh['cx']), fnum(sh['cy'])
            if a <= cx < c and b <= cy < d:
                pts.append(((x0 + cx) / z, (y0 + cy) / z))
            else:
                s['outside'] += 1
    return per, stat


def load_full(model, did_num):
    rows = []
    extra = Path(f'results/{model}_full_extra.jsonl')
    if extra.exists():
        for l in open(extra):
            r = json.loads(l)
            if r['status'] == 'success' and int(r['did'][1:]) == did_num:
                s = r['scale']
                rows.append([(sh['cx'] / s, sh['cy'] / s) for sh in r['shapes']
                             if isinstance(sh, dict) and num(sh.get('cx')) and num(sh.get('cy'))])
    if rows:
        return rows
    for l in open(E2 / f'{model}.jsonl'):
        r = json.loads(l)
        if r['status'] == 'success' and r['order'] == 'counts_first' and r['item_id'] == did_num:
            s = r['scale']
            rows.append([(sh['cx'] / s, sh['cy'] / s) for sh in r['shapes']
                         if isinstance(sh, dict) and num(sh.get('cx')) and num(sh.get('cy'))])
    return rows


def main(model='gpt', levels=('full', 'tile0', 's1.5', 's3')):
    per, stat = load_tiles(model)
    res = defaultdict(dict)
    for did in sorted(per):
        ann = json.loads((BENCH / 'annotations' / f'{did}.json').read_text())
        gt = ann['points']; g = len(gt)
        tw, th = ann['template_size']
        L = max(ann['width'], ann['height'])
        t16 = 30 / min(1, 1600 / L); tsym = 0.5 * math.sqrt(tw * th)
        for lev in levels:
            if lev == 'full':
                runs = load_full(model, int(did[1:]))
                if not runs:
                    continue
                res[did][lev] = {'n': np.mean([len(p) for p in runs]),
                                 'err': np.mean([abs(len(p) - g) for p in runs]),
                                 'f1_16': np.mean([f1(p, gt, t16) for p in runs]),
                                 'f1_sym': np.mean([f1(p, gt, tsym) for p in runs]), 'runs': len(runs)}
            elif lev in per[did]:
                p = per[did][lev]
                res[did][lev] = {'n': len(p), 'err': abs(len(p) - g), 'f1_16': f1(p, gt, t16),
                                 'f1_sym': f1(p, gt, tsym), 'tiles': stat[did][lev]['tiles'],
                                 'bad': stat[did][lev]['bad'], 'outside': stat[did][lev]['outside']}
        res[did]['gt'] = g
        res[did]['stok_full'] = min(tw, th) * min(1, 1600 / L) / TOK[model]
    print(f'model={model}')
    print('did   gt  S_tok@full | ' + ' | '.join(f'{lv:>5}: n  err  F1_30  F1_sym' for lv in levels))
    for did, r in res.items():
        line = f"{did} {r['gt']:4d}  {r['stok_full']:.2f}      | "
        for lv in levels:
            x = r.get(lv)
            line += (f"{lv:>5}: {x['n']:5.1f} {x['err']:5.1f} {x['f1_16']:.2f} {x['f1_sym']:.2f} | " if x else f"{lv:>5}: {'-':>24} | ")
        print(line)
    print('mean  ', ' | '.join(
        f"{lv}: err {np.mean([r[lv]['err'] for r in res.values() if lv in r]):.1f} "
        f"relerr {np.mean([r[lv]['err']/r['gt'] for r in res.values() if lv in r]):.2f} "
        f"F1_30 {np.mean([r[lv]['f1_16'] for r in res.values() if lv in r]):.2f} "
        f"F1_sym {np.mean([r[lv]['f1_sym'] for r in res.values() if lv in r]):.2f} (n={sum(lv in r for r in res.values())})"
        for lv in levels))
    json.dump(res, open(f'score_{model}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'gpt')
