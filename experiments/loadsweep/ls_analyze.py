#!/usr/bin/env python3
"""Score the S = 1.5 load sweep (ls_run.py) with the visual-spec reader (vrun.py, reader v2): per model and load,
pooled recall and precision, images with recall and precision both >= 0.9, empty answers, truncated answers,
and the answers that are not success (missing images). Writes results/ls_summary.json."""
import json, sys
from pathlib import Path
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ls_run as L
import vrun as R

HERE = Path(__file__).resolve().parent
items = {it['id']: it for it in json.load(open(L.RES / 'items.json'))}
out = {}
for m, ns in L.LOADS.items():
    p = L.RES / f'LS_{m}.jsonl'
    recs = {}
    if p.exists():
        for l in p.open():
            r = json.loads(l)
            if r['status'] == 'success': recs[r['id']] = r
    for n in ns:
        its = [it for it in items.values() if it['model'] == m and it['n'] == n]
        rows = []
        for it in its:
            if it['id'] not in recs: continue
            sc = R.score_record(it, recs[it['id']]); u = recs[it['id']].get('usage', {})
            sc['out_tokens'] = (u.get('output_tokens') or 0)
            rows.append(sc)
        if not rows:
            out[f'{m}|{n}'] = dict(model=m, n=n, k=0, planned=len(its)); continue
        po = R.pooled(rows)
        joint = sum(1 for r in rows if r['n_pred'] > 0 and r['tp'] / r['n_gt'] >= 0.9 and r['tp'] / r['n_pred'] >= 0.9)
        out[f'{m}|{n}'] = dict(model=m, n=n, planned=len(its), k=len(rows), R=po['R'], P=po['P'], F1=po['F1'], joint=joint,
                               empty=sum(1 for r in rows if r['n_pred'] == 0), trunc=sum(r['truncated'] for r in rows),
                               parse_fail=sum(1 for r in rows if not r['parse_ok']),
                               over=sum(1 for r in rows if r['n_pred'] > r['n_gt']), under=sum(1 for r in rows if r['n_pred'] < r['n_gt']),
                               max_out=max(r['out_tokens'] for r in rows))
json.dump(out, open(L.RES / 'ls_summary.json', 'w'), indent=1)
for k, v in out.items():
    if v['k']:
        print(f"{k:14s} k={v['k']:2d}/{v['planned']} R={v['R']:.2f} P={v['P']:.2f} joint={v['joint']:2d} empty={v['empty']} trunc={v['trunc']} parsefail={v['parse_fail']} over={v['over']} under={v['under']} maxout={v['max_out']}")
    else:
        print(k, 'no answers')
