#!/usr/bin/env python3
"""Gemini 4 whole-image answers under the pixel prompt (A1): how many contain coordinates above 1000, and recall under
three single readings (all pixels, all 0-1000, per answer). Run from the release's experiments/ directory."""
import contextlib, io, json, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
with contextlib.redirect_stdout(io.StringIO()):
    import cs_analyze as C
from pathlib import Path
import glob
out = {}
for bench in ('redp', 'fpc'):
    if bench == 'fpc':
        C.CR.BENCH_ARG = 'fpc'; C.M.BENCH = Path('../data/fpc/sheets'); f = C.RF / 'A1_gem4_run0.jsonl'; metric = C.F.metrics
    else:
        f = C.RG / 'A1_gem4_run0.jsonl'; metric = C.A.metrics
    recs = C.load_last(f, lambda r: r['did'])
    over = 0; R = {'px': [], 'n1000': [], 'mixed': []}
    for did, r in recs.items():
        s = r['scale']; w, h = r['sent_size']; raw = [C.M.point_of(q) for q in (r.get('shapes') or [])]
        raw = [p for p in raw if p is not None]
        big = any(max(p) > 1000 for p in raw); over += big
        for mode in R:
            pts = []
            for x, y in raw:
                if mode == 'n1000' or (mode == 'mixed' and not big): x, y = x / 1000 * w, y / 1000 * h
                pts.append((x / s, y / s))
            m = metric(did, pts)
            if not np.isnan(m[0]): R[mode].append(m[1])
    out[bench] = {'answers': len(recs), 'with_coord_over_1000': over, **{k: float(np.mean(v)) for k, v in R.items()}}
print(json.dumps(out, indent=1))
json.dump(out, open('../gem4_rerun_logs/A1_pixel_prompt_readings.json', 'w'), indent=1)
