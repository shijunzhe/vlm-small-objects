#!/usr/bin/env python3
"""Open-model zoom agent (Qwen3-VL via OpenRouter, provider Alibaba) on the 34 REDP-X40 test drawings.
Qwen3-VL reports points on a 0 to 1000 scale of the image it was shown (documented convention); the agent's final
answer refers to the overview. Reading 'n1000' (documented, primary) and 'px' (overview pixels, sensitivity).
Malformed JSON is read leniently, including Qwen's '"cx": [x, "cy": y]' pattern. Compared with the same model's
round-1 whole-drawing call (A2, 0-1000) and tiles (D). Writes results_x/x4q_scores.json."""
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import mech as M, ceilings as K
import analyze_test as A

M.FORMAT_AWARE = True
XR = Path('results_x')
T = K.TEST; SHEETS = [d for d in T if A.SUB[d] == 'sheet']
_BRK = re.compile(r'"cx"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*"cy"\s*:\s*(-?\d+(?:\.\d+)?)')


def raw_points(r):
    if r['status'] == 'success':
        pts = [p for p in (M.point_of(q) for q in r.get('shapes', [])) if p is not None]
        # shapes may silently drop malformed entries: add any bracket-pattern points from raw
        return pts + [(float(a), float(b)) for a, b in _BRK.findall(r.get('raw') or '')]
    return M.lenient_points(r.get('raw')) + [(float(a), float(b)) for a, b in _BRK.findall(r.get('raw') or '')]


def agent_per(reading, tag='run0'):
    per = {}; logs = {}
    for l in open(XR / 'agent' / f'qwen3vl_{tag}.jsonl'):
        r = json.loads(l)
        if r['status'] not in ('success', 'parse_error'): continue
        s = r['scale']; ow, oh = r['overview']; pts = []
        for x, y in raw_points(r):
            if reading == 'n1000': x, y = x / 1000 * ow, y / 1000 * oh
            pts.append((x / s, y / s))
        per[r['did']] = [A.metrics(r['did'], pts)]; logs[r['did']] = r['log']
    return per, logs


def f1(per, ds):
    s = A.summary(per, ds)[0]; return s if s is not None else np.full(5, np.nan)


def main():
    out = {}
    pd = A.score_tiles('D', 'qwen3vl'); pa = A.score_whole('qwen3vl', 'n1000')
    for reading in ('n1000', 'px'):
        per, logs = agent_per(reading)
        ds = [d for d in T if d in per]; sh = [d for d in SHEETS if d in per]
        r = {'n': len(ds), 'n_sheets': len(sh), 'f1': [f1(per, ds)[0], f1(per, sh)[0]],
             'D': [f1(pd, ds)[0], f1(pd, sh)[0]], 'A2': [f1(pa, ds)[0], f1(pa, sh)[0]],
             'ag_minus_D': [A.boot_diff(per, pd, ds), A.boot_diff(per, pd, sh)],
             'ag_minus_A2': [A.boot_diff(per, pa, ds), A.boot_diff(per, pa, sh)],
             'exact_count': [float(np.mean([A.mean_per(per)[d][3] < 1e-9 for d in ds])), float(np.mean([A.mean_per(pd)[d][3] < 1e-9 for d in ds]))],
             'count_err': [float(np.median([A.mean_per(per)[d][3] for d in ds])), float(np.median([A.mean_per(pd)[d][3] for d in ds]))],
             'zooms': float(np.mean([logs[d]['zooms_used'] for d in ds])),
             'tok_in': float(np.mean([logs[d]['in'] for d in ds])), 'tok_out': float(np.mean([logs[d]['out'] for d in ds]))}
        out[reading] = r
        print(reading, json.dumps(r, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))[:900])
    json.dump(out, open(XR / 'x4q_scores.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))


if __name__ == '__main__':
    main()
