#!/usr/bin/env python3
"""Zero-call check of why the Gemini 3.8 and Qwen3-VL zoom agents fail (Supp J): from the stored trajectories only.
Per LegendSpot test drawing: count ratio (returned points / instances), matched F1 under the reading used in the paper,
F1 under the other coordinate reading (overview pixels vs 0-1000 of the overview), and the distance from each returned
point to its nearest instance in units of tau. Also the share of answers whose points all lie inside the overview.
usage (from the release's experiments/ directory): python3 agent_err_check.py  -> writes results_x/agent_err_check.json"""
import json, math, os, sys
sys.path.insert(0, os.getcwd())
from pathlib import Path
import numpy as np
import mech as M, ceilings as K
import analyze_test as A

M.FORMAT_AWARE = True
XR = Path('results_x'); T = K.TEST
PAPER = {'gpt56': 'px', 'sonnet55': 'px', 'gem38': 'px', 'qwen3vl': 'n1000'}


from x4q_score import raw_points as _qwen_points   # Qwen reader of the paper (adds '"cx": [x, "cy": y]' points)


def pts_of(r):
    if r.get('vendor') == 'qwen3vl':
        return _qwen_points(r)
    if r['status'] == 'success':
        return [p for p in (M.point_of(q) for q in r.get('shapes', [])) if p is not None]
    return M.lenient_points(r.get('raw'))


def native(r, reading):
    s = r['scale']; ow, oh = r['overview']; out = []
    for x, y in pts_of(r):
        if reading == 'n1000': x, y = x / 1000 * ow, y / 1000 * oh
        out.append((x / s, y / s))
    return out


def nearest_tau(did, pts):
    a = A.ANN[did]; tau, _ = M.taus(a); G = np.asarray(a['points'], float)
    if not pts or not len(G): return []
    P = np.asarray(pts, float); D = np.linalg.norm(P[:, None] - G[None], axis=2).min(1)
    return list(D / tau)


res = {}
for v, rd in PAPER.items():
    f = XR / 'agent' / f'{v}_run0.jsonl'
    rows = []
    for l in open(f):
        r = json.loads(l)
        if r['did'] not in T or r['status'] not in ('success', 'parse_error'): continue
        g = A.ANN[r['did']]['count']; ow, oh = r['overview']
        raw = pts_of(r)
        p_paper = native(r, rd); p_other = native(r, 'n1000' if rd == 'px' else 'px')
        m1, m2 = A.metrics(r['did'], p_paper), A.metrics(r['did'], p_other)
        d = nearest_tau(r['did'], p_paper)
        inside_px = all(0 <= x <= ow and 0 <= y <= oh for x, y in raw)
        rows.append(dict(did=r['did'], n=g, ret=len(raw), count_ratio=len(raw) / g, f1=float(m1[0]), f1_other=float(m2[0]),
                         prec=float(m1[2]), med_dist_tau=float(np.median(d)) if d else None,
                         within_2tau=float(np.mean(np.array(d) <= 2)) if d else None, inside_overview_px=inside_px))
    cr = np.array([x['count_ratio'] for x in rows])
    dd = [x['med_dist_tau'] for x in rows if x['med_dist_tau'] is not None]
    res[v] = dict(reading=rd, n=len(rows), f1=float(np.mean([x['f1'] for x in rows])), f1_other_reading=float(np.mean([x['f1_other'] for x in rows])),
                  count_within_10pct=float(np.mean(np.abs(cr - 1) <= 0.10)), median_count_ratio=float(np.median(cr)),
                  median_dist_tau=float(np.median(dd)) if dd else None,
                  share_points_within_tau=float(np.mean([x['prec'] for x in rows])),
                  share_points_within_2tau=float(np.mean([x['within_2tau'] for x in rows if x['within_2tau'] is not None])),
                  share_answers_inside_overview_px=float(np.mean([x['inside_overview_px'] for x in rows])), rows=rows)
    print(v, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in res[v].items() if k != 'rows'})
# reference: the same model's tiles (44 px recipe or its fixed-grid counterpart), whose points the caller maps back
for v, rdir in (('gem38', 'results_gen'), ('qwen3vl', 'results_test')):
    A.R = Path(rdir); mp = A.mean_per(A.score_tiles('D', v)); ds = [d for d in T if d in mp]
    res[v]['tiles'] = dict(n=len(ds), f1=float(np.mean([mp[d][0] for d in ds])), share_points_within_tau=float(np.mean([mp[d][2] for d in ds])),
                           median_count_ratio=float(np.median([mp[d][4] for d in ds])),
                           count_within_10pct=float(np.mean([abs(mp[d][4] - 1) <= 0.10 for d in ds])))
    print(v, 'tiles', {k: round(x, 3) if isinstance(x, float) else x for k, x in res[v]['tiles'].items()})
json.dump(res, open(XR / 'agent_err_check.json', 'w'), indent=1)
# macros for the paper (shares in percent)
pc = lambda x: f'{100 * x:.0f}'
G = Path(__file__).resolve().parent.parent / 'generated'
mac = {'CagLocGem': pc(res['gem38']['share_points_within_tau']), 'CagLocGemTwo': pc(res['gem38']['share_points_within_2tau']),
       'CagLocQwen': pc(res['qwen3vl']['share_points_within_tau']), 'CagLocQwenTwo': pc(res['qwen3vl']['share_points_within_2tau']),
       'CtileLocGem': pc(res['gem38']['tiles']['share_points_within_tau']), 'CtileLocQwen': pc(res['qwen3vl']['tiles']['share_points_within_tau']),
       'CagCntGem': pc(res['gem38']['count_within_10pct']), 'CtileCntGem': pc(res['gem38']['tiles']['count_within_10pct']),
       'CagCntQwen': pc(res['qwen3vl']['count_within_10pct']), 'CtileCntQwen': pc(res['qwen3vl']['tiles']['count_within_10pct']),
       'CagRatioQwen': f"{res['qwen3vl']['median_count_ratio']:.2f}", 'CtileRatioQwen': f"{res['qwen3vl']['tiles']['median_count_ratio']:.2f}",
       'CagDistGem': f"{res['gem38']['median_dist_tau']:.2f}", 'CagDistQwen': f"{res['qwen3vl']['median_dist_tau']:.2f}"}
open(G / 'macros_agentcheck.tex', 'w').write(''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in mac.items()))
print(mac)
