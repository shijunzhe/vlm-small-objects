#!/usr/bin/env python3
"""Theorem 1 checked exactly as stated: hit recall over targets interior to the whole-image call and to some tile
(box inside the tile's view, centre at least tau inside its kept core), on items where every tile has S >= S_whole.
Fixed-grid S follows the paper (S = m g / long side of the view). Bootstrap clustered by source drawing
(FPC-300 and FPC-sheets) or by item. Also the same contrast on items where some tile has S < S_whole.
Writes results_law/theory_check_thm1.json."""
import json, glob, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np
import mech as M, ceilings as K
import law_data as LD

M.FORMAT_AWARE = True
rng = np.random.default_rng(7)
FIXED_G = {'gemini25': math.sqrt(258), 'gemma4': math.sqrt(280), 'gem38': math.sqrt(1090), 'gem38lo': math.sqrt(257), 'gem38uh': math.sqrt(2210)}


def S_view(v, view_w_native, view_h_native, sent_w, sent_h, m):
    """Tokens per target side for a view of the given native size sent at the given pixel size."""
    if v in FIXED_G: return m * FIXED_G[v] / max(view_w_native, view_h_native)
    return LD.S_of(v, sent_w, sent_h, m * sent_w / view_w_native)


def a1_points(files, decode):
    """{did: [points per run]} for whole-drawing calls (mirrors analyze_test.score_whole)."""
    out = defaultdict(list); meta = {}
    for f in files:
        last = {}
        for l in open(f):
            r = json.loads(l)
            if r['status'] in ('success', 'parse_error'): last[r['did']] = r
        for d, r in last.items():
            s = r['scale']; w, h = r['sent_size']; pts = []
            for q in (r.get('shapes') or []) if r['status'] == 'success' else []:
                pt = M.point_of(q)
                if pt is None: continue
                x, y = pt
                if decode == 'n1000': x, y = x / 1000 * w, y / 1000 * h
                elif decode == 'px_claude': rw, rh = M.claude_resized(w, h); x, y = x * w / rw, y * h / rh
                pts.append((x / s, y / s))
            out[d].append(pts); meta[d] = (s, w, h)
    return out, meta


def d_points(files):
    out = defaultdict(list); geom = {}
    for k, f in enumerate(files):
        last = {}
        for l in open(f):
            r = json.loads(l); last[(r['did'], r['i'], r['j'])] = r
        bydid = defaultdict(list)
        for (d, i, j), r in last.items(): bydid[d].append(r)
        for d, rs in bydid.items():
            if any(r['status'] != 'success' for r in rs): continue
            pts = []
            for r in rs: pts += M.to_native(r)
            out[d].append(pts)
            if d not in geom:
                g = []
                for r in rs:
                    zu = r['z0'] * r['up']; w, h = r['tile_size']; ox, oy = r['origin_z0']; c0, c1, c2, c3 = r['core_px']
                    g.append({'core': ((ox + c0) / zu, (oy + c1) / zu, (ox + c2) / zu, (oy + c3) / zu),
                              'view': (ox / zu, oy / zu, (ox + w) / zu, (oy + h) / zu), 'sent': (w, h)})
                geom[d] = g
    return out, geom


def inside(b, box): return box[0] <= b[0] and box[1] <= b[1] and b[2] <= box[2] and b[3] <= box[3]


def interior_targets(a, tiles, W, H, exclude=lambda p: False):
    w, h = a['template_size']; m = math.sqrt(w * h); tau = m / 2; out = []
    for p in a['points']:
        x, y = p
        if exclude(p): continue
        bx = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
        if not inside(bx, (0, 0, W, H)) or min(x, y, W - x, H - y) < tau: continue
        if any(inside(bx, t['view']) and min(x - t['core'][0], y - t['core'][1], t['core'][2] - x, t['core'][3] - y) >= tau for t in tiles):
            out.append((x, y))
    return out, tau


def hit(pts, targets, tau):
    if not targets: return np.nan
    P = np.array(pts, float).reshape(-1, 2)
    if len(P) == 0: return 0.0
    return float(np.mean([np.min(np.hypot(P[:, 0] - x, P[:, 1] - y)) <= tau for x, y in targets]))


def contrast(A, Dd, Dg, ann_of, v, items, exclude_of=lambda d: (lambda p: False)):
    rows = []
    for d in items:
        if d not in A or d not in Dd or d not in Dg: continue
        a = ann_of(d); W, H = a['width'], a['height']; m = math.sqrt(a['template_size'][0] * a['template_size'][1])
        tg, tau = interior_targets(a, Dg[d], W, H, exclude_of(d))
        if not tg: continue
        s_w = S_view(v, W, H, A[d][1][1], A[d][1][2], m)
        s_t = min(S_view(v, t['view'][2] - t['view'][0], t['view'][3] - t['view'][1], t['sent'][0], t['sent'][1], m) for t in Dg[d])
        ha = np.mean([hit(p, tg, tau) for p in A[d][0]]); hd = np.mean([hit(p, tg, tau) for p in Dd[d]])
        rows.append({'item': d, 'cond': s_t >= s_w - 1e-9, 'diff': hd - ha, 'n_targets': len(tg), 'S_w': s_w, 'S_t': s_t})
    return rows


def boot(rows, clus):
    x = np.array([r['diff'] for r in rows]); g = defaultdict(list)
    for r, c in zip(rows, clus): g[c].append(r['diff'])
    keys = list(g); bs = []
    for _ in range(4000):
        sel = [keys[i] for i in rng.integers(0, len(keys), len(keys))]; bs.append(np.mean([v for k in sel for v in g[k]]))
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(x)


RES = []
def report(ds, v, rows, clusfun):
    for cond, fam in ((True, 'Thm1'), (False, 'outside')):
        rs = [r for r in rows if r['cond'] == cond]
        if len(rs) < 5: continue
        m, lo, hi, n = boot(rs, [clusfun(r['item']) for r in rs])
        RES.append({'family': fam, 'label': f'{ds} {v}', 'diff': m, 'lo': lo, 'hi': hi, 'n': n,
                    'targets': int(sum(r['n_targets'] for r in rs)),
                    'verdict': ('n/a' if fam == 'outside' else ('violated' if hi < 0 else ('supported' if lo > 0 else 'consistent')))})
        print(f'{fam:8s} {ds:10s} {v:9s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] n={n} targets={RES[-1]["targets"]}', flush=True)


# ---- REDP-X40 ----
import analyze_test as A
GEN1 = {'gpt': ('gpt54', 'px'), 'claude': ('sonnet46', 'px_claude'), 'gemini': ('gemini25', 'n1000'), 'gemma4': ('gemma4', 'n1000'), 'qwen3vl': ('qwen3vl', 'n1000')}
GEN2 = {'gpt56': ('gpt56', 'px'), 'sonnet55': ('sonnet55', 'px'), 'gem38': ('gem38', 'n1000')}
for root, table in ((Path('results_test'), GEN1), (Path('results_gen'), GEN2)):
    for k, (v, dec) in table.items():
        Ap, Am = a1_points(sorted(glob.glob(str(root / f'A1_{k}_run*.jsonl'))), dec)
        Dp, Dg = d_points(sorted(glob.glob(str(root / f'D_{k}_run*.jsonl'))))
        Ax = {d: (Ap[d], Am[d]) for d in Ap}
        report('REDP-X40', v, contrast(Ax, Dp, Dg, lambda d: A.ANN[d], v, K.TEST), lambda d: d)

# ---- FloorPlanCAD ----
import fpc_score as F
KEY = {'gpt': 'gpt54', 'claude': 'sonnet46', 'gemini': 'gemini25'}
for b in ('bench', 'sheets'):
    items = F.TEST if b == 'bench' else F.SHEETS
    clus = (lambda d: d.split('-')[0]) if b == 'bench' else (lambda d: d.split('_')[0])
    for k in ('gpt', 'claude', 'gemini', 'gemma4', 'qwen3vl', 'gpt56', 'sonnet55', 'gem38'):
        v = KEY.get(k, k)
        Ap, Am = a1_points([F.RF / b / f'A1_{k}_run0.jsonl'], F.DEC[k])
        Dp, Dg = d_points([F.RF / b / f'D_{k}_run0.jsonl'])
        Ax = {d: (Ap[d], Am[d]) for d in Ap}
        def excl(d):
            a = F.ANN[d]; tau = M.taus(a)[0]
            return lambda p: F.near_edge(p[0], p[1], a['ignore_rects'], tau)
        report('FPC-300' if b == 'bench' else 'FPC-sheets', v, contrast(Ax, Dp, Dg, lambda d: F.ANN[d], v, items, excl), clus)

json.dump(RES, open('results_law/theory_check_thm1.json', 'w'), indent=1)
