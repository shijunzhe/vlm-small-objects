#!/usr/bin/env python3
"""Scoring for the expansion experiments on the 34 test drawings (prereg claude/expansion_prereg_v3.md):
X2 mosaic capacity (14 sheets), X3 reasoning mode, X4 zoom agent. Paired bootstrap over drawings (4000 resamples).
READING=robust (default) applies the label-free Claude 5.5 frame rule to mosaics that exceed its high-res tier;
READING=registered uses the fixed resized-frame decode."""
import json, math, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import mech as M, ceilings as K
import analyze_test as A
import mosaic as MO

M.FORMAT_AWARE = True
READING = os.environ.get('READING', 'robust')
GEN = Path('results_gen'); XR = Path('results_x')
T = K.TEST; SHEETS = [d for d in T if A.SUB[d] == 'sheet']; VIEWS = [d for d in T if A.SUB[d] == 'view']
DEC = {'gpt56': 'px', 'sonnet55': 'px', 'gem38': 'n1000', 'gpt56r': 'px', 'sonnet55t': 'px', 'gem38t': 'n1000'}
OUT = {}


def at(fn, *a):
    old = A.R; A.R = GEN
    try:
        return fn(*a)
    finally:
        A.R = old


def f1(per, ds):
    s = A.summary(per, ds)[0]; return s if s is not None else np.full(5, np.nan)


def diff(pa, pb, ds, k=0):
    d = A.boot_diff(pa, pb, ds, k=k); return d if d else (np.nan, np.nan, np.nan)


def fmt(d): return f'{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]'


# ---------------- X2 mosaic ----------------
_MOS = {}


def mosaic_image(did, k, g):
    if (did, k) not in _MOS:
        _MOS.clear()
        ann, img, tpl = M.load(did); ts, z = MO.x2_tiles(ann, img, 1024); del img
        _MOS[(did, k)] = {gg: can.convert('L') for gg, can, cells in MO.mosaics(ts, k)}
    return _MOS[(did, k)][g]


def mosaic_to_native(rec):
    pts = []; W, H = rec['size']
    raw = [pt for pt in (M.point_of(q) for q in rec.get('shapes', [])) if pt is not None]
    if rec['vendor'].startswith('sonnet55'):
        if READING == 'robust':
            rw, rh = M.claude_resized(W, H, 2576, 4784)
            fx, fy = (1.0, 1.0) if (rw, rh) == (W, H) else M.claude55_scale_ink(raw, mosaic_image(rec['did'], rec['k'], rec['g']), 44)
        else:
            rw, rh = M.claude_resized(W, H, 2576, 4784); fx, fy = W / rw, H / rh
    else:
        fx = fy = 1.0
    for x, y in raw:
        x, y = x * fx, y * fy
        for cx, cy, core, origin, z0, size in rec['cells']:
            lx, ly = x - cx, y - cy
            if 0 <= lx < size[0] and 0 <= ly < size[1]:
                a, b, c, e = core
                if a <= lx < c and b <= ly < e:
                    pts.append(((origin[0] + lx) / z0, (origin[1] + ly) / z0))
                break
    return pts


def score_mosaic(v, k):
    f = XR / 'mosaic' / f'{v}_k{k}_run0.jsonl'
    if not f.exists(): return None
    last = {}
    for l in open(f):
        r = json.loads(l)
        if r['status'] in ('success', 'parse_error'): last[(r['did'], r['g'])] = r
    pts = defaultdict(list); per = defaultdict(list)
    for (d, g), r in sorted(last.items()):
        pts[d] += mosaic_to_native(r) if r['status'] == 'success' else []
    for d in {d for d, g in last}:
        per[d].append(A.metrics(d, pts[d]))
    return per


def x2():
    print('\n=== X2 mosaic capacity (14 sheets): F1 / recall / precision; diff vs k=1 (paired)')
    res = {}
    for v in ('gpt56', 'sonnet55'):
        base = at(A.score_tiles, 'D', v); res[(v, 1)] = base
        b = f1(base, MO.SHEETS); print(f'{v:9s} k=1  F1 {b[0]:.2f} R {b[1]:.2f} P {b[2]:.2f}')
        for k in (2, 4, 8):
            per = score_mosaic(v, k)
            if per is None: continue
            res[(v, k)] = per; s = f1(per, MO.SHEETS); ds = [d for d in MO.SHEETS if d in per]
            print(f'{v:9s} k={k}  F1 {s[0]:.2f} R {s[1]:.2f} P {s[2]:.2f}  dF1 {fmt(diff(per, base, ds))}  dRecall {fmt(diff(per, base, ds, 1))}  (n={len(ds)})')
            OUT[f'X2|{v}|k{k}'] = {'f1': s[0], 'recall': s[1], 'precision': s[2], 'dF1': diff(per, base, ds), 'dR': diff(per, base, ds, 1), 'n': len(ds)}
        OUT[f'X2|{v}|k1'] = {'f1': b[0], 'recall': b[1], 'precision': b[2]}
    return res


# ---------------- X3 reasoning ----------------
def tok(v, cond='A1'):
    o = []; i = []
    for f in (GEN / f'{cond}_{v}_run0.jsonl',):
        if not f.exists(): continue
        for l in open(f):
            r = json.loads(l); u = r.get('usage') or {}
            if r['status'] == 'success': o.append((u.get('output_tokens') or 0) + (u.get('thinking_tokens') or 0)); i.append(u.get('input_tokens') or 0)
    return (np.mean(i) if i else np.nan, np.mean(o) if o else np.nan, len(o))


def x3():
    print('\n=== X3 reasoning mode, whole drawing (A1). F1 all / sheets / views; reasoning - plain; reasoning - D')
    for plain, rsn in (('gpt56', 'gpt56r'), ('sonnet55', 'sonnet55t'), ('gem38', 'gem38t')):
        pa = at(A.score_whole, plain, DEC[plain]); pr = at(A.score_whole, rsn, DEC[rsn]); pd = at(A.score_tiles, 'D', plain)
        if not A.mean_per(pr): continue
        n = len(A.mean_per(pr))
        line = ' / '.join(f'{f1(pr, ds)[0]:.2f}' for ds in (T, SHEETS, VIEWS))
        lp = ' / '.join(f'{f1(pa, ds)[0]:.2f}' for ds in (T, SHEETS, VIEWS))
        ti, to, _ = tok(rsn); pi, po, _ = tok(plain)
        print(f'{rsn:10s} (n={n}) reasoning {line} | plain {lp} | D {f1(pd, T)[0]:.2f}/{f1(pd, SHEETS)[0]:.2f}')
        print(f'           rsn-plain all {fmt(diff(pr, pa, T))} sheets {fmt(diff(pr, pa, SHEETS))} | rsn-D all {fmt(diff(pr, pd, T))} sheets {fmt(diff(pr, pd, SHEETS))} | out tokens {to:.0f} vs {po:.0f}')
        OUT[f'X3|{rsn}'] = {'f1': [f1(pr, ds)[0] for ds in (T, SHEETS, VIEWS)], 'plain': [f1(pa, ds)[0] for ds in (T, SHEETS, VIEWS)],
                            'D': [f1(pd, ds)[0] for ds in (T, SHEETS, VIEWS)], 'rsn_minus_plain': [diff(pr, pa, T), diff(pr, pa, SHEETS)],
                            'rsn_minus_D': [diff(pr, pd, T), diff(pr, pd, SHEETS)], 'out_tokens': [to, po], 'n': n}


# ---------------- X4 agent ----------------
SIDE = {'gpt56': 32, 'sonnet55': 28}


def agent_per(v, tag='run0'):
    per = defaultdict(list); logs = {}
    f = XR / 'agent' / f'{v}_{tag}.jsonl'
    if not f.exists(): return per, logs
    for l in open(f):
        r = json.loads(l)
        if r['status'] not in ('success', 'parse_error'): continue
        s = r['scale']; pts = [(p[0] / s, p[1] / s) for p in (M.point_of(q) for q in r.get('shapes', [])) if p is not None] if r['status'] == 'success' else []
        per[r['did']] = [A.metrics(r['did'], pts)]; logs[r['did']] = r['log'] | {'scale': s}
    return per, logs


def coverage(did, log, v, S0=1.0):
    """Fraction of the drawing area seen at >= S0 tokens per symbol, across the overview and all zoom views."""
    a = A.ANN[did]; W, H = a['width'], a['height']; m = math.sqrt(a['template_size'][0] * a['template_size'][1])
    t = SIDE.get(v, None)
    g = 8; mask = np.zeros((math.ceil(H / g), math.ceil(W / g)), bool)
    if t is None:   # fixed grid (Gemini 3.8, ~1090 tokens per image): token side = sqrt(area/1090) in view pixels
        def S_of(view_w, view_h, native_w, native_h): return m * (view_w / native_w) / math.sqrt(view_w * view_h / 1090)
    else:
        def S_of(view_w, view_h, native_w, native_h): return m * (view_w / native_w) / t
    if S_of(*[round(W * log['scale']), round(H * log['scale'])], W, H) >= S0: mask[:] = True
    for z in log['zooms']:
        x0, y0, x1, y1 = z['native_box']
        if S_of(z['shown'][0], z['shown'][1], x1 - x0, y1 - y0) >= S0:
            mask[int(y0 // g):int(math.ceil(y1 / g)), int(x0 // g):int(math.ceil(x1 / g))] = True
    return mask.mean()


def x4():
    from scipy.stats import spearmanr
    print('\n=== X4 zoom agent. F1 all / sheets / views; agent - A1; agent - D; zooms, tokens, coverage')
    for v in ('gpt56', 'sonnet55', 'gem38'):
        per, logs = agent_per(v)
        if not per: continue
        ds = sorted(per); pa = at(A.score_whole, v, DEC[v]); pd = at(A.score_tiles, 'D', v)
        n = len(ds); sh = [d for d in SHEETS if d in per]; vw = [d for d in VIEWS if d in per]
        cov = {d: coverage(d, logs[d], v) for d in ds}
        rec = {d: A.mean_per(per)[d][1] for d in ds}
        rho = spearmanr([cov[d] for d in ds], [rec[d] for d in ds]) if n > 4 else (np.nan, np.nan)
        rhoF = spearmanr([cov[d] for d in ds], [A.mean_per(per)[d][0] for d in ds]) if n > 4 else (np.nan, np.nan)
        zooms = np.mean([logs[d]['zooms_used'] for d in ds]); tin = np.mean([logs[d]['in'] for d in ds]); tout = np.mean([logs[d]['out'] for d in ds])
        print(f'{v:9s} (n={n}) agent {f1(per, ds)[0]:.2f}/{f1(per, sh)[0]:.2f}/{f1(per, vw)[0]:.2f}  A1 {f1(pa, ds)[0]:.2f}  D {f1(pd, ds)[0]:.2f}')
        print(f'          agent-A1 {fmt(diff(per, pa, ds))} | agent-D all {fmt(diff(per, pd, ds))} sheets {fmt(diff(per, pd, sh))}')
        print(f'          zooms {zooms:.1f}  tokens in {tin:.0f} out {tout:.0f}  coverage(S>=1) mean {np.mean(list(cov.values())):.2f} sheets {np.mean([cov[d] for d in sh]) if sh else float("nan"):.2f}  rho(coverage, recall) {rho[0]:+.2f} (p={rho[1]:.3f})')
        OUT.setdefault('agent_count', {})[v] = float(np.mean([A.mean_per(per)[d][3] < 1e-9 for d in ds]))
        OUT[f'X4|{v}'] = {'n': n, 'f1': [f1(per, x)[0] for x in (ds, sh, vw)], 'A1': f1(pa, ds)[0], 'D': f1(pd, ds)[0],
                          'agent_minus_A1': diff(per, pa, ds), 'agent_minus_D': [diff(per, pd, ds), diff(per, pd, sh)],
                          'zooms': zooms, 'tok_in': tin, 'tok_out': tout, 'coverage': np.mean(list(cov.values())),
                          'coverage_sheets': np.mean([cov[d] for d in sh]) if sh else None, 'rho_cov_recall': [rho[0], rho[1]], 'rho_cov_f1': [rhoF[0], rhoF[1]]}


def x4b():
    print('\n=== X4b zoom without new information (crop of the overview) and X4c run-to-run stability')
    for v in ('sonnet55', 'gpt56'):
        nat, ln = agent_per(v); ov, lo = agent_per(v, 'ovzoom_run0'); r1, l1 = agent_per(v, 'run1')
        ds = sorted(set(A.mean_per(nat)) & set(A.mean_per(ov)))
        if ds:
            sh = [d for d in ds if A.SUB[d] == 'sheet']; vw = [d for d in ds if A.SUB[d] == 'view']
            # overview pixels per symbol: m * s (information available in the overview)
            mpx = {d: min(A.ANN[d]['template_size']) * ln[d]['scale'] for d in ds}
            gap = {d: A.mean_per(nat)[d][0] - A.mean_per(ov)[d][0] for d in ds}
            small = [d for d in ds if mpx[d] < 12]; large = [d for d in ds if mpx[d] >= 12]
            print(f'{v:9s} (n={len(ds)}) native {f1(nat, ds)[0]:.2f} overview-crop {f1(ov, ds)[0]:.2f}  diff all {fmt(diff(nat, ov, ds))}  sheets {fmt(diff(nat, ov, sh))}  views {fmt(diff(nat, ov, vw))}')
            print(f'          symbol < 12 px in overview (n={len(small)}): {fmt(diff(nat, ov, small)) if small else "-"} | >= 12 px (n={len(large)}): {fmt(diff(nat, ov, large)) if large else "-"}')
            OUT[f'X4b|{v}'] = {'n': len(ds), 'native': f1(nat, ds)[0], 'ov': f1(ov, ds)[0], 'diff': [diff(nat, ov, ds), diff(nat, ov, sh), diff(nat, ov, vw)],
                               'n_sheets': len(sh), 'small': [len(small), diff(nat, ov, small) if small else None], 'large': [len(large), diff(nat, ov, large) if large else None],
                               'zooms_ov': np.mean([lo[d]['zooms_used'] for d in ds])}
            OUT[f'X4bsheets|{v}'] = {'ov': f1(ov, sh)[0], 'native': f1(nat, sh)[0], 'n': len(sh)}
        d1 = sorted(set(A.mean_per(nat)) & set(A.mean_per(r1)))
        if d1:
            a = A.mean_per(nat); b = A.mean_per(r1)
            print(f'          run0 {np.mean([a[d][0] for d in d1]):.2f} run1 {np.mean([b[d][0] for d in d1]):.2f} (n={len(d1)}), mean |diff| per drawing {np.mean([abs(a[d][0] - b[d][0]) for d in d1]):.3f}, diff {fmt(diff(r1, nat, d1))}')
            OUT[f'X4c|{v}'] = {'n': len(d1), 'run0': np.mean([a[d][0] for d in d1]), 'run1': np.mean([b[d][0] for d in d1]), 'mad': np.mean([abs(a[d][0] - b[d][0]) for d in d1]), 'diff': diff(r1, nat, d1)}


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'x2'): x2()
    if which in ('all', 'x3'): x3()
    if which in ('all', 'x4'):
        x4(); x4b()
        dev, loo, thr = A.tm_scores(); OUT['TMsheets'] = [f1(dev, SHEETS)[0], f1(loo, SHEETS)[0]]
    if which != 'all':
        old = XR / f'x_new_scores_{READING}.json'
        if old.exists(): OUT = {**json.load(open(old)), **OUT}
    json.dump(OUT, open(XR / f'x_new_scores_{READING}.json', 'w'), indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else float(o))
