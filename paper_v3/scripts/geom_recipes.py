#!/usr/bin/env python3
"""Zero-call geometric check of common splitting recipes against the protection condition (2026-10-28).
For each image and interface: S of the standard whole-image call, and for each recipe whether some view zooms out
(S_view < S_whole), whether views reach S* = 1.5, whether adjacent views overlap by one target, the share of targets
that lie in no view's interior (cut by a boundary), and the tokens relative to the rule.
usage: python3 geom_recipes.py <release root>   -> generated/geom_recipes.json"""
import json, math, sys, glob
from pathlib import Path
ROOT = Path(sys.argv[1]); PAPER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools')); import safe_tiles as ST
IFACE = {'GPT-6.1': ('res', 32, 8192, None), 'Claude Opus 5.5': ('res', 28, 4784, 2576),
         'Qwen3-VL': ('res', 32, 2500, None), 'Gemini 3.8/4': ('grid', None, 1090, None)}
S_STAR = 1.5


def sent_scale(W, H, kind, t, N, cap):
    """effective scale of an image of W x H native px sent as standard input (long side <= 1600, then the vendor's resize)"""
    if kind == 'grid': return None
    s = min(1.0, 1600 / max(W, H)); w, h = W * s, H * s
    v = min(1.0, math.sqrt(N * t * t / (w * h)))
    if cap: v = min(v, cap / max(w, h))
    return s * v


def S_of(W, H, m, kind, t, N, cap, native=False):
    """S of a view of W x H native px; native=True: the view is sent at native resolution (SAHI), resized only by the vendor"""
    if kind == 'grid': return m * math.sqrt(N) / max(W, H)
    if native:
        v = min(1.0, math.sqrt(N * t * t / (W * H)))
        if cap: v = min(v, cap / max(W, H))
        return m * v / t
    return m * sent_scale(W, H, kind, t, N, cap) / t


def tokens_of(W, H, kind, t, N, cap, native=False):
    if kind == 'grid': return N
    s = (min(1.0, math.sqrt(N * t * t / (W * H))) if native else sent_scale(W, H, kind, t, N, cap))
    if native and cap: s = min(s, cap / max(W, H))
    return math.ceil(W * s / t) * math.ceil(H * s / t)


def grid_cuts(W, H, size, overlap):
    """SAHI-style slices: start positions with stride size - overlap, last slice flush with the border"""
    def starts(L):
        if L <= size: return [0]
        st = list(range(0, int(L - size) + 1, int(size - overlap)))
        if st[-1] + size < L: st.append(L - size)
        return st
    return [(x, y, min(W, x + size), min(H, y + size)) for y in starts(H) for x in starts(W)]


def interior(pt, box, w, h, tau):
    x, y = pt; x0, y0, x1, y1 = box
    return x - w / 2 >= x0 and x + w / 2 <= x1 and y - h / 2 >= y0 and y + h / 2 <= y1 and \
        x - tau >= x0 and x + tau <= x1 and y - tau >= y0 and y + tau <= y1


def items():
    out = []
    for f in sorted(glob.glob(str(ROOT / 'data/redp40/redp_x40/annotations/*.json'))):
        a = json.load(open(f)); out.append(('LegendSpot', a))
    for f in sorted(glob.glob(str(ROOT / 'data/fpc/sheets/annotations/*_k3.json'))):
        a = json.load(open(f)); out.append(('Floor-plan mosaics', a))
    for f in sorted(glob.glob(str(ROOT / 'data/fpc/bench/annotations/*.json'))):
        a = json.load(open(f)); out.append(('Floor-plan blocks', a))
    return out


def recipes(W, H):
    return {'SAHI 512, 20% overlap (native px)': (grid_cuts(W, H, 512, 0.2 * 512), True),
            'SAHI 1024, 20% overlap (native px)': (grid_cuts(W, H, 1024, 0.2 * 1024), True),
            '2x2 quadrants, no overlap': ([(i * W / 2, j * H / 2, (i + 1) * W / 2, (j + 1) * H / 2) for j in range(2) for i in range(2)], False),
            '4x4 cells, no overlap': ([(i * W / 4, j * H / 4, (i + 1) * W / 4, (j + 1) * H / 4) for j in range(4) for i in range(4)], False)}


res = {}
for ds, a in items():
    W, H = a['width'], a['height']; w, h = a['template_size']; m = math.sqrt(w * h); tau = m / 2; pts = a['points']
    pts = [p for p in pts if interior(p, (0, 0, W, H), w, h, tau)]   # targets interior to the whole-image call
    if not pts: continue
    for nm, (kind, t, N, cap) in IFACE.items():
        Sw = S_of(W, H, m, kind, t, N, cap)
        try:
            P = ST.plan(W, H, w, h, interface=kind, token=t or 32, budget=N, cap=cap, S_star=S_STAR); rule_tok = P['tokens']
        except ValueError:
            rule_tok = None
        for rn, (boxes, native) in recipes(W, H).items():
            Ss = [S_of(x1 - x0, y1 - y0, m, kind, t, N, cap, native=native) for x0, y0, x1, y1 in boxes]
            tok = sum(tokens_of(x1 - x0, y1 - y0, kind, t, N, cap, native=native) for x0, y0, x1, y1 in boxes)
            cut = sum(1 for p in pts if not any(interior(p, b, w, h, tau) for b in boxes)) / len(pts)
            r = res.setdefault(ds, {}).setdefault(nm, {}).setdefault(rn, {'n': 0, 'zoom_out': 0, 'below_Sstar': 0, 'cut_share': [], 'tok_ratio': [], 'views': []})
            r['n'] += 1; r['zoom_out'] += min(Ss) < Sw * 0.999; r['below_Sstar'] += min(Ss) < S_STAR * 0.999
            r['cut_share'].append(cut); r['views'].append(len(boxes))
            if rule_tok: r['tok_ratio'].append(tok / rule_tok)
out = {}
for ds, d in res.items():
    for nm, dd in d.items():
        for rn, r in dd.items():
            med = lambda v: sorted(v)[len(v) // 2] if v else None
            out.setdefault(ds, {}).setdefault(nm, {})[rn] = {
                'items': r['n'], 'share_zoom_out': r['zoom_out'] / r['n'], 'share_below_Sstar': r['below_Sstar'] / r['n'],
                'mean_share_targets_unprotected': sum(r['cut_share']) / r['n'], 'median_views': med(r['views']),
                'median_tokens_over_rule': med(r['tok_ratio'])}
json.dump(out, open(PAPER / 'generated/geom_recipes.json', 'w'), indent=1)
for ds, d in out.items():
    print('==', ds)
    for nm, dd in d.items():
        for rn, r in dd.items():
            print(f"  {nm:16s} {rn:36s} zoomout {r['share_zoom_out']:.2f}  belowS* {r['share_below_Sstar']:.2f}  unprotected {r['mean_share_targets_unprotected']:.3f}  views {r['median_views']}  tok/rule {r['median_tokens_over_rule']}")

# supplement table: per data set and recipe; targets cut do not depend on the interface
rows = []
for ds in ('LegendSpot', 'Floor-plan mosaics', 'Floor-plan blocks'):
    for rn in ('SAHI 512, 20% overlap (native px)', 'SAHI 1024, 20% overlap (native px)', '2x2 quadrants, no overlap', '4x4 cells, no overlap'):
        g, q, c, z = (out[ds][k][rn] for k in ('GPT-6.1', 'Qwen3-VL', 'Claude Opus 5.5', 'Gemini 3.8/4'))
        zo = max(x['share_zoom_out'] for x in (g, q, c, z))
        lab = {'SAHI 512, 20% overlap (native px)': 'SAHI, 512 px, 20\\%', 'SAHI 1024, 20% overlap (native px)': 'SAHI, 1{,}024 px, 20\\%',
               '2x2 quadrants, no overlap': '$2 \\times 2$, no overlap', '4x4 cells, no overlap': '$4 \\times 4$, no overlap'}[rn]
        rows.append(f"{ds if rn.startswith('SAHI 512') else ''} & {lab} & {100 * zo:.0f}\\% & {100 * g['share_below_Sstar']:.0f}\\% & "
                    f"{100 * g['mean_share_targets_unprotected']:.1f}\\% & {g['median_tokens_over_rule']:.2f} & {z['median_tokens_over_rule']:.2f} \\\\")
    rows.append('\\midrule')
open(PAPER / 'generated/tabG_recipes.tex', 'w').write('\n'.join(rows[:-1]) + '\n')
