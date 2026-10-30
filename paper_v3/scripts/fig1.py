#!/usr/bin/env python3
"""Figure 1: S too small / L too large / splitting can zoom out / the fix.
Schematic interface for (a), (c), (d): 32 px tokens, 2,500 tokens per image. (b) is a schematic of packing;
its numbers are the GPT-5.6 packing measurements (XmosR*). Run from p2release/experiments."""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg'); matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path as _P
_PAPER = str(_P(__file__).resolve().parent.parent); _UNI = str(_P.cwd().parent)   # run from the release's experiments/
sys.path.insert(0, _UNI + '/tools'); import safe_tiles as ST
Image.MAX_IMAGE_PIXELS = None
B = Path(_UNI + '/data/redp40/redp_x40'); OUT = Path(_PAPER + '/figures')
T, N = 32, 2500
def load(d):
    a = json.load(open(B / 'annotations' / f'{d}.json')); im = Image.open(B / 'images' / f'{d}.png').convert('L'); return a, im
def received(img, cx, cy, m, S, half=2.0):
    s = S * T / m; r = half * m
    crop = img.crop((round(cx - r), round(cy - r), round(cx + r), round(cy + r)))
    k = max(1, round(crop.width * s)); return np.asarray(crop.resize((k, k), Image.LANCZOS)), s
def show_received(ax, a, s, m, title):
    big = np.kron(a, np.ones((8, 8), dtype=a.dtype))   # each received pixel as an 8 x 8 block, so the PDF shows the pixels as sent
    ax.imshow(big, cmap='gray', vmin=0, vmax=255, interpolation='none', extent=(0, a.shape[1], a.shape[0], 0))
    for g in np.arange(0, a.shape[1] + 1e-9, T):
        ax.axvline(g, color='#2a78d6', lw=0.5, alpha=0.8); ax.axhline(g, color='#2a78d6', lw=0.5, alpha=0.8)
    c = a.shape[0] / 2; side = m * s
    ax.add_patch(Rectangle((c - side / 2, c - side / 2), side, side, fill=False, ec='#e8710a', lw=1.0))
    ax.set_xlim(0, a.shape[1]); ax.set_ylim(a.shape[0], 0); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=6.8)
# big sheet d018
A, IM = load('d018'); W, H = IM.size; w, h = A['template_size']; m = math.sqrt(w * h)
P = ST.plan(W, H, w, h, interface='res', token=T, budget=N, S_star=1.5)
pts = np.array(A['points']); cx, cy = pts[0]
vid = next(v for v in P['views'] if v['box'][0] <= cx < v['box'][2] and v['box'][1] <= cy < v['box'][3])
n_view = int(sum((vid['box'][0] <= x < vid['box'][2]) and (vid['box'][1] <= y < vid['box'][3]) for x, y in pts))
# small sheet d025
A2, IM2 = load('d025'); w2, h2 = A2['template_size']; m2 = math.sqrt(w2 * h2); c2 = A2['points'][0]
S2w = m2 * (1600 / max(IM2.size)) / T; S2t = 44 * m2 / (min(w2, h2) * T)
plt.rcParams.update({'font.size': 7, 'font.family': 'serif'})
fig = plt.figure(figsize=(6.9, 1.9))
gs = fig.add_gridspec(1, 6, width_ratios=[1, 1.35, 1, 1, 0.12, 1.5], wspace=0.18)
ax = fig.add_subplot(gs[0]); a, s = received(IM, cx, cy, m, P['S_whole'])
show_received(ax, a, s, m, f'(a) $S$ too small:\nwhole, $S={P["S_whole"]:.2f}$')
# (b) packing schematic: 8 same-scale tiles, each around one target, packed into one image
ax = fig.add_subplot(gs[1]); z = 0.5; tile = 1024; tz = round(tile * z)
canvas = Image.new('L', (4 * tz + 3 * 6, 2 * tz + 6), 255)
for k, (x, y) in enumerate(pts[:8]):
    i, j = k % 4, k // 4
    cr = IM.crop((round(x - tile / 2), round(y - tile / 2), round(x + tile / 2), round(y + tile / 2))).resize((tz, tz), Image.LANCZOS)
    canvas.paste(cr, (i * (tz + 6), j * (tz + 6)))
ax.imshow(np.asarray(canvas), cmap='gray', vmin=0, vmax=255, interpolation='none')
for k in range(8):
    i, j = k % 4, k // 4
    ax.add_patch(Rectangle((i * (tz + 6), j * (tz + 6)), tz, tz, fill=False, ec='#e8710a' if k == 0 else '#888888', lw=1.0 if k == 0 else 0.5))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('(b) $L$ too large:\n8 tiles in one call', fontsize=6.8)
# (c) zoom-out on the small sheet
ax = fig.add_subplot(gs[2]); a, s = received(IM2, c2[0], c2[1], m2, S2w, half=1.2)
show_received(ax, a, s, m2, f'(c) Small sheet:\nwhole, $S={S2w:.2f}$')
ax = fig.add_subplot(gs[3]); a, s = received(IM2, c2[0], c2[1], m2, S2t, half=1.2)
show_received(ax, a, s, m2, f'44 px tiles zoom out:\n$S={S2t:.2f}$')
# (d) the fix
ax = fig.add_subplot(gs[5]); thb = IM.copy(); thb.thumbnail((1400, 1400)); f = thb.width / W
ax.imshow(np.asarray(thb), cmap='gray', vmin=0, vmax=255, interpolation='none')
for v in P['views']:
    x0, y0, x1, y1 = v['box']; hl = v is vid
    ax.add_patch(Rectangle((x0 * f, y0 * f), (x1 - x0) * f, (y1 - y0) * f, fill=False, lw=1.2 if hl else 0.4,
                           ec='#d62728' if hl else '#2a78d6', alpha=1 if hl else 0.8, zorder=3 if hl else 2))
ax.scatter(pts[:, 0] * f, pts[:, 1] * f, s=4, c='#e8710a', lw=0, zorder=4)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f'(d) The rule: $S={P["S"]:.1f}$\n{P["n_views"]} views overlapping by one target', fontsize=6.8)
fig.savefig(OUT / 'fig1.pdf', bbox_inches='tight', pad_inches=0.02, dpi=600); fig.savefig(OUT / 'fig1.png', bbox_inches='tight', pad_inches=0.02, dpi=220)
print(json.dumps({'S_whole': P['S_whole'], 'S_rule': P['S'], 'views': P['n_views'], 'tokens': P['tokens'], 'n_view': n_view,
                  'keys': sorted(P.keys()), 'S2w': S2w, 'S2t': S2t}))
