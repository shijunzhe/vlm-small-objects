#!/usr/bin/env python3
"""Figure 1: what S and L are, on a real REDP-X40 sheet (d018).
Generic resolution-scaled interface: 32 px tokens, budget N tokens per image.
(a) the sheet and the views of the safe rule at S* = 1.5 (one view highlighted);
(b)-(d) the same 5m x 5m neighbourhood of one target, resampled to what the model receives, with the token grid:
whole image at N = 2,500; a high-resolution mode with 4x the budget; one view of the safe rule."""
import json, math, sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'p2release_tools/extra/tools'))
try:
    import safe_tiles as ST
except ImportError:                       # inside the release: tools/ is a sibling of experiments/
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'tools')); import safe_tiles as ST

Image.MAX_IMAGE_PIXELS = None
BENCH = Path('../data/redp40/redp_x40')
OUT = Path('../paper/figures')
D, T, N = 'd018', 32, 2500
ann = json.load(open(BENCH / 'annotations' / f'{D}.json'))
img = Image.open(BENCH / 'images' / f'{D}.png').convert('L')
W, H = img.size
w, h = ann['template_size']; m = math.sqrt(w * h)
P = ST.plan(W, H, w, h, interface='res', token=T, budget=N, S_star=1.5)
S_whole = P['S_whole']; S_hi = ST.whole_image_S(W, H, m, 'res', T, 4 * N); S_tile = P['S']
pts = np.array(ann['points'])
cx, cy = pts[0]                                   # the target shown in panels (b)-(d)
vid = next(v for v in P['views'] if v['box'][0] <= cx < v['box'][2] and v['box'][1] <= cy < v['box'][3])
n_view = int(sum((vid['box'][0] <= x < vid['box'][2]) and (vid['box'][1] <= y < vid['box'][3]) for x, y in pts))

def received(S, half=2.5):
    """The 2*half*m neighbourhood of the target as the model receives it at S tokens per target side."""
    s = S * T / m; r = half * m
    crop = img.crop((round(cx - r), round(cy - r), round(cx + r), round(cy + r)))
    k = max(1, round(crop.width * s))
    return np.asarray(crop.resize((k, k), Image.LANCZOS)), s

plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif'})
fig, axs = plt.subplots(1, 4, figsize=(6.9, 2.2), gridspec_kw={'width_ratios': [1.45, 1, 1, 1], 'wspace': 0.12})
th = img.copy(); th.thumbnail((1400, 1400)); f = th.width / W
ax = axs[0]; ax.imshow(np.asarray(th), cmap='gray', vmin=0, vmax=255)
for v in P['views']:
    x0, y0, x1, y1 = v['box']; hl = v is vid
    ax.add_patch(Rectangle((x0 * f, y0 * f), (x1 - x0) * f, (y1 - y0) * f, fill=False, lw=1.2 if hl else 0.35,
                           ec='#d62728' if hl else '#1f77b4', alpha=1 if hl else 0.7, zorder=3 if hl else 2))
ax.scatter(pts[:, 0] * f, pts[:, 1] * f, s=5, c='#ff7f0e', lw=0, zorder=4)
ax.set_title(f'(a) Sheet, {W}$\\times${H} px\n{len(pts)} targets (orange)\n{P["n_views"]} safe views (blue)', fontsize=6.5)
ax.set_xticks([]); ax.set_yticks([])
for ax, S, title, tok, note in [(axs[1], S_whole, '(b) Whole image', f'{N:,} tokens', f'$L$: sheet ({len(pts)} targets)'),
                           (axs[2], S_hi, '(c) High-resolution mode', f'{4 * N:,} tokens', f'$L$: sheet ({len(pts)} targets)'),
                           (axs[3], S_tile, '(d) One safe view', f'{N:,} tokens', f'$L$: view ({n_view} targets)')]:
    a, s = received(S)
    ax.imshow(a, cmap='gray', vmin=0, vmax=255, interpolation='nearest', extent=(0, a.shape[1], a.shape[0], 0))
    # token grid aligned to the view origin (approximately: the grid phase is illustrative)
    for g in np.arange(0, a.shape[1] + 1e-9, T):
        ax.axvline(g, color='#1f77b4', lw=0.5, alpha=0.8); ax.axhline(g, color='#1f77b4', lw=0.5, alpha=0.8)
    c = a.shape[0] / 2; side = m * s
    ax.add_patch(Rectangle((c - side / 2, c - side / 2), side, side, fill=False, ec='#ff7f0e', lw=1.0))
    ax.set_xlim(0, a.shape[1]); ax.set_ylim(a.shape[0], 0); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f'{title}\n$S = {S:.2f}$, {tok}\n{note}', fontsize=6.5)
fig.savefig(OUT / 'fig_intro.pdf', bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT / 'fig_intro.png', bbox_inches='tight', pad_inches=0.02, dpi=200)
print(json.dumps({'S_whole': S_whole, 'S_hi': S_hi, 'S_tile': S_tile, 'n_views': P['n_views'], 'n_view': n_view,
                  'tokens_rule': P['tokens'], 'm': m, 'target': [cx, cy]}))
