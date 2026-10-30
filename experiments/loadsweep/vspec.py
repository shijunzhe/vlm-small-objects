#!/usr/bin/env python3
"""
Visual-spec experiment (design v3, claude/paper2_visual_spec_experiment_design_v3.md).

Task family F0: n targets + n distractors (3 other legend templates, cycled), lattice pitch 1.6 m, ~50% slot
occupancy, grey horizontal/vertical linework (0.16 lines per slot, length 1.25 m .. 10 m, grey 150, width
max(1, round(m/40)) px) drawn first, symbols pasted on top through their ink mask (grey < 200), as in syn.py.
Layouts live in symbol units (m = 1) and are fixed per (set, i, n); every render of a base scene is the same
unit layout scaled by m = S * t. Overlap is checked in unit coordinates (ink boxes, margin 0.1) and again in
pixels for every planned render (ink boxes must not touch, 1 px gap, inside the scene); a failure at any
planned render replaces the layout for all renders of that (set, i, n) with the next seed k.
Template combinations are fixed per (set, i) and never changed.
"""
import json, math, random, zlib, os, sys, time, threading, base64, io
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent                     # release copy: experiments/loadsweep/
TPL_DIR = ROOT.parents[1] / 'data' / 'redp40' / 'redp_x40' / 'templates'
DIDS = [f'd{i:03d}' for i in range(1, 41)]
PITCH, JIT, BORDER, UNIT_MARGIN = 1.6, 1.6 / 6, 0.5, 0.1
LINE_DENSITY, LINE_MIN, LINE_MAX = 0.16, 1.25, 10.0
MAX_K = 50

# operational interface definitions (design v3, section 3); maxside is the largest canvas side used
MODELS = {
    # Amendment 1 (2026-10-25, approved by the user): GPT and Opus do not enlarge small inputs (billing verified),
    # so their minimum canvas is one token unit (no padding). The original 16t rule (512 / 448 px) padded small
    # phase-1 scenes with blank area and depressed Opus recall/precision; those runs are archived in
    # archive_padded16t/. Qwen keeps 512 px (vendor enlarges small images); Gemini keeps its fixed 1024 canvas.
    'gpt61':   dict(kind='res', t=32.0, minside=32, maxside=3072, coord='px'),
    'gpt56':   dict(kind='res', t=32.0, minside=32, maxside=2896, coord='px'),
    'opus55':  dict(kind='res', t=28.0, minside=28, maxside=1932, coord='px'),
    'gem38':   dict(kind='grid', t=1024 / math.sqrt(1090), canvas=1024, coord='n1000'),
    # extension (2026-10-26): Gemini 4 Argon, measured fixed 1,091 image tokens at every size (phase0_calib_gem4.jsonl)
    'gem4':    dict(kind='grid', t=1024 / math.sqrt(1091), canvas=1024, coord='n1000'),
    'qwen3vl': dict(kind='res', t=32.0, minside=512, maxside=1600, coord='n1000'),
}
ORDER = ['gpt61', 'gpt56', 'opus55', 'gem38', 'qwen3vl']


def crc(s):
    return zlib.crc32(s.encode())


# ---------------- templates ----------------
_TPL, _INK = {}, {}


def tpl(d):
    if d not in _TPL:
        _TPL[d] = Image.open(TPL_DIR / f'{d}.png').convert('RGB')
    return _TPL[d]


def ink_box_unit(d):
    """Ink bounding box of template d relative to its centre, in symbol units (m = sqrt(w*h) = 1)."""
    if d not in _INK:
        im = np.asarray(tpl(d).convert('L')); h, w = im.shape; s = math.sqrt(w * h)
        ys, xs = np.where(im < 200)
        _INK[d] = ((xs.min() - w / 2) / s, (ys.min() - h / 2) / s, (xs.max() + 1 - w / 2) / s, (ys.max() + 1 - h / 2) / s)
    return _INK[d]


_SC = {}


def scaled(d, m):
    key = (d, round(m, 4))
    if key not in _SC:
        t = tpl(d); w, h = t.size; f = m / math.sqrt(w * h)
        im = t.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)
        a = np.asarray(im.convert('L')) < 200
        ys, xs = np.where(a)
        box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1) if len(xs) else (0, 0, im.width, im.height)
        mask = Image.fromarray((a * 255).astype(np.uint8))
        _SC[key] = (im, mask, box)
    return _SC[key]


# ---------------- unit layouts ----------------
def combo(set_name, i):
    rng = random.Random(crc(f'{set_name}|combo|{i}'))
    tgt = rng.choice(DIDS); dis = rng.sample([d for d in DIDS if d != tgt], 3)
    return tgt, dis


def lattice(n):
    cols = math.ceil(math.sqrt(4 * n)); rows = math.ceil(4 * n / cols)
    return cols, rows


def extent(n):
    cols, rows = lattice(n)
    return 2 * BORDER + PITCH * cols, 2 * BORDER + PITCH * rows


def _boxes_clear(b, placed, margin):
    for q in placed:
        if b[0] < q[2] + margin and q[0] < b[2] + margin and b[1] < q[3] + margin and q[1] < b[3] + margin:
            return False
    return True


def layout(set_name, i, n, k):
    """Unit layout or None if placement failed for this k."""
    tgt, dis = combo(set_name, i)
    cols, rows = lattice(n); Wu, Hu = extent(n)
    rng = random.Random(crc(f'{set_name}|lay|{i}|{n}|{k}'))
    objs = [(tgt, True)] * n + [(dis[j % 3], False) for j in range(n)]
    order = sorted(range(len(objs)), key=lambda j: -((lambda b: (b[2] - b[0]) * (b[3] - b[1]))(ink_box_unit(objs[j][0]))))
    slots = [(BORDER + PITCH * (c + 0.5), BORDER + PITCH * (r + 0.5)) for r in range(rows) for c in range(cols)]
    free = list(range(len(slots))); rng.shuffle(free)
    placed_boxes, out = [], [None] * len(objs)
    for j in order:
        d, is_t = objs[j]; ib = ink_box_unit(d); ok = False
        for attempt in range(2):
            for si in list(free):
                sx, sy = slots[si]
                ux, uy = sx + rng.uniform(-JIT, JIT), sy + rng.uniform(-JIT, JIT)
                b = (ux + ib[0], uy + ib[1], ux + ib[2], uy + ib[3])
                if b[0] < UNIT_MARGIN or b[1] < UNIT_MARGIN or b[2] > Wu - UNIT_MARGIN or b[3] > Hu - UNIT_MARGIN:
                    continue
                if _boxes_clear(b, placed_boxes, UNIT_MARGIN):
                    placed_boxes.append(b); out[j] = (d, is_t, ux, uy); free.remove(si); ok = True; break
            if ok:
                break
        if not ok:
            return None
    lrng = random.Random(crc(f'{set_name}|lines|{i}|{n}'))
    lines = []
    for _ in range(round(LINE_DENSITY * cols * rows)):
        x0, y0 = lrng.uniform(0, Wu), lrng.uniform(0, Hu); L = lrng.uniform(LINE_MIN, LINE_MAX)
        lines.append((x0, y0, min(Wu, x0 + L), y0) if lrng.random() < 0.5 else (x0, y0, x0, min(Hu, y0 + L)))
    return dict(set=set_name, i=i, n=n, k=k, target=tgt, distractors=dis, Wu=Wu, Hu=Hu, objs=out, lines=lines)


# ---------------- rendering ----------------
def canvas_for(model, Wu, Hu, m):
    g = MODELS[model]
    if g['kind'] == 'grid':
        W = H = g['canvas']
        if Wu * m > W or Hu * m > H:
            return None
        return W, H
    t = g['t']
    W = max(g['minside'], math.ceil(Wu * m / t) * int(t)); H = max(g['minside'], math.ceil(Hu * m / t) * int(t))
    if max(W, H) > g['maxside']:
        return None
    return W, H


def render(lay, model, S, pad=True, check_only=False):
    """Returns dict with image (unless check_only), target points (px), geometry; or {'unreachable': reason}."""
    g = MODELS[model]; t = g['t']; m = S * t
    Wu, Hu = lay['Wu'], lay['Hu']
    cv = canvas_for(model, Wu, Hu, m)
    if cv is None:
        return {'unreachable': 'canvas'}
    W, H = cv
    if not pad and g['kind'] == 'res':
        W, H = max(int(t), math.ceil(Wu * m / t) * int(t)), max(int(t), math.ceil(Hu * m / t) * int(t))
    ox, oy = (W - Wu * m) / 2, (H - Hu * m) / 2
    # pixel check: ink boxes of every object at this render
    boxes, pastes = [], []
    for (d, is_t, ux, uy) in lay['objs']:
        im, mask, ib = scaled(d, m)
        px, py = round(ox + ux * m - im.width / 2), round(oy + uy * m - im.height / 2)
        boxes.append((px + ib[0], py + ib[1], px + ib[2], py + ib[3])); pastes.append((im, mask, px, py, is_t))
    B = np.array(boxes)
    sx0, sy0, sx1, sy1 = ox, oy, ox + Wu * m, oy + Hu * m
    if (B[:, 0] < max(0, math.floor(sx0))).any() or (B[:, 1] < max(0, math.floor(sy0))).any() or (B[:, 2] > min(W, math.ceil(sx1))).any() or (B[:, 3] > min(H, math.ceil(sy1))).any():
        return {'pixel_fail': 'bounds'}
    # pairwise: boxes must be separated by >= 1 px gap
    ix = (B[:, None, 0] < B[None, :, 2] + 1) & (B[None, :, 0] < B[:, None, 2] + 1) & (B[:, None, 1] < B[None, :, 3] + 1) & (B[None, :, 1] < B[:, None, 3] + 1)
    np.fill_diagonal(ix, False)
    if ix.any():
        return {'pixel_fail': 'overlap'}
    tim, _, _ = scaled(lay['target'], m)
    m_act = math.sqrt(tim.width * tim.height)
    geo = dict(model=model, S_nom=S, m_nom=m, m_act=m_act, S_act=m_act / t, canvas=[W, H], offset=[ox, oy],
               scene_px=[Wu * m, Hu * m], padded=bool(g['kind'] == 'grid' or W > math.ceil(Wu * m / t) * int(t) or H > math.ceil(Hu * m / t) * int(t)),
               line_width=max(1, round(m / 40)))
    if check_only:
        return {'geo': geo}
    can = Image.new('RGB', (W, H), 'white'); dr = ImageDraw.Draw(can); lw = max(1, round(m / 40))
    for (x0, y0, x1, y1) in lay['lines']:
        dr.line([ox + x0 * m, oy + y0 * m, ox + x1 * m, oy + y1 * m], fill=(150, 150, 150), width=lw)
    pts = []
    for (im, mask, px, py, is_t) in pastes:
        can.paste(im, (px, py), mask=mask)
        if is_t:
            pts.append([px + im.width / 2, py + im.height / 2])
    return {'img': can, 'points': pts, 'geo': geo}


def frozen_layout(set_name, i, n, renders, log=None):
    """First k whose unit layout passes the pixel check at every planned (model, S) render.
    Returns (layout, k) or (None, None) when no k <= MAX_K works (geometrically unreachable)."""
    for k in range(MAX_K):
        lay = layout(set_name, i, n, k)
        if lay is None:
            continue
        bad = None
        for (model, S) in renders:
            r = render(lay, model, S, check_only=True)
            if 'pixel_fail' in r:
                bad = (model, S, r['pixel_fail']); break
        if bad is None:
            if log is not None:
                log.append(dict(set=set_name, i=i, n=n, k=k, target=lay['target'], distractors=lay['distractors']))
            return lay, k
    if log is not None:
        log.append(dict(set=set_name, i=i, n=n, k=None, target=combo(set_name, i)[0], distractors=combo(set_name, i)[1]))
    return None, None


# ---------------- expected image tokens (section 3) ----------------
def expected_tokens(model, W, H):
    if model in ('gpt61', 'gpt56'):
        return math.floor(1.2 * math.ceil(W / 32) * math.ceil(H / 32)) + 19
    if model == 'opus55':
        return math.ceil(W / 28) * math.ceil(H / 28)
    if model == 'gem38':
        return 1090
    if model == 'gem4':
        return 1091
    if model == 'qwen3vl':
        W2, H2 = max(W, 256), max(H, 256)
        return math.ceil(W2 / 32) * math.ceil(H2 / 32)
    raise KeyError(model)
