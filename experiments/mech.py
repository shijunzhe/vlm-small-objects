#!/usr/bin/env python3
"""
S_tok mechanism experiments on REDP-X40.

Geometry (per drawing, identical for every model):
  z0      base scale: symbol short side * z0 = 0.75 * 29.2 px  (GPT at 0.75 tokens)
  FOV     tile side in the z0-scaled drawing: L = 768, S = 384 (margin = FOV/8, core = 3/4 FOV)
  up      the tile is delivered at up * FOV pixels (up in {1, 2})
A blue rectangle marks the core; the model reports instances whose center is inside it.
Kept centers (inside the core) are mapped back to native pixels:
  native = (tile_origin_z0 + local / up) / z0
Template: native legend crop scaled by max(1, z0 * up).

Coordinate interface per model: 'px' (tile pixels) or 'n1000' (0-1000 normalized),
chosen by the interface calibration (step 1) and fixed afterwards.
"""
import json, math, os, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, 'vlm_api')
os.environ.setdefault('RELEASE_ROOT', 'vlm_api')
import arr_common as C

Image.MAX_IMAGE_PIXELS = None
BENCH = Path('../data/redp40/redp_x40')
OUT = Path('results_mech')
OUT.mkdir(parents=True, exist_ok=True)
GPT_TOKEN = 29.2
FOVS = {'L': 768, 'S': 384}

COORD_TEXT = {
    'bbox999': ("Report each instance as a bounding box [x_min, y_min, x_max, y_max] in a fixed 0..999\n"
                "coordinate space over the tile, with the origin at the top-left corner.",
                '"bbox": [<x_min>, <y_min>, <x_max>, <y_max>]', 'bounding box (0..999 coordinates)'),
    'px': ("The tile image is {w}x{h} pixels. All coordinates you output must be in this\n"
           "tile's pixel space, where (0,0) is the top-left corner.",
           '"cx": <pixel_x>, "cy": <pixel_y>', 'center pixel coordinate'),
    'n1000': ("Report coordinates normalized to the tile: x and y each range from 0 to 1000,\n"
              "where (0,0) is the top-left corner and (1000,1000) is the bottom-right corner of the tile.",
              '"cx": <x_0_to_1000>, "cy": <y_0_to_1000>', 'center coordinate (0-1000 normalized)'),
}

SYSTEM = """\
You are a precise engineering drawing analysis assistant.

You will be shown a reference image of ONE target symbol, followed by one tile
of a larger engineering drawing. Your task is to find every instance of that
target symbol whose center lies inside the blue rectangle drawn on the tile.
The area outside the blue rectangle is context only.

{coord}

Respond with ONLY a JSON object (no markdown, no backticks, no commentary). Format:
{{
  "target_type": "target_symbol",
  "count": <integer>,
  "shapes": [
    {{"type": "target_symbol", {fmt}}},
    ...
  ]
}}

The "shapes" array should contain one entry per instance of the target symbol
whose center is inside the blue rectangle, with its {desc}.
The "count" must equal the length of "shapes". Count only symbols matching the
reference image; do NOT count other symbols, text labels, or drawing linework.
If there is no instance inside the rectangle, return a count of 0 and an empty list.
"""

USER = """\
The first image is the reference crop of the target symbol (taken from the
drawing's legend). The second image is one tile of the full drawing.

Count every instance of the target symbol whose center is inside the blue
rectangle. Instances may vary slightly in scale or rotation but share the
reference's distinctive shape.

Respond with JSON only. Be thorough; do not miss any instances, do not invent ones."""


def system_prompt(coord, w, h):
    t, f, d = COORD_TEXT[coord]
    return SYSTEM.format(coord=t.format(w=w, h=h), fmt=f, desc=d)


def load(did):
    ann = json.loads((BENCH / 'annotations' / f'{did}.json').read_text())
    img = Image.open(BENCH / ann['image']).convert('RGB')
    tpl = Image.open(BENCH / ann['template']).convert('RGB')
    return ann, img, tpl


def z0_of(ann):
    return 0.75 * GPT_TOKEN / min(ann['template_size'])


def tiles_for(ann, img, fov_key, up, native=False):
    """Tile dicts for one (fov, up) cell. native=False: the z0-scaled drawing is cut and each
    tile is upsampled by `up` (more tokens, same pixel information). native=True: the same
    regions are resampled directly from the native drawing at scale z0*up (more tokens AND
    more pixel information)."""
    z0 = z0_of(ann)
    F = FOVS[fov_key]; m = F // 8; core = F - 2 * m
    W, H = img.size
    Wz, Hz = max(1, round(W * z0)), max(1, round(H * z0))
    zimg = img.resize((Wz, Hz), Image.LANCZOS)
    nimg = img.resize((max(1, round(W * z0 * up)), max(1, round(H * z0 * up))), Image.LANCZOS) if native else None
    out = []
    for j in range(math.ceil(Hz / core)):
        for i in range(math.ceil(Wz / core)):
            cx0, cy0 = i * core, j * core
            cx1, cy1 = min(Wz, cx0 + core), min(Hz, cy0 + core)
            x0, y0 = max(0, cx0 - m), max(0, cy0 - m)
            x1, y1 = min(Wz, cx1 + m), min(Hz, cy1 + m)
            if native:
                t = nimg.crop((round(x0 * up), round(y0 * up), round(x1 * up), round(y1 * up)))
            else:
                t = zimg.crop((x0, y0, x1, y1))
                if up != 1:
                    t = t.resize((round(t.width * up), round(t.height * up)), Image.LANCZOS)
            d = ImageDraw.Draw(t)
            a, b, c, e = [(v) * up for v in (cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0)]
            d.rectangle([a, b, c - 1, e - 1], outline=(0, 90, 255), width=3)
            out.append({'i': i, 'j': j, 'img': t, 'origin_z0': (x0, y0), 'core_px': (a, b, c, e),
                        'z0': z0, 'up': up})
    return out


def scaled_template(tpl, z0, up):
    s = max(1.0, z0 * up)
    return tpl.resize((round(tpl.width * s), round(tpl.height * s)), Image.LANCZOS) if s > 1 else tpl


def claude_resized(w, h, max_edge=1568, max_tokens=1568):
    """Exact size Claude (standard tier) resizes an image to: the largest aspect-preserving
    size with long edge <= 1568 and ceil(w/28)*ceil(h/28) <= 1568 (platform docs, vision-coordinates)."""
    if max(w, h) <= max_edge and math.ceil(w / 28) * math.ceil(h / 28) <= max_tokens:
        return w, h
    lo, hi = 1, max(w, h)
    best = (1, 1)
    while lo <= hi:
        L = (lo + hi) // 2
        s = L / max(w, h); rw, rh = max(1, round(w * s)), max(1, round(h * s))
        if L <= max_edge and math.ceil(rw / 28) * math.ceil(rh / 28) <= max_tokens:
            best = (rw, rh); lo = L + 1
        else:
            hi = L - 1
    return best


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


FORMAT_AWARE = False   # registered decoding reads numeric cx/cy only; format-aware decoding also reads list-valued points


def point_of(q):
    """(x, y) from one shape dict. Registered: numeric cx/cy. Format-aware (FORMAT_AWARE=True) also accepts
    Qwen-style list points: cx=[x, y], point_2d=[x, y], bbox_2d=[x0, y0, x1, y1], and one-element lists."""
    if not isinstance(q, dict):
        return None
    x, y = fnum(q.get('cx')), fnum(q.get('cy'))
    if x is not None and y is not None:
        return x, y
    if not FORMAT_AWARE:
        return None
    cx, cy = q.get('cx'), q.get('cy')
    for key in ('point_2d', 'point'):
        v = q.get(key)
        if isinstance(v, list) and len(v) == 2 and all(fnum(a) is not None for a in v):
            return fnum(v[0]), fnum(v[1])
    v = q.get('bbox_2d')
    if isinstance(v, list) and len(v) == 4 and all(fnum(a) is not None for a in v):
        return (fnum(v[0]) + fnum(v[2])) / 2, (fnum(v[1]) + fnum(v[3])) / 2
    if isinstance(cx, list) and len(cx) == 2 and all(fnum(a) is not None for a in cx):
        return fnum(cx[0]), fnum(cx[1])
    if isinstance(cx, list) and len(cx) == 1:
        cx = cx[0]
    if isinstance(cy, list) and len(cy) >= 1:
        cy = cy[0]
    x, y = fnum(cx), fnum(cy)
    return (x, y) if x is not None and y is not None else None


def to_native(rec):
    """Kept (inside-core) centers of one tile record, in native pixels."""
    w, h = rec['tile_size']; a, b, c, e = rec['core_px']
    x0, y0 = rec['origin_z0']; z0, up = rec['z0'], rec['up']
    pts = []
    for s in rec.get('shapes', []):
        if not isinstance(s, dict):
            continue
        mode = rec.get('decode', rec['coord'])
        if mode == 'bbox999':
            bb = s.get('bbox')
            if not isinstance(bb, (list, tuple)) or len(bb) != 4 or any(fnum(v) is None for v in bb):
                continue
            x = (fnum(bb[0]) + fnum(bb[2])) / 2 / 999 * w
            y = (fnum(bb[1]) + fnum(bb[3])) / 2 / 999 * h
            mode = 'px'
        else:
            pt = point_of(s)
            x, y = pt if pt else (None, None)
        if x is None or y is None:
            continue
        if mode == 'n1000':
            x, y = x / 1000 * w, y / 1000 * h
        elif mode == 'n1000_yx':
            x, y = y / 1000 * w, x / 1000 * h
        elif mode == 'px_claude':
            # Claude answers in the coordinates of the image it actually saw, which the API
            # downsizes to <= 1568 px long side and <= 1.15 MP; map back to tile pixels.
            rw, rh = claude_resized(w, h)
            x, y = x * w / rw, y * h / rh
        if a <= x < c and b <= y < e:
            pts.append(((x0 + x / up) / z0, (y0 + y / up) / z0))
    return pts


def match(pred, gt, tau):
    if not pred or not gt:
        return 0, len(pred), len(gt)
    P, G = np.asarray(pred, float), np.asarray(gt, float)
    D = np.linalg.norm(P[:, None] - G[None], axis=2)
    # maximum one-to-one matching within tau (pairs beyond tau are forbidden; ties broken by total distance)
    r, c = linear_sum_assignment(np.where(D <= tau, D, 1e6))
    return int((D[r, c] <= tau).sum()), len(pred), len(gt)


def taus(ann):
    tw, th = ann['template_size']
    L = max(ann['width'], ann['height'])
    return 0.5 * math.sqrt(tw * th), 30 / min(1, 1600 / L)


def run_jobs(jobs, out_path, key, workers=6):
    """jobs: list of dicts with 'model','coord','tpl','tile' + metadata."""
    out_path = Path(out_path)
    done = set()
    if out_path.exists():
        for l in out_path.open():
            r = json.loads(l)
            if r['status'] in ('success', 'parse_error'):
                done.add(key(r))
    todo = [j for j in jobs if key(j['meta']) not in done]
    print(f'{out_path.name}: {len(done)} done, {len(todo)} to go', flush=True)
    lock = threading.Lock()

    def one(job):
        t = job['tile']; w, h = t['img'].size
        rec = dict(job['meta'], tile_size=[w, h], core_px=list(t['core_px']), origin_z0=list(t['origin_z0']),
                   z0=t['z0'], up=t['up'], coord=job['coord'])
        for attempt in range(3):
            try:
                raw, usage = C.call(job['model'], [job['tpl'], t['img']],
                                    job.get('system') or system_prompt(job['coord'], w, h), job.get('user', USER))
                p = C.parse(raw)
                sh = p.get('shapes', []) if isinstance(p, dict) else []
                rec.update(status='parse_error' if p.get('parse_error') else 'success',
                           declared=p.get('count'), shapes=sh if isinstance(sh, list) else [],
                           usage=usage, raw=raw)
                break
            except Exception as e:
                rec.update(status='api_error', error=f'{type(e).__name__}: {e}'[:300])
                time.sleep(5 * 2 ** attempt)
        with lock:
            with out_path.open('a') as f:
                f.write(json.dumps(rec) + '\n')
        return rec

    n = 0
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, todo):
            n += 1
            if n % 50 == 0 or r['status'] != 'success':
                print(f'  [{n}/{len(todo)}] {r.get("did")} {r.get("cell","")} {r["status"]} {r.get("error","")[:100]}', flush=True)


# ---- robustness readings added for the expansion experiments (reported as deviations, see paper appendix) ----
import re as _re
_PAIR = _re.compile(r'"cx"\s*:\s*(-?\d+(?:\.\d+)?)\s*,\s*"cy"\s*:\s*\[?\s*(-?\d+(?:\.\d+)?)')
_LIST = _re.compile(r'"cx"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]?')


def lenient_points(raw):
    """Point centres recovered from a reply whose JSON is malformed (missing brackets, Qwen 'cx':[x,y] lists).
    Numeric cx/cy pairs first; if none, Qwen-style list pairs. Used only for records with status parse_error."""
    raw = raw or ''
    pts = [(float(a), float(b)) for a, b in _PAIR.findall(raw)]
    if not pts:
        pts = [(float(a), float(b)) for a, b in _LIST.findall(raw)]
    return pts


def claude55_scale(pts, W, H):
    """Label-free frame rule for Claude 5.5 (high-resolution tier). When the image exceeds the tier limits the model
    sometimes answers in its resized frame and sometimes in the original frame. If any predicted coordinate lies
    outside the resized frame the reply must be in the original frame; otherwise the resized frame is assumed
    (the registered reading). Returns (fx, fy) to multiply predicted coordinates by."""
    rw, rh = claude_resized(W, H, max_edge=2576, max_tokens=4784)
    if (rw, rh) == (W, H):
        return 1.0, 1.0
    if any(x > rw * 1.02 or y > rh * 1.02 for x, y in pts):
        return 1.0, 1.0
    return W / rw, H / rh


def claude55_scale_ink(pts, img, m):
    """Image-based (label-free) frame rule for Claude 5.5 when the input exceeds its high-resolution tier: among the
    two candidate frames (original, resized), choose the one under which more predicted centres fall on ink
    (a pixel darker than 128 within m/2). Points outside the resized frame force the original frame; ties keep the
    registered (resized) frame. `img` is the image as sent (PIL, RGB or L)."""
    W, H = img.size
    rw, rh = claude_resized(W, H, max_edge=2576, max_tokens=4784)
    if (rw, rh) == (W, H) or not pts:
        return 1.0, 1.0
    if any(x > rw * 1.02 or y > rh * 1.02 for x, y in pts):
        return 1.0, 1.0
    import numpy as _np
    g = _np.asarray(img.convert('L')); r = max(1, int(m / 2))
    def hits(f):
        h = 0
        for x, y in pts:
            X, Y = int(round(x * f)), int(round(y * f))
            win = g[max(0, Y - r):Y + r + 1, max(0, X - r):X + r + 1]
            h += int(win.size > 0 and win.min() < 128)
        return h
    return (1.0, 1.0) if hits(1.0) > hits(W / rw) else (W / rw, H / rh)
