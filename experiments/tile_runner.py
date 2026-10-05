#!/usr/bin/env python3
"""
S_tok pilot: does enlarging each symbol to more visual tokens (zoom + tiles) improve
VLM symbol counting/localization on REDP-X40?

Levels (per drawing):
  full   : whole drawing downscaled to 1600 px (the existing E2 protocol; results reused)
  tile0  : same zoom as `full` (long side 1600) but split into tiles  -> isolates tiling
  s1.5   : zoom so that the symbol's short side ~= 1.5 visual tokens of the model
  s3     : ~= 3 visual tokens
Each tile = core region (owned) + context margin; a blue rectangle marks the core.
The model reports instances whose center is inside the rectangle, in tile pixels;
the program keeps only centers inside the core and maps them back to native pixels,
so tiles partition the drawing and no de-duplication is needed.
Template (legend crop) is shown at scale max(1, zoom).
"""
import argparse, json, math, os, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw

sys.path.insert(0, 'vlm_api')
os.environ.setdefault('RELEASE_ROOT', 'vlm_api')
import arr_common as C

BENCH = Path('../data/redp40/redp_x40')
OUT = Path('results')
TOKEN_SIDE = {'gpt': 29.2, 'claude': 27.4, 'qwen3vl': 33.0}   # measured px per token side (calib_tokens.json)
TILE, MARGIN = 1024, 128
# Models that spend a FIXED number of tokens per image (~16x16 grid, measured in calib_tokens.json):
# the token side is tile_size/16, so S_tok is set by choosing the tile size at native zoom.
FIXED_GRID = {'gemini': 16, 'gemma4': 16}
Image.MAX_IMAGE_PIXELS = None

SYSTEM = """\
You are a precise engineering drawing analysis assistant.

You will be shown a reference image of ONE target symbol, followed by one tile
of a larger engineering drawing. Your task is to find every instance of that
target symbol whose center lies inside the blue rectangle drawn on the tile.
The area outside the blue rectangle is context only.

The tile image is {w}x{h} pixels. All coordinates you output must be in this
tile's pixel space, where (0,0) is the top-left corner.

Respond with ONLY a JSON object (no markdown, no backticks, no commentary). Format:
{{
  "target_type": "target_symbol",
  "count": <integer>,
  "shapes": [
    {{"type": "target_symbol", "cx": <pixel_x>, "cy": <pixel_y>}},
    ...
  ]
}}

The "shapes" array should contain one entry per instance of the target symbol
whose center is inside the blue rectangle, with its center pixel coordinate.
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


def zoom_for(level, ann, model):
    W, H = ann['width'], ann['height']
    if level in ('full', 'tile0'):
        return min(1.0, 1600 / max(W, H))
    S = float(level[1:])
    return S * TOKEN_SIDE[model] / min(ann['template_size'])


def tile_geometry(level, ann, model):
    """Returns (zoom, tile, margin)."""
    if model in FIXED_GRID and level.startswith('s'):
        S = float(level[1:])
        T = int(round(FIXED_GRID[model] * min(ann['template_size']) / S))
        T = max(256, min(1536, T))
        return 1.0, T, max(16, int(0.125 * T))
    return zoom_for(level, ann, model), TILE, MARGIN


def make_tiles(img, z, tile=TILE, margin=MARGIN):
    TILE_, MARGIN_ = tile, margin
    W, H = img.size
    Wz, Hz = max(1, round(W * z)), max(1, round(H * z))
    zimg = img.resize((Wz, Hz), Image.LANCZOS) if abs(z - 1) > 1e-6 else img
    core = TILE_ - 2 * MARGIN_
    tiles = []
    for j in range(math.ceil(Hz / core)):
        for i in range(math.ceil(Wz / core)):
            cx0, cy0 = i * core, j * core
            cx1, cy1 = min(Wz, cx0 + core), min(Hz, cy0 + core)
            x0, y0 = max(0, cx0 - MARGIN_), max(0, cy0 - MARGIN_)
            x1, y1 = min(Wz, cx1 + MARGIN_), min(Hz, cy1 + MARGIN_)
            t = zimg.crop((x0, y0, x1, y1)).convert('RGB')
            d = ImageDraw.Draw(t)
            d.rectangle([cx0 - x0, cy0 - y0, cx1 - x0 - 1, cy1 - y0 - 1], outline=(0, 90, 255), width=3)
            tiles.append({'i': i, 'j': j, 'img': t, 'origin': (x0, y0),
                          'core': (cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0)})
    return tiles


def run(model, dids, levels, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    outp = OUT / f'{model}.jsonl'
    done = set()
    if outp.exists():
        for l in outp.open():
            r = json.loads(l)
            if r['status'] in ('success', 'parse_error'):
                done.add((r['did'], r['level'], r['i'], r['j']))
    jobs = []
    for did in dids:
        ann = json.loads((BENCH / 'annotations' / f'{did}.json').read_text())
        img = Image.open(BENCH / ann['image']).convert('RGB')
        tpl = Image.open(BENCH / ann['template']).convert('RGB')
        for lev in levels:
            z, T, M = tile_geometry(lev, ann, model)
            tz = max(1.0, z)
            tpl_z = tpl.resize((round(tpl.width * tz), round(tpl.height * tz)), Image.LANCZOS) if tz > 1 else tpl
            for t in make_tiles(img, z, T, M):
                if (did, lev, t['i'], t['j']) in done:
                    continue
                jobs.append((did, lev, z, t, tpl_z))
    print(f'{model}: {len(jobs)} tile calls to go', flush=True)
    import threading
    lock = threading.Lock()

    def one(job):
        did, lev, z, t, tpl_z = job
        w, h = t['img'].size
        rec = {'did': did, 'level': lev, 'zoom': z, 'i': t['i'], 'j': t['j'], 'origin': t['origin'],
               'core': t['core'], 'tile_size': [w, h], 'model': model}
        for attempt in range(3):
            try:
                raw, usage = C.call(model, [tpl_z, t['img']], SYSTEM.format(w=w, h=h), USER)
                p = C.parse(raw)
                shapes = p.get('shapes', []) if isinstance(p, dict) else []
                rec.update({'status': 'parse_error' if p.get('parse_error') else 'success',
                            'declared': p.get('count'), 'shapes': shapes if isinstance(shapes, list) else [],
                            'usage': usage, 'raw': raw})
                break
            except Exception as e:
                rec.update({'status': 'api_error', 'error': f'{type(e).__name__}: {e}'[:300]})
                time.sleep(5 * 2 ** attempt)
        with lock:
            with outp.open('a') as f:
                f.write(json.dumps(rec) + '\n')
        return rec

    with ThreadPoolExecutor(workers) as ex:
        for k, r in enumerate(ex.map(one, jobs), 1):
            if k % 25 == 0 or r['status'] != 'success':
                print(f'  [{k}/{len(jobs)}] {r["did"]} {r["level"]} {r["status"]} {r.get("error","")[:100]}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gpt')
    ap.add_argument('--dids', default='d001,d008,d010,d017,d019')
    ap.add_argument('--levels', default='tile0,s1.5,s3')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--dry', action='store_true')
    a = ap.parse_args()
    C.load_env('../config.env')
    if a.dry:
        for did in a.dids.split(','):
            ann = json.loads((BENCH / 'annotations' / f'{did}.json').read_text())
            img = Image.open(BENCH / ann['image'])
            for lev in a.levels.split(','):
                z, T, M = tile_geometry(lev, ann, a.model)
                print(did, lev, round(z, 3), T, len(make_tiles(img, z, T, M)))
    else:
        run(a.model, a.dids.split(','), a.levels.split(','), a.workers)
