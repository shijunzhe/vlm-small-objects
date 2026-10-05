#!/usr/bin/env python3
"""
X2: per-call capacity on real sheets at fixed tokens per symbol. The model-specific 1024-px tiles (symbol 44 px)
of the 14 test sheets are packed k per call into a mosaic (k=2: 1x2, k=4: 2x2, k=8: 2x4), separated by grey gutters.
Each tile keeps its blue core frame; the model reports centres in mosaic pixels, which are mapped back to the tile and
then to native pixels; only centres inside a tile's core are kept. k=1 is the existing D condition.
usage: python3 mosaic.py run <vendor> <k> [run] | score
"""
import json, math, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import mech as M, ceilings as K
from x_small import x2_tiles

OUT = Path('results_x/mosaic'); OUT.mkdir(parents=True, exist_ok=True)
SHEETS = [d for d in K.TEST if json.loads((M.BENCH / 'annotations' / f'{d}.json').read_text())['subset'] == 'sheet']
CELL, GUT = 1024, 0   # no gutter: k=8 stays at exactly 4096x2048 (8192 GPT patches); cells are separated by drawn lines
LAYOUT = {2: (1, 2), 4: (2, 2), 8: (2, 4)}
SYS = """You are a precise engineering drawing analysis assistant.

You will be shown a reference image of ONE target symbol, followed by ONE image that contains {k} tiles of a larger
engineering drawing, arranged in a grid and separated by dark grey lines. Each tile has a blue rectangle drawn on it.
Find every instance of the target symbol whose center lies inside any blue rectangle. Areas outside the blue
rectangles are context only.

The image is {w}x{h} pixels. All coordinates you output must be in this image's pixel space, where (0,0) is the
top-left corner of the whole image.

Respond with ONLY a JSON object (no markdown, no commentary). Format:
{{"target_type": "target_symbol", "count": <integer>, "shapes": [{{"type": "target_symbol", "cx": <pixel_x>, "cy": <pixel_y>}}, ...]}}
The "count" must equal the length of "shapes". Count only symbols matching the reference image."""
USER = ("The first image is the reference crop of the target symbol. The second image contains several tiles of the drawing. "
        "Count every instance of the target symbol whose center is inside a blue rectangle. Be thorough; do not miss any "
        "instances, do not invent ones. JSON only.")


def mosaics(ts, k):
    rows, cols = LAYOUT[k]
    for start in range(0, len(ts), k):
        group = ts[start:start + k]
        W = cols * CELL + (cols - 1) * GUT; H = rows * CELL + (rows - 1) * GUT
        if len(group) <= cols:   # last partial group fits in one row
            H = CELL
        can = Image.new('RGB', (W, H), (128, 128, 128)); cells = []
        for i, t in enumerate(group):
            r, c = divmod(i, cols); x, y = c * (CELL + GUT), r * (CELL + GUT)
            can.paste(t['img'], (x, y)); cells.append((x, y, t))
        from PIL import ImageDraw
        dr = ImageDraw.Draw(can)
        for c in range(1, cols):
            dr.line([c * CELL, 0, c * CELL, H], fill=(90, 90, 90), width=4)
        for r in range(1, rows):
            dr.line([0, r * CELL, W, r * CELL], fill=(90, 90, 90), width=4)
        yield start // k, can, cells


def run(vendor, k, run_id=0):
    import gen_test  # installs dispatch for new-generation vendors
    out = OUT / f'{vendor}_k{k}_run{run_id}.jsonl'
    done = set()
    if out.exists():
        for l in out.open():
            r = json.loads(l)
            if r['status'] in ('success', 'parse_error'): done.add((r['did'], r['g']))
    lock = threading.Lock()
    for did in SHEETS:
        ann, img, tpl = M.load(did)
        ts, z = x2_tiles(ann, img, 1024); tplz = M.scaled_template(tpl, z, 1)
        jobs = [(g, can, cells) for g, can, cells in mosaics(ts, k) if (did, g) not in done]

        def one(job):
            g, can, cells = job
            rec = {'did': did, 'g': g, 'k': k, 'vendor': vendor, 'run': run_id, 'size': list(can.size),
                   'cells': [[x, y, list(t['core_px']), list(t['origin_z0']), t['z0'], list(t['img'].size)] for x, y, t in cells]}
            for a in range(3):
                try:
                    raw, u = M.C.call(vendor, [tplz, can], SYS.format(k=len(cells), w=can.size[0], h=can.size[1]), USER)
                    p = M.C.parse(raw); sh = p.get('shapes', []) if isinstance(p, dict) else []
                    rec.update(status='parse_error' if p.get('parse_error') else 'success', shapes=sh if isinstance(sh, list) else [], usage=u, raw=raw[:3000]); break
                except Exception as e:
                    rec.update(status='api_error', error=str(e)[:200]); time.sleep(10 * 2 ** a)
            with lock, out.open('a') as f:
                f.write(json.dumps(rec) + '\n')
            return rec
        with ThreadPoolExecutor(4) as ex:
            for r in ex.map(one, jobs):
                if r['status'] != 'success': print(' ', did, r['g'], r['status'], r.get('error', '')[:100], flush=True)
        print(vendor, k, did, 'done', flush=True)


def to_native(rec):
    pts = []
    W, H = rec['size']
    if rec['vendor'].startswith('sonnet55'):   # Claude answers in the frame of its internally resized image
        rw, rh = M.claude_resized(W, H, max_edge=2576, max_tokens=4784); fx, fy = W / rw, H / rh
    else:
        fx = fy = 1.0
    for q in rec.get('shapes', []):
        if not isinstance(q, dict): continue
        x, y = M.fnum(q.get('cx')), M.fnum(q.get('cy'))
        if x is None or y is None: continue
        x, y = x * fx, y * fy
        for cx, cy, core, origin, z0, size in rec['cells']:
            lx, ly = x - cx, y - cy
            if 0 <= lx < size[0] and 0 <= ly < size[1]:
                a, b, c, e = core
                if a <= lx < c and b <= ly < e:
                    pts.append(((origin[0] + lx) / z0, (origin[1] + ly) / z0))
                break
    return pts


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        M.C.load_env('../config.env')
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 0)
