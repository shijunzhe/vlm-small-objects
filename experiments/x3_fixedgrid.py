#!/usr/bin/env python3
"""
X3: field-of-view sweep for fixed-grid models (Gemini 2.5 Flash, Gemma 4) on the dev set.
These models spend a fixed ~16x16 token grid per image, so the token side is FOV/16 and the
symbol occupies k = 16 * symbol_px / FOV tokens. Tiles are cut from the NATIVE drawing with
FOV = 16 * symbol_px / k for k in {1.0, 1.5, 2.5} (core = 3/4 FOV, margin = FOV/8); a tile
larger than 1536 px is downscaled to 1536 before sending (no effect for a fixed grid).
Coordinates: 0-1000 normalized (interface calibration, step 1).
"""
import math, sys
from PIL import Image, ImageDraw
import mech as M

DIDS = ('d001', 'd004', 'd008', 'd010', 'd017', 'd019')


def fov_tiles(ann, img, k):
    sym = min(ann['template_size'])
    F = max(96, int(round(16 * sym / k)))
    m = F // 8; core = F - 2 * m
    W, H = img.size
    out = []
    for j in range(math.ceil(H / core)):
        for i in range(math.ceil(W / core)):
            cx0, cy0 = i * core, j * core
            cx1, cy1 = min(W, cx0 + core), min(H, cy0 + core)
            x0, y0 = max(0, cx0 - m), max(0, cy0 - m)
            x1, y1 = min(W, cx1 + m), min(H, cy1 + m)
            t = img.crop((x0, y0, x1, y1))
            s = min(1.0, 1536 / max(t.size))
            if s < 1:
                t = t.resize((round(t.width * s), round(t.height * s)), Image.LANCZOS)
            a, b, c, e = [(v) * s for v in (cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0)]
            ImageDraw.Draw(t).rectangle([a, b, c - 1, e - 1], outline=(0, 90, 255), width=3)
            # origin/zoom chosen so that M.to_native maps tile pixels back to native pixels
            out.append({'i': i, 'j': j, 'img': t, 'origin_z0': (x0 * s, y0 * s), 'core_px': (a, b, c, e),
                        'z0': s, 'up': 1})
    return out, F


def run(model, ks=(1.0, 1.5, 2.5), run_idx=0):
    out = M.OUT / f'x3_{model}_run{run_idx}.jsonl'
    for did in DIDS:
        ann, img, tpl = M.load(did)
        for k in ks:
            ts, F = fov_tiles(ann, img, k)
            jobs = [{'model': model, 'coord': 'n1000', 'tpl': tpl, 'tile': t,
                     'meta': {'did': did, 'model': model, 'cell': f'k{k}', 'fov': F, 'i': t['i'], 'j': t['j'],
                              'decode': 'n1000'}} for t in ts]
            M.run_jobs(jobs, out, key=lambda r: (r['did'], r['cell'], r['i'], r['j']), workers=8)


if __name__ == '__main__':
    M.C.load_env('../config.env')
    if len(sys.argv) > 2 and sys.argv[2] == 'dry':
        for did in DIDS:
            ann, img, _ = M.load(did)
            print(did, [(k, fov_tiles(ann, img, k)[1], len(fov_tiles(ann, img, k)[0])) for k in (1.0, 1.5, 2.5)])
    else:
        ks = tuple(float(x) for x in sys.argv[2].split(',')) if len(sys.argv) > 2 else (1.0, 1.5, 2.5)
        run(sys.argv[1], ks=ks)
