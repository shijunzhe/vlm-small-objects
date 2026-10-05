#!/usr/bin/env python3
"""
Phase 0 + mechanism checks on the dev set (d001, d004, d008, d010, d017, d019).

  qwen      : Qwen3-VL, recipe tiles (1024 px, symbol 44 px), 0-1000 coordinates (native convention)
  gpt_bbox  : GPT-5.4, recipe tiles, bounding boxes in 0..999 (OpenAI cookbook convention)
  pad       : 384-px tiles (cell S1) pasted at the top-left of a white 768x768 canvas;
              same pixels, larger image -> does recall recover? (GPT, Claude)
  mask      : recipe tiles (1024 px) with the left or the right half whitened; each half queried
              separately -> less content per call, same pixels per region (GPT, Claude)
  nobias    : Claude, 512-px tiles (where empty-tile false positives were highest) with one added
              sentence: most tiles contain no instance; report 0 if unsure
  ruler     : Gemini, fixed-grid FOV tiles (k=1.5) with numbered 0-1000 rulers on the top and left edges
"""
import sys
from PIL import Image, ImageDraw
import mech as M
from x_small import x2_tiles
from x3_fixedgrid import fov_tiles

DIDS = ('d001', 'd004', 'd008', 'd010', 'd017', 'd019')
IFACE = {'gpt': ('px', 'px'), 'claude': ('px', 'px_claude'), 'gemini': ('n1000', 'n1000'),
         'gemma4': ('n1000', 'n1000'), 'qwen3vl': ('n1000', 'n1000')}
NOBIAS = ("\nMost tiles contain no instance of the target symbol. If you are not sure that an "
          "instance is present, do not report it; an empty list is a valid answer.\n")


def jobs_for(model, did, cell, tiles, tpl, coord=None, decode=None, system_fn=None, extra=None):
    c, d = IFACE[model]
    coord, decode = coord or c, decode or d
    out = []
    for t in tiles:
        meta = {'did': did, 'model': model, 'cell': cell, 'i': t['i'], 'j': t['j'], 'decode': decode}
        meta.update(t.get('meta', {}))
        job = {'model': model, 'coord': coord, 'tpl': tpl, 'tile': t, 'meta': meta}
        if system_fn:
            w, h = t['img'].size
            job['system'] = system_fn(coord, w, h)
        out.append(job)
    return out


def run(exp, model):
    out = M.OUT / f'p0_{exp}_{model}.jsonl'
    key = lambda r: (r['did'], r['cell'], r['i'], r['j'], r.get('half', ''))
    for did in DIDS:
        ann, img, tpl = M.load(did)
        jobs = []
        if exp in ('qwen', 'gpt_bbox', 'mask'):
            ts, z = x2_tiles(ann, img, 1024)
            tplz = M.scaled_template(tpl, z, 1)
            if exp == 'qwen':
                jobs = jobs_for(model, did, 'T1024', ts, tplz)
            elif exp == 'gpt_bbox':
                jobs = jobs_for(model, did, 'T1024', ts, tplz, coord='bbox999', decode='bbox999')
            else:
                halves = []
                for t in ts:
                    w, h = t['img'].size
                    mid = w // 2
                    for half, (lo, hi) in (('L', (0, mid)), ('R', (mid, w))):
                        im = t['img'].copy()
                        d = ImageDraw.Draw(im)
                        if half == 'L':
                            d.rectangle([mid, 0, w, h], fill=(255, 255, 255))
                        else:
                            d.rectangle([0, 0, mid - 1, h], fill=(255, 255, 255))
                        a, b, c, e = t['core_px']   # redraw the core frame on top of the mask
                        d.rectangle([a, b, c - 1, e - 1], outline=(0, 90, 255), width=3)
                        halves.append(dict(t, img=im, meta={'half': half, 'keep_x': [lo, hi]}))
                jobs = jobs_for(model, did, 'T1024mask', halves, tplz)
        elif exp == 'pad':
            ts = M.tiles_for(ann, img, 'S', 1)
            padded = []
            for t in ts:
                can = Image.new('RGB', (768, 768), 'white')
                can.paste(t['img'], (0, 0))
                padded.append(dict(t, img=can))
            jobs = jobs_for(model, did, 'S1pad', padded, M.scaled_template(tpl, ts[0]['z0'], 1))
        elif exp == 'nobias':
            ts, z = x2_tiles(ann, img, 512)
            jobs = jobs_for(model, did, 'T512nobias', ts, M.scaled_template(tpl, z, 1),
                            system_fn=lambda c, w, h: M.system_prompt(c, w, h) + NOBIAS)
        elif exp == 'ruler':
            ts, F = fov_tiles(ann, img, 1.5)
            ruled = []
            for t in ts:
                im = t['img'].copy(); d = ImageDraw.Draw(im); w, h = im.size
                for k in range(0, 1001, 100):
                    x = round(k / 1000 * (w - 1)); y = round(k / 1000 * (h - 1))
                    d.line([x, 0, x, 8], fill=(200, 0, 0), width=1); d.line([0, y, 8, y], fill=(200, 0, 0), width=1)
                    if k % 200 == 0:
                        d.text((min(x + 1, w - 22), 9), str(k), fill=(200, 0, 0))
                        d.text((10, min(y + 1, h - 10)), str(k), fill=(200, 0, 0))
                ruled.append(dict(t, img=im))
            jobs = jobs_for(model, did, 'k1.5ruler', ruled, tpl,
                            system_fn=lambda c, w, h: M.system_prompt(c, w, h) +
                            "\nRed tick marks with numbers along the top and left edges show the 0-1000 coordinate scale.\n")
        for j in jobs:   # keep_x / half live in meta and must reach the stored record
            j['meta'] = dict(j['meta'])
        M.run_jobs(jobs, out, key=key, workers=8 if model in ('gemini', 'qwen3vl', 'gemma4') else 6)


if __name__ == '__main__':
    M.C.load_env('../config.env')
    run(sys.argv[1], sys.argv[2])
