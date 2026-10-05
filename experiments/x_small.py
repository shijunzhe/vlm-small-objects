#!/usr/bin/env python3
"""
Small verification experiments (2026-10-01).

X1  Exemplar-scale confound in the whole-drawing condition.
    The whole-drawing protocol (E2) shows the legend crop at native size while the
    drawing is downscaled to 1600 px, so exemplar and instances differ in scale.
    X1 repeats the whole-drawing call with the legend crop resized by the same factor
    as the drawing (floor 16 px on the short side). Compare with the existing E2
    counts-first runs (native crop). Drawings d001-d010, GPT and Claude, 2 runs.

X2  Delivered tile size at fixed real detail.
    Symbols are resampled from the native drawing so that the legend's short side
    is 44 px (~1.5 GPT tokens), then the drawing is tiled with delivered tile size
    T in {512, 768, 1024} (core = 3/4 T, margin = T/8). T = 1536 at the same detail
    is the existing L2n cell. Same six drawings as step 2, GPT and Claude, 1 run.
    Compare with S2 (768 px, same geometry but upsampled from the 22 px version).
"""
import json, math, sys
from pathlib import Path
from PIL import Image, ImageDraw
import mech as M

sys.path.insert(0, 'vlm_api')
IFACE = {'gpt': ('px', 'px'), 'claude': ('px', 'px_claude')}


ZCAP = None   # FPC only: cap on the enlarged long side (px); one FPC-sheets item (6 px wide reference) would otherwise need a 22,000 px image


def x2_tiles(ann, img, T, sym_px=44):
    z = sym_px / min(ann['template_size'])
    if ZCAP is not None: z = min(z, ZCAP / max(img.size))
    m = T // 8; core = T - 2 * m
    W, H = img.size
    zimg = img.resize((max(1, round(W * z)), max(1, round(H * z))), Image.LANCZOS)
    Wz, Hz = zimg.size
    out = []
    for j in range(math.ceil(Hz / core)):
        for i in range(math.ceil(Wz / core)):
            cx0, cy0 = i * core, j * core
            cx1, cy1 = min(Wz, cx0 + core), min(Hz, cy0 + core)
            x0, y0 = max(0, cx0 - m), max(0, cy0 - m)
            x1, y1 = min(Wz, cx1 + m), min(Hz, cy1 + m)
            t = zimg.crop((x0, y0, x1, y1))
            d = ImageDraw.Draw(t)
            a, b, c, e = cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0
            d.rectangle([a, b, c - 1, e - 1], outline=(0, 90, 255), width=3)
            out.append({'i': i, 'j': j, 'img': t, 'origin_z0': (x0, y0), 'core_px': (a, b, c, e), 'z0': z, 'up': 1})
    return out, z


def run_x2(model, sizes=(512, 768, 1024), dids=('d001', 'd004', 'd008', 'd010', 'd017', 'd019'), run=0):
    coord, decode = IFACE[model]
    out = M.OUT / (f'x2_{model}.jsonl' if run == 0 else f'x2_{model}_run{run}.jsonl')
    for did in dids:
        ann, img, tpl = M.load(did)
        for T in sizes:
            ts, z = x2_tiles(ann, img, T)
            tplz = M.scaled_template(tpl, z, 1)
            jobs = [{'model': model, 'coord': coord, 'tpl': tplz, 'tile': t,
                     'meta': {'did': did, 'model': model, 'cell': f'T{T}', 'i': t['i'], 'j': t['j'], 'decode': decode}}
                    for t in ts]
            M.run_jobs(jobs, out, key=lambda r: (r['did'], r['cell'], r['i'], r['j']), workers=6)


def run_x1(model, runs=2):
    import arr_common as C, e2_redpx_order as E2
    from concurrent.futures import ThreadPoolExecutor
    out = M.OUT / f'x1_{model}.jsonl'
    done = set()
    if out.exists():
        for l in out.open():
            r = json.loads(l)
            if r['status'] == 'success':
                done.add((r['did'], r['run']))
    jobs = [(f'd{i:03d}', k) for i in range(1, 11) for k in range(runs) if (f'd{i:03d}', k) not in done]

    def one(job):
        did, run = job
        ann, img, tpl = M.load(did)
        W, H = img.size
        s = min(1.0, 1600 / max(W, H))
        im = img.resize((round(W * s), round(H * s)), Image.LANCZOS) if s < 1 else img
        ts = max(s, 16 / min(tpl.size))
        tpl_s = tpl.resize((max(1, round(tpl.width * ts)), max(1, round(tpl.height * ts))), Image.LANCZOS)
        sysp, user = E2.PROMPTS['counts_first']
        sysp = sysp.format(canvas_w=im.size[0], canvas_h=im.size[1])
        raw, usage = C.call(model, [tpl_s, im], sysp, user)
        p = C.parse(raw)
        sh = p.get('shapes', []) if isinstance(p, dict) else []
        return {'did': did, 'run': run, 'scale': s, 'tpl_scale': ts, 'status': 'parse_error' if p.get('parse_error') else 'success',
                'declared': p.get('count'), 'shapes': sh, 'usage': usage, 'raw': raw}

    with ThreadPoolExecutor(6) as ex, out.open('a') as f:
        for r in ex.map(one, jobs):
            f.write(json.dumps(r) + '\n'); f.flush()
            print(r['did'], r['run'], r['status'], len(r['shapes']), flush=True)


if __name__ == '__main__':
    M.C.load_env('../config.env')
    exp, model = sys.argv[1], sys.argv[2]
    if exp == 'x1':
        run_x1(model)
    else:
        sizes = tuple(int(x) for x in sys.argv[3].split(',')) if len(sys.argv) > 3 else (512, 768, 1024)
        run = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        run_x2(model, sizes=sizes, run=run)
