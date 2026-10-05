#!/usr/bin/env python3
"""
Pre-registered main experiment on the 34 REDP-X40 test drawings (see claude/main_plan_and_preregistration.md).

usage: python3 main_test.py <cond> <model> [runs]
  cond: A1 | B | C | D | Dmask | H
Outputs: results_test/<cond>_<model>_run<r>.jsonl (resumable).
"""
import json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageDraw
import mech as M
import ceilings as K
from x_small import x2_tiles
from x3_fixedgrid import fov_tiles

TEST = K.TEST
OUT = Path('results_test'); OUT.mkdir(exist_ok=True)
TILE_IFACE = {'gpt': ('px', 'px'), 'claude': ('px', 'px_claude'), 'qwen3vl': ('n1000', 'n1000'),
              'gemini': ('n1000', 'n1000'), 'gemma4': ('n1000', 'n1000')}
FOV_K = {'gemini': 2.5, 'gemma4': 1.5}
XTILE = {'gpt', 'claude', 'qwen3vl'}
WORKERS = {'gpt': 6, 'claude': 6, 'qwen3vl': 8, 'gemini': 10, 'gemma4': 8}


def whole(model, run):
    """A1: whole drawing, long side <= 1600, E2 counts-first prompt, pixel coordinates requested."""
    sys.path.insert(0, 'vlm_api')
    import e2_redpx_order as E2
    out = OUT / f'A1_{model}_run{run}.jsonl'
    done = {json.loads(l)['did'] for l in out.open() if json.loads(l)['status'] in ('success', 'parse_error')} if out.exists() else set()

    def one(did):
        ann, img, tpl = M.load(did); W, H = img.size
        s = min(1.0, 1600 / max(W, H)); im = img.resize((round(W * s), round(H * s)), Image.LANCZOS) if s < 1 else img
        sysp, user = E2.PROMPTS['counts_first']; sysp = sysp.format(canvas_w=im.size[0], canvas_h=im.size[1])
        rec = {'did': did, 'model': model, 'run': run, 'scale': s, 'sent_size': list(im.size)}
        for attempt in range(3):
            try:
                raw, u = M.C.call(model, [tpl, im], sysp, user); p = M.C.parse(raw)
                sh = p.get('shapes', []) if isinstance(p, dict) else []
                rec.update(status='parse_error' if p.get('parse_error') else 'success', declared=p.get('count'),
                           shapes=sh if isinstance(sh, list) else [], usage=u, raw=raw); break
            except Exception as e:
                rec.update(status='api_error', error=f'{type(e).__name__}: {e}'[:300]); time.sleep(10 * 2 ** attempt)
        return rec
    with ThreadPoolExecutor(2) as ex, out.open('a') as f:
        for r in ex.map(one, [d for d in TEST if d not in done]):
            f.write(json.dumps(r) + '\n'); f.flush(); print('A1', model, run, r['did'], r['status'], len(r.get('shapes', [])), flush=True)


def vendor_highres(model, run):
    import x4_vendor_highres as X4, x4b_gemini_high as X4b
    out = OUT / f'B_{model}_run{run}.jsonl'
    done = {json.loads(l)['did'] for l in out.open() if json.loads(l)['status'] in ('success', 'parse_error')} if out.exists() else set()

    def one(did):
        for attempt in range(3):
            try:
                return X4.call('gpt_original', did, run) if model == 'gpt' else X4b.one((did, run))
            except Exception as e:
                err = f'{type(e).__name__}: {e}'[:300]; time.sleep(10 * 2 ** attempt)
        return {'did': did, 'run': run, 'status': 'api_error', 'error': err}
    with ThreadPoolExecutor(3) as ex, out.open('a') as f:
        for r in ex.map(one, [d for d in TEST if d not in done]):
            f.write(json.dumps(r) + '\n'); f.flush(); print('B', model, run, r['did'], r['status'], flush=True)


def mask_halves(ts):
    halves = []
    for t in ts:
        w, h = t['img'].size; mid = w // 2
        for half, (lo, hi) in (('L', (0, mid)), ('R', (mid, w))):
            im = t['img'].copy(); d = ImageDraw.Draw(im)
            if half == 'L':
                d.rectangle([mid, 0, w, h], fill=(255, 255, 255))
            else:
                d.rectangle([0, 0, mid - 1, h], fill=(255, 255, 255))
            a, b, c, e = t['core_px']; d.rectangle([a, b, c - 1, e - 1], outline=(0, 90, 255), width=3)
            halves.append(dict(t, img=im, meta={'half': half, 'keep_x': [lo, hi]}))
    return halves


def tiles(cond, model, run):
    coord, decode = TILE_IFACE[model]
    out = OUT / f'{cond}_{model}_run{run}.jsonl'
    key = lambda r: (r['did'], r['cell'], r['i'], r['j'], r.get('half', ''))
    for did in TEST:
        ann, img, tpl = M.load(did)
        if cond in ('C', 'Dmask') or (cond == 'D' and model in XTILE):
            ts, z = x2_tiles(ann, img, 1024); tp = M.scaled_template(tpl, z, 1); cell = 'T1024'
            if cond == 'Dmask':
                ts = mask_halves(ts); cell = 'T1024mask'
        else:
            ts, F = fov_tiles(ann, img, FOV_K[model]); tp = tpl; cell = f'k{FOV_K[model]}'
        del img
        jobs = []
        for t in ts:
            meta = {'did': did, 'model': model, 'cell': cell, 'cond': cond, 'run': run, 'i': t['i'], 'j': t['j'], 'decode': decode}
            meta.update(t.get('meta', {}))
            jobs.append({'model': model, 'coord': coord, 'tpl': tp, 'tile': t, 'meta': meta})
        M.run_jobs(jobs, out, key=key, workers=WORKERS[model])


def hybrid(model):
    import hybrid_verify as HV
    HV.M.OUT = OUT  # write hyb_<model>.jsonl into results_test
    HV.run([model], TEST)


if __name__ == '__main__':
    M.C.load_env('../config.env')
    cond, model = sys.argv[1], sys.argv[2]
    runs = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0]
    for r in runs:
        if cond == 'A1':
            whole(model, r)
        elif cond == 'B':
            vendor_highres(model, r)
        elif cond in ('C', 'D', 'Dmask'):
            tiles(cond, model, r)
        elif cond == 'H':
            hybrid(model)
