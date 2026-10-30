#!/usr/bin/env python3
"""S = 1.5 load sweep (supplement L), rerun 2026-10-28 after the original answers (visual-spec record 14) were lost.

Protocol (as recorded in the rule-of-thumb reference, section 6):
  * scenes of the visual-spec family F0 (vspec.py): n targets of one legend symbol and n distractors of three other
    symbols, lattice pitch 1.6 symbol units, grey linework; layouts in symbol units, shared by all models for (i, n);
  * S = 1.5 for every model; canvas capped by the per-view budget of the rule
    (GPT 8,192 patches -> 2,896 px; Claude 4,784 tokens -> 1,932 px; Gemini fixed 1,024 canvas; Qwen 1,600 px);
  * grey-preserving paste: each pixel takes the darker of canvas and Lanczos-reduced symbol;
  * 12 images per load; reasoning settings of the main experiments (gen_common.py), output cap 16,384 for all;
  * prompt and reader of the visual-spec experiment (vrun.py, reader v2).
Layout set name 'LS15' (new seeds; the lost run used other layouts).
usage: ls_run.py build | run <model> [workers] | status
(build is deterministic: it regenerates images/ and items.json exactly; run makes model calls and needs keys)
"""
import json, math, os, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import vspec as V
V.MODELS['gpt61'] = dict(V.MODELS['gpt61'], maxside=2896)
import vrun as R, vcall as VC

HERE = Path(__file__).resolve().parent; IMG = HERE / 'images'; RES = HERE.parent / 'results_ls'
SET, S, K = 'LS15', 1.5, 12
LOADS = {'gpt61': [8, 16, 32, 64, 128, 256, 324], 'gpt56': [8, 16, 32, 64, 128, 256, 324],
         'opus55': [8, 16, 32, 64, 128, 196], 'gem4': [8, 16, 32, 42], 'gem38': [8, 16, 32, 42], 'qwen3vl': [8, 16, 32, 64]}
MAX_OUT = 16384


def load_env():
    for line in (HERE.parents[1] / 'config.env').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    VC._env_loaded = True


def render_grey(lay, model):
    r = V.render(lay, model, S, check_only=True)
    if 'geo' not in r:
        return r
    g = V.MODELS[model]; m = S * g['t']; W, H = r['geo']['canvas']; ox, oy = r['geo']['offset']
    can = Image.new('L', (W, H), 255); dr = ImageDraw.Draw(can); lw = max(1, round(m / 40))
    for (x0, y0, x1, y1) in lay['lines']:
        dr.line([ox + x0 * m, oy + y0 * m, ox + x1 * m, oy + y1 * m], fill=150, width=lw)
    a = np.asarray(can).copy(); pts = []
    for (d, is_t, ux, uy) in lay['objs']:
        im, mask, ib = V.scaled(d, m)
        px, py = round(ox + ux * m - im.width / 2), round(oy + uy * m - im.height / 2)
        s = np.asarray(im.convert('L'))
        x0, y0 = max(0, px), max(0, py); x1, y1 = min(W, px + im.width), min(H, py + im.height)
        a[y0:y1, x0:x1] = np.minimum(a[y0:y1, x0:x1], s[y0 - py:y1 - py, x0 - px:x1 - px])
        if is_t:
            pts.append([px + im.width / 2, py + im.height / 2])
    return {'img': Image.fromarray(a).convert('RGB'), 'points': pts, 'geo': r['geo']}


def build():
    items, laylog = [], []
    allns = sorted({n for ns in LOADS.values() for n in ns})
    for n in allns:
        renders = [(m, S) for m, ns in LOADS.items() if n in ns]
        for i in range(K):
            lay, k = V.frozen_layout(SET, i, n, renders, log=laylog)
            assert lay is not None, (i, n)
            for (m, _) in renders:
                r = render_grey(lay, m); assert 'img' in r, (m, i, n, r)
                iid = f'ls_{m}_{SET}_i{i}_n{n}'
                d = IMG / m; d.mkdir(parents=True, exist_ok=True); r['img'].save(d / f'{iid}.png')
                W, H = r['geo']['canvas']
                items.append(dict(id=iid, model=m, set=SET, i=i, n=n, k=k, S=S, target=lay['target'], distractors=lay['distractors'],
                                  points=r['points'], geo=r['geo'], exp_img_tokens=V.expected_tokens(m, W, H), image=str((d / f'{iid}.png').relative_to(HERE))))
    RES.mkdir(exist_ok=True); json.dump(items, open(RES / 'items.json', 'w')); json.dump(laylog, open(RES / 'layouts.json', 'w'), indent=0)
    print(len(items), 'items')


def call(model, images, system, user):
    if model == 'qwen3vl':
        VC.MAX_OUT = MAX_OUT
        return VC.call(model, images, system, user)
    import gen_common as G
    G.MAX_TOKENS = MAX_OUT
    raw, u = G.call(model, images, system, user)
    return raw, u


def run(model, workers=4):
    load_env()
    items = [it for it in json.load(open(RES / 'items.json')) if it['model'] == model]
    out = RES / f'LS_{model}.jsonl'; RES.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        for l in out.open():
            r = json.loads(l)
            if r['status'] == 'success':
                done.add(r['id'])
    todo = [it for it in items if it['id'] not in done]
    print(model, len(done), 'done', len(todo), 'to go', flush=True)
    lock = threading.Lock()

    def one(it):
        img = Image.open(HERE / it['image']).convert('RGB'); ref = V.tpl(it['target']); W, H = img.size
        rec = {'id': it['id'], 'model': model}
        for a in range(4):
            try:
                t0 = time.time()
                raw, u = call(model, [ref, img], R.system_prompt(model, W, H), R.USER)
                rec.update(status='success', raw=raw, usage=u, latency=time.time() - t0, ts=time.time(),
                           truncated=VC.truncated(model, u), cost=VC.cost(model, u))
                if model == 'qwen3vl' and u.get('provider') not in (None, VC.QWEN_PROVIDER):
                    rec.update(status='provider_mismatch')
                break
            except Exception as e:
                rec.update(status='api_error', error=f'{type(e).__name__}: {e}'[:400]); time.sleep(10 * 2 ** a)
        with lock:
            with out.open('a') as f:
                f.write(json.dumps(rec) + '\n')
        print(it['id'], rec['status'], rec.get('usage', {}).get('output_tokens'), round(rec.get('cost', 0), 4), rec.get('error', '')[:150], flush=True)
        return rec

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, todo))


def status():
    items = json.load(open(RES / 'items.json'))
    for m in LOADS:
        p = RES / f'LS_{m}.jsonl'
        recs = [json.loads(l) for l in p.open()] if p.exists() else []
        ok = {r['id'] for r in recs if r['status'] == 'success'}
        print(m, 'items', sum(it['model'] == m for it in items), 'success', len(ok), 'records', len(recs),
              'cost $%.2f' % sum(r.get('cost', 0) for r in recs if r['status'] == 'success'))


if __name__ == '__main__':
    a = sys.argv[1]
    if a == 'build': build()
    elif a == 'run': run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    else: status()
