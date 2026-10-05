#!/usr/bin/env python3
"""
X8: falsification test on FSC-147 (natural images, 384-px versions; 150 test/val images in three object-size strata).
Conditions per image (reference = first exemplar box crop):
  W : native image
  U : whole image upsampled so the long side is 1536 px (no new information; more tokens per object for
      resolution-scaled interfaces, unchanged for fixed-grid Gemini)
  T : 2x2 tiles (core = image quarter, margin 1/8 of the side), each upsampled by the same factor as U
      (same pixels per object as U, about a quarter of the instances per call)
Geometric predictions: U - W > 0 only where S(W) is low (small objects); fixed-grid Gemini U - W ~ 0;
T - U (capacity) grows with the number of instances. Large objects (S >= 2): no gain from U or T.
usage: python3 fsc_run.py run <vendor> <cond> | score
"""
import json, math, os, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import mech as M
import syn as SY

ROOT = Path('../data/fsc147'); OUT = ROOT / 'results'
ITEMS = json.load(open(ROOT / 'x8_items.json'))
_SUB = ROOT / 'x8_ann_subset.json'
if not _SUB.exists():
    _all = json.load(open(ROOT / 'annotation_FSC147_384.json')); json.dump({it['name']: _all[it['name']] for it in ITEMS}, open(_SUB, 'w')); del _all
ANN = json.load(open(_SUB))
UP_L = 1536
PROMPT = os.environ.get('FSC_PROMPT', 'symbol')   # 'object': domain-neutral wording (v3.2 deviation, declared before running)
SYS_OBJ = """You are a precise visual counting assistant.

You will be shown an exemplar crop of ONE object, followed by an image.
Find every object in the image that is of the same kind as the exemplar (the exemplar itself is one of them).
Objects of the same kind may differ slightly in pose, colour, or size.

{coord}

Respond with ONLY a JSON object (no markdown, no commentary). Format:
{{"target_type": "object", "count": <integer>, "shapes": [{{"type": "object", {fmt}}}, ...]}}
The "count" must equal the length of "shapes". Give the centre of each object."""
USER_OBJ = "The first image is the exemplar crop; the second is the image. Find the centre of every object of the same kind. JSON only."
SFX = '' if PROMPT == 'symbol' else '_obj'
CONV = {'gpt56': 'px', 'sonnet55': 'px', 'gem38': 'n1000'}


def reference(name):
    im = Image.open(ROOT / 'images' / name).convert('RGB'); b = ANN[name]['box_examples_coordinates'][0]
    xs, ys = [p[0] for p in b], [p[1] for p in b]
    return im.crop((min(xs), min(ys), max(xs), max(ys)))


def views(name, cond):
    """[(image, (ox, oy), scale, core_box_native)] ; core in native coordinates."""
    im = Image.open(ROOT / 'images' / name).convert('RGB'); W, H = im.size
    u = UP_L / max(W, H)
    if cond == 'W':
        return [(im, (0, 0), 1.0, (0, 0, W, H))]
    if cond == 'U':
        return [(im.resize((round(W * u), round(H * u)), Image.LANCZOS), (0, 0), u, (0, 0, W, H))]
    out = []; mx, my = W / 8, H / 8
    for i in range(2):
        for j in range(2):
            core = (i * W / 2, j * H / 2, (i + 1) * W / 2, (j + 1) * H / 2)
            box = (max(0, round(core[0] - mx)), max(0, round(core[1] - my)), min(W, round(core[2] + mx)), min(H, round(core[3] + my)))
            t = im.crop(box); out.append((t.resize((round(t.width * u), round(t.height * u)), Image.LANCZOS), box[:2], u, core))
    return out


def run(vendor, cond):
    out = OUT / f'{vendor}_{cond}{SFX}.jsonl'
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            if r['status'] in ('success', 'parse_error'): done.add((r['name'], r['v']))
    conv = CONV[vendor]; t, f, _ = M.COORD_TEXT['px' if conv == 'px' else 'n1000']; lock = threading.Lock()
    jobs = []
    for it in ITEMS:
        for k, (img, org, sc, core) in enumerate(views(it['name'], cond)):
            if (it['name'], k) not in done: jobs.append((it, k, img, org, sc, core))

    def one(job):
        it, k, img, org, sc, core = job
        base = SY.SYS if PROMPT == 'symbol' else SYS_OBJ
        ct = t.format(w=img.width, h=img.height)
        ct = ct.replace('tile', 'image') if PROMPT == 'symbol' else ct.replace('tile image', 'image').replace('tile', 'image')
        sysp = base.format(coord=ct, fmt=f)
        rec = {'name': it['name'], 'v': k, 'cond': cond, 'vendor': vendor, 'size': list(img.size), 'origin': list(org), 'scale': sc, 'core': list(core)}
        for a in range(3):
            try:
                raw, u = SY.call(vendor, [reference(it['name']), img], sysp, SY.USER if PROMPT == 'symbol' else USER_OBJ); p = M.C.parse(raw)
                sh = p.get('shapes', []) if isinstance(p, dict) else []
                rec.update(status='parse_error' if p.get('parse_error') else 'success', shapes=sh if isinstance(sh, list) else [], usage=u, raw=raw[:6000]); break
            except Exception as e:
                rec.update(status='api_error', error=str(e)[:200]); time.sleep(5 * 2 ** a)
        with lock, out.open('a') as fo:
            fo.write(json.dumps(rec) + '\n')
        return rec
    print(vendor, cond, len(jobs), 'calls to go', flush=True)
    with ThreadPoolExecutor(6) as ex:
        for r in ex.map(one, jobs):
            if r['status'] != 'success': print(' ', r['name'], r['v'], r['status'], r.get('error', '')[:100], flush=True)


def native_points(rec):
    W, H = rec['size']; raw = [p for p in (M.point_of(q) for q in rec.get('shapes', [])) if p is not None]
    if not raw and rec['status'] == 'parse_error' and not os.environ.get('FSC_STRICT'): raw = M.lenient_points(rec.get('raw'))
    if CONV[rec['vendor']] == 'n1000': raw = [(x / 1000 * W, y / 1000 * H) for x, y in raw]
    pts = []
    for x, y in raw:
        X, Y = rec['origin'][0] + x / rec['scale'], rec['origin'][1] + y / rec['scale']
        a, b, c, d = rec['core']
        if a <= X < c and b <= Y < d: pts.append((X, Y))
    return pts


def token_side(vendor, cond, W, H):
    u = UP_L / max(W, H)
    if vendor == 'gem38':
        f = 1 if cond != 'T' else 0.5 + 1 / 8   # tile side relative to the image side (with margins)
        return math.sqrt((W * f) * (H * f) / 1090)
    t = 32 if vendor == 'gpt56' else 28
    return t / (1 if cond == 'W' else u)


def score():
    res = {}
    for f in sorted(OUT.glob(f'*_?{SFX}.jsonl')):
        vendor, cond = f.stem[:len(f.stem) - len(SFX)].rsplit('_', 1)
        recs = {}
        for l in open(f):
            r = json.loads(l)
            if r['status'] in ('success', 'parse_error'): recs[(r['name'], r['v'])] = r
        for it in ITEMS:
            nv = 4 if cond == 'T' else 1
            if not all((it['name'], k) in recs for k in range(nv)): continue
            pts = sum((native_points(recs[(it['name'], k)]) for k in range(nv)), [])
            gt = [tuple(p) for p in ANN[it['name']]['points']]
            tp, n_p, n_g = M.match(pts, gt, it['m'] / 2)
            im = Image.open(ROOT / 'images' / it['name'])
            res[(vendor, cond, it['name'])] = {'f1': 2 * tp / (n_p + n_g), 'recall': tp / n_g, 'abs_err': abs(n_p - n_g), 'rel_err': abs(n_p - n_g) / n_g,
                                               'S': it['m'] / token_side(vendor, cond, *im.size), 'n': n_g, 'stratum': it['stratum']}
    json.dump({'|'.join(k): v for k, v in res.items()}, open(ROOT / f'x8_scores{SFX}.json', 'w'), indent=1)
    rng = np.random.default_rng(0)
    def boot(x):
        x = np.array(x); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(4000)]; return x.mean(), *np.percentile(bs, [2.5, 97.5])
    summ = {}
    print('vendor   stratum  medS(W)  F1 W / U / T      relErr W / U / T     U-W F1 [CI]            T-U F1 [CI]            T-W F1 [CI]')
    for v in CONV:
        for st in ('small', 'medium', 'large'):
            names = [it['name'] for it in ITEMS if it['stratum'] == st and all((v, c, it['name']) in res for c in 'WUT')]
            if not names: continue
            g = lambda c, k: [res[(v, c, n)][k] for n in names]
            d = lambda a, b: boot(np.array(g(a, 'f1')) - np.array(g(b, 'f1')))
            uw, tu, tw = d('U', 'W'), d('T', 'U'), d('T', 'W')
            summ[f'{v}|{st}'] = {'f1': {c: float(np.mean(g(c, 'f1'))) for c in 'WUT'}, 'rel': {c: float(np.mean(g(c, 'rel_err'))) for c in 'WUT'},
                                 'UW': list(uw), 'TU': list(tu), 'TW': list(tw), 'S': float(np.median(g('W', 'S'))), 'n': len(names), 'count': int(np.median(g('W', 'n')))}
            print(f'{v:8s} {st:7s} {np.median(g("W", "S")):5.2f}   {np.mean(g("W","f1")):.2f}/{np.mean(g("U","f1")):.2f}/{np.mean(g("T","f1")):.2f}     '
                  f'{np.mean(g("W","rel_err")):.2f}/{np.mean(g("U","rel_err")):.2f}/{np.mean(g("T","rel_err")):.2f}      '
                  f'{uw[0]:+.2f} [{uw[1]:+.2f},{uw[2]:+.2f}]   {tu[0]:+.2f} [{tu[1]:+.2f},{tu[2]:+.2f}]   {tw[0]:+.2f} [{tw[1]:+.2f},{tw[2]:+.2f}]  n={len(names)}')
    json.dump(summ, open(ROOT / f'x8_summary{SFX}{"_strict" if os.environ.get("FSC_STRICT") else ""}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        M.C.load_env('../config.env'); run(sys.argv[2], sys.argv[3])
    else:
        score()
