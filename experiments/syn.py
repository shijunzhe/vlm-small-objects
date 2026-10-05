#!/usr/bin/env python3
"""
X1: controlled synthetic stimuli built from the 40 REDP-X40 legend templates.
  res : 1024x1024 canvas, 8 targets + 8 distractors, symbol scale m in {8,12,16,24,32,48,64} px, 6 images per level
  cap : 1024x1024 canvas, symbol scale 40 px, n targets in {2,4,8,16,32,64,96} + n distractors, 6 images per level
Light clutter (thin grey line segments) emulates drawing linework. The reference is the native legend template.
usage: python3 syn.py build | run <vendor>[,<vendor>] | score
"""
import json, math, os, random, sys, threading, time, zlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageDraw
import mech as M

ROOT = Path('syn'); IMG = ROOT / 'images'; IMG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'results'; OUT.mkdir(exist_ok=True)
DIDS = [f'd{i:03d}' for i in range(1, 41)]
RES_SIZES = [8, 12, 16, 24, 32, 48, 64]
CAP_NS = [2, 4, 8, 16, 32, 64, 96]
CANVAS = 1024
AREA_CANVAS = [(1024, 1024), (1448, 1448), (2048, 2048), (2896, 2896), (4096, 2048)]


def scaled(tpl, m):
    w, h = tpl.size; f = m / math.sqrt(w * h)
    return tpl.resize((max(1, round(w * f)), max(1, round(h * f))), Image.LANCZOS)


def render(target, distractors, m, n_t, n_d, seed, W=1024, H=1024, clutter=40):
    rng = random.Random(seed)
    can = Image.new('RGB', (W, H), 'white'); d = ImageDraw.Draw(can)
    for _ in range(round(clutter * W * H / 1024 ** 2)):  # clutter proportional to area
        x0, y0 = rng.randrange(W), rng.randrange(H)
        if rng.random() < 0.5:
            d.line([x0, y0, min(W - 1, x0 + rng.randrange(50, 400)), y0], fill=(150, 150, 150), width=1)
        else:
            d.line([x0, y0, x0, min(H - 1, y0 + rng.randrange(50, 400))], fill=(150, 150, 150), width=1)
    pitch = max(int(1.6 * m), m + 6)
    slots = [(x, y) for x in range(pitch // 2 + 4, W - pitch // 2 - 4, pitch) for y in range(pitch // 2 + 4, H - pitch // 2 - 4, pitch)]
    rng.shuffle(slots)
    assert len(slots) >= n_t + n_d, (m, n_t, n_d, len(slots))
    tt = scaled(target, m); pts = []
    for k, (x, y) in enumerate(slots[:n_t + n_d]):
        jx, jy = rng.randint(-pitch // 6, pitch // 6), rng.randint(-pitch // 6, pitch // 6)
        sym = tt if k < n_t else scaled(distractors[k % len(distractors)], m)
        cx, cy = x + jx, y + jy
        can.paste(sym, (cx - sym.width // 2, cy - sym.height // 2), mask=sym.convert('L').point(lambda v: 255 if v < 200 else 0))
        if k < n_t:
            pts.append([cx, cy])
    return can, pts


def build():
    tpls = {d: Image.open(M.BENCH / f'templates/{d}.png').convert('RGB') for d in DIDS}
    items = []
    rng = random.Random(0)
    for exp, levels in (('res', RES_SIZES), ('cap', CAP_NS)):
        for lv in levels:
            for k in range(6):
                tgt = rng.choice(DIDS); dis = rng.sample([d for d in DIDS if d != tgt], 3)
                m, n_t = (lv, 8) if exp == 'res' else (40, lv)
                img, pts = render(tpls[tgt], [tpls[x] for x in dis], m, n_t, n_t, seed=zlib.crc32(f'{exp}|{lv}|{k}'.encode()))
                iid = f'{exp}_{lv}_{k}'
                img.save(IMG / f'{iid}.png')
                items.append({'id': iid, 'exp': exp, 'level': lv, 'm': m, 'n': n_t, 'target': tgt, 'distractors': dis, 'points': pts, 'canvas': [CANVAS, CANVAS]})
    for (W, H) in AREA_CANVAS:   # X1c: same targets per call and symbol size, more image area (and clutter) per call
        for k in range(6):
            tgt = rng.choice(DIDS); dis = rng.sample([d for d in DIDS if d != tgt], 3)
            img, pts = render(tpls[tgt], [tpls[x] for x in dis], 40, 16, 16, seed=zlib.crc32(f'area|{W}|{k}'.encode()), W=W, H=H)
            iid = f'area_{W}x{H}_{k}'; img.save(IMG / f'{iid}.png')
            items.append({'id': iid, 'exp': 'area', 'level': f'{W}x{H}', 'm': 40, 'n': 16, 'target': tgt, 'distractors': dis, 'points': pts, 'canvas': [W, H]})
    json.dump(items, open(ROOT / 'items.json', 'w'), indent=1)
    print(len(items), 'items')


UP_SIZES, UP_FACTOR = [8, 12, 16], 2
CLUTTER = [0, 40, 160, 640]


def build_extra():
    """X1d (clutter) and X1e (information-fixed upsampling). Appends to items.json without touching existing items.
    X1e: the res-sweep images at m in {8,12,16} upsampled x2 (Lanczos) to 2048^2: identical information, twice the
    pixels per symbol. Resolution-scaled interfaces then see ~2x the tokens per symbol; fixed-grid interfaces see the
    same tokens per symbol (prediction: gain for the former only).
    X1d: m=16, 8 targets + 8 distractors on 1024^2, clutter density in {0, 40, 160, 640} line segments."""
    items = json.load(open(ROOT / 'items.json')); have = {it['id'] for it in items}
    tpls = {d: Image.open(M.BENCH / f'templates/{d}.png').convert('RGB') for d in DIDS}
    for it in [x for x in items if x['exp'] == 'res' and x['level'] in UP_SIZES]:
        iid = f"up{UP_FACTOR}_{it['level']}_{it['id'].rsplit('_', 1)[1]}"
        if iid in have: continue
        im = Image.open(IMG / f"{it['id']}.png").convert('RGB')
        im.resize((im.width * UP_FACTOR, im.height * UP_FACTOR), Image.LANCZOS).save(IMG / f'{iid}.png')
        items.append({**it, 'id': iid, 'exp': 'up', 'm': it['m'] * UP_FACTOR, 'points': [[x * UP_FACTOR, y * UP_FACTOR] for x, y in it['points']],
                      'canvas': [CANVAS * UP_FACTOR] * 2, 'source': it['id'], 'native_m': it['m']})
    rng = random.Random(7)
    for c in CLUTTER:
        for k in range(6):
            iid = f'clut_{c}_{k}'
            tgt = rng.choice(DIDS); dis = rng.sample([d for d in DIDS if d != tgt], 3)
            if iid in have: continue
            img, pts = render(tpls[tgt], [tpls[x] for x in dis], 16, 8, 8, seed=zlib.crc32(f'clut|{c}|{k}'.encode()), clutter=c)
            img.save(IMG / f'{iid}.png')
            items.append({'id': iid, 'exp': 'clut', 'level': c, 'm': 16, 'n': 8, 'target': tgt, 'distractors': dis, 'points': pts, 'canvas': [CANVAS, CANVAS]})
    json.dump(items, open(ROOT / 'items.json', 'w'), indent=1)
    print(len(items), 'items')


SYS = """You are a precise engineering drawing analysis assistant.

You will be shown a reference image of ONE target symbol, followed by a drawing.
Find every instance of that target symbol in the drawing. Other symbols and linework are distractors.

{coord}

Respond with ONLY a JSON object (no markdown, no commentary). Format:
{{"target_type": "target_symbol", "count": <integer>, "shapes": [{{"type": "target_symbol", {fmt}}}, ...]}}
The "count" must equal the length of "shapes". Count only symbols matching the reference."""
USER = "The first image is the reference symbol; the second is the drawing. Find every instance of the reference symbol. JSON only."
CONV = {'gpt54': 'px', 'sonnet46': 'px_claude', 'gemini25': 'n1000', 'gemma4': 'n1000', 'qwen3vl': 'n1000',
        'gpt56': 'px', 'sonnet55': 'px_claude55', 'gem38': 'n1000', 'gem38lo': 'n1000', 'gem38uh': 'n1000'}
VKEY = {'gpt54': 'gpt', 'sonnet46': 'claude', 'gemini25': 'gemini', 'gemma4': 'gemma4', 'qwen3vl': 'qwen3vl',
        'gpt56': 'gpt56', 'sonnet55': 'sonnet55', 'gem38': 'gem38', 'gem38lo': 'gem38lo', 'gem38uh': 'gem38uh', 'opus55': 'opus55', 'gpt61': 'gpt61'}


def call(vendor, images, system, user):
    import gen_common as G
    G.REASONING['gemini-3.8-flash'] = {'thinking_budget': 0}
    k = VKEY[vendor]
    return G.call(k, images, system, user) if k in G.VENDORS else M.C.call(k, images, system, user)


def run(vendor, run_id=0):
    items = json.load(open(ROOT / 'items.json'))
    out = OUT / f'{vendor}_run{run_id}.jsonl'
    done = {json.loads(l)['id'] for l in out.open() if json.loads(l)['status'] in ('success', 'parse_error')} if out.exists() else set()
    conv = CONV[vendor]; coord = 'px' if conv.startswith('px') else 'n1000'
    tpls = {d: Image.open(M.BENCH / f'templates/{d}.png').convert('RGB') for d in DIDS}
    lock = threading.Lock()

    def one(it):
        img = Image.open(IMG / f"{it['id']}.png").convert('RGB')
        t, f, desc = M.COORD_TEXT[coord]
        cw, ch = it.get('canvas', [CANVAS, CANVAS])
        sysp = SYS.format(coord=t.format(w=cw, h=ch).replace('tile', 'image'), fmt=f)
        rec = {'id': it['id'], 'vendor': vendor, 'run': run_id}
        for a in range(3):
            try:
                raw, u = call(vendor, [tpls[it['target']], img], sysp, USER); p = M.C.parse(raw)
                sh = p.get('shapes', []) if isinstance(p, dict) else []
                rec.update(status='parse_error' if p.get('parse_error') else 'success', shapes=sh if isinstance(sh, list) else [], usage=u, raw=raw[:4000]); break
            except Exception as e:
                rec.update(status='api_error', error=str(e)[:200]); time.sleep(5 * 2 ** a)
        with lock, out.open('a') as fo:
            fo.write(json.dumps(rec) + '\n')
        return rec
    only = os.environ.get('SYN_ONLY')
    todo = [it for it in items if it['id'] not in done and (not only or it['exp'] == only)]
    print(vendor, len(todo), 'to go', flush=True)
    with ThreadPoolExecutor(6) as ex:
        for r in ex.map(one, todo):
            if r['status'] != 'success':
                print(' ', r['id'], r['status'], r.get('error', '')[:100], flush=True)


READING = os.environ.get('SYN_READING', 'robust')   # 'registered': strict JSON, fixed frame; 'robust': lenient + frame rule


def decode(vendor, shapes, canvas=(1024, 1024), raw=None, iid=None, m=None):
    conv = CONV[vendor]; CW, CH = canvas
    raw_pts = [pt for pt in (M.point_of(q) for q in shapes) if pt is not None]
    if not raw_pts and raw is not None and READING == 'robust':
        raw_pts = M.lenient_points(raw)
    if conv == 'px_claude55' and READING == 'robust':
        fx, fy = M.claude55_scale_ink(raw_pts, Image.open(IMG / f'{iid}.png'), m) if iid else M.claude55_scale(raw_pts, CW, CH)
    elif conv == 'px_claude55':
        rw, rh = M.claude_resized(CW, CH, 2576, 4784); fx, fy = CW / rw, CH / rh
    elif conv == 'px_claude':
        rw, rh = M.claude_resized(CW, CH); fx, fy = CW / rw, CH / rh
    elif conv == 'n1000':
        fx, fy = CW / 1000, CH / 1000
    else:
        fx = fy = 1.0
    return [(x * fx, y * fy) for x, y in raw_pts]


def score():
    items = {it['id']: it for it in json.load(open(ROOT / 'items.json'))}
    res = {}
    for f in sorted(OUT.glob('*_run*.jsonl')):
        vendor = f.stem.rsplit('_run', 1)[0]
        for l in open(f):
            r = json.loads(l)
            if r['status'] not in ('success', 'parse_error'): continue
            it = items[r['id']]; pts = decode(vendor, r.get('shapes', []) if r['status'] == 'success' else [], it.get('canvas', [CANVAS, CANVAS]), raw=r.get('raw') if r['status'] == 'parse_error' else None, iid=r['id'], m=it['m'])
            tp, n_p, n_g = M.match(pts, [tuple(p) for p in it['points']], it['m'] / 2)
            res.setdefault((vendor, it['exp'], it['level']), []).append((tp / n_g, tp / max(n_p, 1), 2 * tp / (n_p + n_g)))
    json.dump({f'{k[0]}|{k[1]}|{k[2]}': np.mean(v, axis=0).tolist() + [len(v)] for k, v in res.items()}, open(ROOT / f'scores_{READING}.json', 'w'), indent=1)
    for exp, levels in (('res', RES_SIZES), ('cap', CAP_NS), ('area', [f'{w}x{h}' for w, h in AREA_CANVAS])):
        print(f'\n=== {exp}: recall (F1) by level')
        print('vendor    ' + ''.join(f'{lv:>12}' for lv in levels))
        for v in CONV:
            row = [res.get((v, exp, lv)) for lv in levels]
            if not any(row): continue
            print(f'{v:10s}' + ''.join(f'{np.mean([x[0] for x in r]):6.2f}({np.mean([x[2] for x in r]):.2f})' if r else f'{"":>12}' for r in row))


M.FORMAT_AWARE = True   # new experiments read each model's native point format

if __name__ == '__main__':
    if sys.argv[1] == 'build':
        build()
    elif sys.argv[1] == 'build_extra':
        build_extra()
    elif sys.argv[1] == 'run':
        M.C.load_env('../config.env')
        for v in sys.argv[2].split(','):
            run(v, int(sys.argv[3]) if len(sys.argv) > 3 else 0)
    else:
        score()
