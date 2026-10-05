#!/usr/bin/env python3
"""Direct test of Assumption 1 (R, M, I) and Theorem 1 on synthetic images (prereg claude/assumption_test_prereg_v7.md).
usage: python3 syn_assume.py build | run <vendor> | score"""
import json, math, os, random, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
import mech as M
import syn as SY

M.FORMAT_AWARE = True
ROOT = Path('syn_assume'); IMG = ROOT / 'images'; IMG.mkdir(parents=True, exist_ok=True)
OUT = ROOT / 'results'; OUT.mkdir(exist_ok=True)
SIDE = 2048; REG = 512; MARG = 64; NT = 48
# per vendor: effective whole-image side (no vendor resizing for resolution-scaled), coordinate convention, fixed grid?
VEND = {'gpt56': (2048, 'px', False), 'gpt54': (1600, 'px', False), 'sonnet55': (1932, 'px', False), 'qwen3vl': (1600, 'n1000', False),
        'gem38': (2048, 'n1000', True), 'gemma4': (2048, 'n1000', True),
        'opus55': (1932, 'px', False), 'gpt61': (2048, 'px', False)}   # v7.1 additions
MIN_CANVAS = {'qwen3vl': 512}   # pad small inputs so the vendor does not enlarge them


def build():
    rng = random.Random(7); items = []
    for m in (12, 20, 32):
        for k in range(4):
            ids = rng.sample(SY.DIDS, 4); tgt, dis = ids[0], ids[1:]
            tpls = {d: Image.open(M.BENCH / f'templates/{d}.png').convert('RGB') for d in ids}
            img, pts = SY.render(tpls[tgt], [tpls[d] for d in dis], m, NT, NT, seed=1000 * m + k, W=SIDE, H=SIDE, clutter=40)
            pid = f'p{m}_{k}'; img.save(IMG / f'{pid}.png')
            boxes = []
            tt = SY.scaled(tpls[tgt], m)
            for x, y in pts: boxes.append([x - tt.width / 2, y - tt.height / 2, x + tt.width / 2, y + tt.height / 2])
            items.append({'id': pid, 'm': m, 'target': tgt, 'points': pts, 'boxes': boxes, 'tw': tt.width, 'th': tt.height})
    json.dump(items, open(ROOT / 'items.json', 'w'), indent=1); print(len(items), 'parents')


def calls_for(it):
    """List of call specs (native region, core, scale factor relative to the whole call, source kind)."""
    cs = [{'kind': 'P', 'region': [0, 0, SIDE, SIDE], 'core': [0, 0, SIDE, SIDE], 'mult': 1, 'src': 'native'}]
    for i in range(SIDE // REG):
        for j in range(SIDE // REG):
            reg = [i * REG, j * REG, (i + 1) * REG, (j + 1) * REG]; core = [reg[0] + MARG, reg[1] + MARG, reg[2] - MARG, reg[3] - MARG]
            cs.append({'kind': f'M{i}{j}', 'region': reg, 'core': core, 'mult': 1, 'src': 'native'})
            cs.append({'kind': f'R{i}{j}', 'region': reg, 'core': core, 'mult': 2, 'src': 'native'})
            cs.append({'kind': f'I{i}{j}', 'region': reg, 'core': core, 'mult': 2, 'src': 'garbled'})
    cell = SIDE / 3
    for i in range(3):
        for j in range(3):
            core = [max(0, i * cell - 16), max(0, j * cell - 16), min(SIDE, (i + 1) * cell + 16), min(SIDE, (j + 1) * cell + 16)]
            view = [max(0, core[0] - 48), max(0, core[1] - 48), min(SIDE, core[2] + 48), min(SIDE, core[3] + 48)]
            cs.append({'kind': f'T{i}{j}', 'region': [round(v) for v in view], 'core': core, 'mult': 1, 'src': 'native'})
    return cs


def render_call(vendor, img, c):
    """Return (image to send, (origin_x, origin_y), caller scale, canvas size) so that native = origin + px / scale."""
    L, coord, fixed = VEND[vendor]
    x0, y0, x1, y1 = c['region']; crop = img.crop((x0, y0, x1, y1))
    if c['src'] == 'garbled':
        w, h = crop.size; crop = crop.resize((max(1, w // 4), max(1, h // 4)), Image.LANCZOS).resize((w, h), Image.LANCZOS)
    if fixed:
        # fixed grid: the vendor maps the (padded) canvas to g x g tokens; the canvas side sets the effective scale
        side = SIDE // c['mult']
        can = Image.new('RGB', (side, side), 'white'); can.paste(crop, (0, 0))
        return can, (x0, y0), 1.0, (side, side)
    s = L / SIDE * c['mult']
    w, h = crop.size; sent = crop.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS) if s != 1 else crop
    mc = MIN_CANVAS.get(vendor, 0)
    if mc and (sent.width < mc or sent.height < mc):
        can = Image.new('RGB', (max(mc, sent.width), max(mc, sent.height)), 'white'); can.paste(sent, (0, 0)); sent = can
    return sent, (x0, y0), s, sent.size


def run(vendor, workers=6):
    items = json.load(open(ROOT / 'items.json')); out = OUT / f'{vendor}.jsonl'
    done = {(json.loads(l)['pid'], json.loads(l)['kind']) for l in out.open() if json.loads(l)['status'] in ('success', 'parse_error')} if out.exists() else set()
    L, coord, fixed = VEND[vendor]; lock = threading.Lock()
    tpl = {it['id']: Image.open(M.BENCH / f"templates/{it['target']}.png").convert('RGB') for it in items}
    imgs = {it['id']: Image.open(IMG / f"{it['id']}.png").convert('RGB') for it in items}
    jobs = [(it, c) for it in items for c in calls_for(it) if (it['id'], c['kind']) not in done]
    print(vendor, len(jobs), 'calls to go', flush=True)

    def one(job):
        it, c = job
        sent, origin, s, canvas = render_call(vendor, imgs[it['id']], c)
        t, f, _ = M.COORD_TEXT[coord]
        sysp = SY.SYS.format(coord=t.format(w=canvas[0], h=canvas[1]).replace('tile', 'image'), fmt=f)
        rec = {'pid': it['id'], 'kind': c['kind'], 'vendor': vendor, 'region': c['region'], 'core': c['core'], 'origin': origin, 'scale': s, 'canvas': canvas}
        for a in range(4):
            try:
                raw, u = SY.call(vendor, [tpl[it['id']], sent], sysp, SY.USER); p = M.C.parse(raw)
                sh = p.get('shapes', []) if isinstance(p, dict) else []
                rec.update(status='parse_error' if p.get('parse_error') else 'success', shapes=sh if isinstance(sh, list) else [], usage=u, raw=raw[:4000]); break
            except Exception as e:
                rec.update(status='api_error', error=str(e)[:200]); time.sleep(5 * 2 ** a)
        with lock, out.open('a') as fo: fo.write(json.dumps(rec) + '\n')
        return rec
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, jobs):
            if r['status'] == 'api_error': print(' ', r['pid'], r['kind'], r.get('error', '')[:120], flush=True)


def kept_points(rec):
    L, coord, fixed = VEND[rec['vendor']]
    pts = [p for p in (M.point_of(q) for q in rec.get('shapes', [])) if p is not None] if rec['status'] == 'success' else []
    if not pts and rec['status'] == 'parse_error': pts = M.lenient_points(rec.get('raw'))
    cw, ch = rec['canvas']; out = []
    for x, y in pts:
        if coord == 'n1000': x, y = x / 1000 * cw, y / 1000 * ch
        X, Y = rec['origin'][0] + x / rec['scale'], rec['origin'][1] + y / rec['scale']
        a, b, c, d = rec['core']
        if a <= X < c and b <= Y < d: out.append((X, Y))
    return out


def interior(box, pt, view, core, tau):
    return view[0] <= box[0] and view[1] <= box[1] and box[2] <= view[2] and box[3] <= view[3] and \
        min(pt[0] - core[0], pt[1] - core[1], core[2] - pt[0], core[3] - pt[1]) >= tau


def score():
    items = {it['id']: it for it in json.load(open(ROOT / 'items.json'))}
    rng = np.random.default_rng(3); res = {}
    for f in sorted(OUT.glob('*.jsonl')):
        v = f.stem; recs = {}
        for l in open(f):
            r = json.loads(l)
            if r['status'] in ('success', 'parse_error'): recs[(r['pid'], r['kind'])] = r
        pairs = {'M: crop - whole (same scale)': [], 'R: 2x crop - crop': [], 'I: native - degraded (same scale)': [], 'Prop crop: 2x crop - whole': [], 'Thm1: safe tiles - whole': []}
        cover_ok = True
        for pid, it in items.items():
            tau = it['m'] / 2; hits = {}
            def hit(kind, k):
                if (pid, kind) not in recs: return None
                key = (kind, k)
                if key not in hits:
                    P = np.array(kept_points(recs[(pid, kind)]), float).reshape(-1, 2); x, y = it['points'][k]
                    hits[key] = float(len(P) > 0 and np.min(np.hypot(P[:, 0] - x, P[:, 1] - y)) <= tau)
                return hits[key]
            whole = [0, 0, SIDE, SIDE]
            for k, (pt, box) in enumerate(zip(it['points'], it['boxes'])):
                if not interior(box, pt, whole, whole, tau): continue
                hp = hit('P', k)
                for c in calls_for(it):
                    if c['kind'][0] != 'M' or not interior(box, pt, c['region'], c['core'], tau): continue
                    ij = c['kind'][1:]; hm, hr, hi = hit('M' + ij, k), hit('R' + ij, k), hit('I' + ij, k)
                    if hm is not None and hp is not None: pairs['M: crop - whole (same scale)'].append((pid, hm - hp))
                    if hr is not None and hm is not None: pairs['R: 2x crop - crop'].append((pid, hr - hm))
                    if hr is not None and hi is not None: pairs['I: native - degraded (same scale)'].append((pid, hr - hi))
                    if hr is not None and hp is not None: pairs['Prop crop: 2x crop - whole'].append((pid, hr - hp))
                tiles = [c for c in calls_for(it) if c['kind'][0] == 'T' and interior(box, pt, c['region'], c['core'], tau)]
                if not tiles: cover_ok = False; continue
                ht = [hit(c['kind'], k) for c in calls_for(it) if c['kind'][0] == 'T']
                if hp is not None and all(h is not None for h in ht):
                    # the decomposition detects the target if any tile returns a kept point within tau
                    pairs['Thm1: safe tiles - whole'].append((pid, max(ht) - hp))
        for name, xs in pairs.items():
            if not xs: continue
            g = {}
            for pid, d in xs: g.setdefault(pid, []).append(d)
            keys = list(g); arr = np.array([d for _, d in xs])
            bs = [np.mean([d for kk in (keys[i] for i in rng.integers(0, len(keys), len(keys))) for d in g[kk]]) for _ in range(4000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            bym = {}
            for m in (12, 20, 32):
                sel = [d for pid, d in xs if items[pid]['m'] == m]
                if sel: bym[m] = float(np.mean(sel))
            res[f'{v}|{name}'] = {'diff': float(arr.mean()), 'lo': float(lo), 'hi': float(hi), 'n_targets': len(xs), 'n_parents': len(keys),
                                  'by_m': bym, 'verdict': 'violated' if hi < 0 else ('supported' if lo > 0 else 'consistent')}
            print(f'{v:9s} {name:36s} {arr.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] targets={len(xs):4d} by m {bym} {res[f"{v}|{name}"]["verdict"]}')
        print(v, 'cover check (every interior target of P interior to some tile):', cover_ok)
    json.dump(res, open(ROOT / 'assume_scores.json', 'w'), indent=1)


if __name__ == '__main__':
    if sys.argv[1] == 'build': build()
    elif sys.argv[1] == 'run':
        M.C.load_env('../config.env')
        import gen_common as G; G.REASONING['gpt-6.1-sol'] = 'low'   # lowest effort the API accepts (v7.1)
        run(sys.argv[2])
    else: score()
