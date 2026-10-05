#!/usr/bin/env python3
"""Prereg v8: the safe decomposition rule of Cor. cor:safe (full-core views of side e/sigma) on REDP-X40 test drawings
and FPC-sheets k3.  usage: python3 cs_rule.py <redp|fpc> <vendor> [plan]"""
import json, math, sys, glob
from pathlib import Path
from PIL import Image, ImageDraw
import mech as M
import ceilings as K

BENCH_ARG = sys.argv[1] if len(sys.argv) > 1 else 'redp'
if BENCH_ARG == 'fpc':
    M.BENCH = Path('../data/fpc/sheets')
import gen_test  # noqa: E402  dispatch for second-round vendors, TILE_IFACE
import main_test as T  # noqa: E402

S_STAR = 1.5
GEO = {'gpt61': ('res', 32, 8192, None), 'opus55': ('res', 28, 4784, 2576), 'qwen3vl': ('res', 32, 2500, None), 'gem38': ('grid', None, 1090, None)}
OUT = Path(f'results_cs/{BENCH_ARG}'); OUT.mkdir(parents=True, exist_ok=True)


def items():
    import os
    only = os.environ.get('CS_ONLY')
    if only: return only.split(',')
    if BENCH_ARG == 'fpc':
        return sorted(Path(p).stem for p in glob.glob('../data/fpc/sheets/annotations/*_k3.json'))
    return list(K.TEST)


def s_whole(v, W, H, m):
    """Tokens per symbol side of the A1 whole-image call (long side <= 1600), from the measured interfaces."""
    s = min(1.0, 1600 / max(W, H)); w, h = round(W * s), round(H * s)
    if v == 'gem38': return m * s / (max(w, h) / math.sqrt(1090))
    if v == 'gpt61': return m * s * K.gpt_fit(w, h, 8192) / 32
    if v == 'qwen3vl': return m * s * min(1.0, math.sqrt(2500 * 1024 / (w * h))) / 32
    if v == 'opus55': rw, rh = M.claude_resized(w, h, 2576, 4784); return m * s * rw / w / 28


def plan(v, ann):
    w, h = ann['template_size']; m = math.sqrt(w * h); tau = M.taus(ann)[0]
    ex, ey = max(w, 2 * tau), max(h, 2 * tau); W, H = ann['width'], ann['height']
    Sw = s_whole(v, W, H, m); S = max(S_STAR, Sw)
    kind, t, N, cap = GEO[v]
    lam = (S / m) ** 2; sigma = math.sqrt(lam * ex * ey / N)
    if kind == 'grid':
        A = math.floor(m * math.sqrt(N) / S); ax = ay = A; s = 1.0; jx = jy = None; T_tok = N; canvas = (A, A)
    else:
        jx, jy = math.floor(ex * S / (m * sigma)), math.floor(ey * S / (m * sigma))
        if cap: jx, jy = min(jx, cap // t), min(jy, cap // t)
        while jx * jy > N: jx, jy = (jx - 1, jy) if jx >= jy else (jx, jy - 1)
        s = S * t / m; ax, ay = jx * m / S, jy * m / S; T_tok = jx * jy; canvas = (jx * t, jy * t)
    px, py = ax - ex, ay - ey
    nx, ny = math.ceil(max(W - ex, 1e-9) / px), math.ceil(max(H - ey, 1e-9) / py)
    chi = T_tok * (1 - sigma) ** 2 / (lam * px * py)
    cmin = max(W - ex, 0) * max(H - ey, 0) * lam / (1 - sigma) ** 2
    return dict(m=m, tau=tau, ex=ex, ey=ey, Sw=Sw, S=S, sigma=sigma, s=s, ax=ax, ay=ay, px=px, py=py, nx=nx, ny=ny, jx=jx, jy=jy,
                T=T_tok, chi=chi, cmin=cmin, canvas=canvas, n_views=nx * ny, tokens=nx * ny * T_tok)


def views(ann, img, p):
    W, H = img.size; out = []
    for j in range(p['ny']):
        for i in range(p['nx']):
            x0, y0 = i * p['px'], j * p['py']
            x1, y1 = min(W, x0 + p['ax']), min(H, y0 + p['ay'])
            box = (round(x0), round(y0), round(x1), round(y1))
            crop = img.crop(box)
            cw, ch = max(1, round((box[2] - box[0]) * p['s'])), max(1, round((box[3] - box[1]) * p['s']))
            if p['s'] != 1.0: crop = crop.resize((cw, ch), Image.LANCZOS)
            can = Image.new('RGB', p['canvas'], 'white'); can.paste(crop, (0, 0))
            ImageDraw.Draw(can).rectangle([0, 0, cw - 1, ch - 1], outline=(0, 90, 255), width=3)
            out.append({'i': i, 'j': j, 'img': can, 'origin_z0': (box[0] * p['s'], box[1] * p['s']), 'core_px': (0, 0, cw, ch),
                        'z0': p['s'], 'up': 1})
    return out


def run(v, run_idx=0):
    coord, decode = T.TILE_IFACE[v]
    out = OUT / f'CS_{v}_run{run_idx}.jsonl'
    key = lambda r: (r['did'], r['i'], r['j'])
    for did in items():
        ann, img, tpl = M.load(did); p = plan(v, ann)
        tp = tpl if GEO[v][0] == 'grid' else M.scaled_template(tpl, p['s'], 1)
        jobs = [{'model': v, 'coord': coord, 'tpl': tp, 'tile': t,
                 'meta': {'did': did, 'model': v, 'cell': 'CS', 'cond': 'CS', 'run': run_idx, 'i': t['i'], 'j': t['j'], 'decode': decode,
                          'S': p['S'], 'Sw': p['Sw'], 'sigma': p['sigma'], 'chi': p['chi']}} for t in views(ann, img, p)]
        del img
        M.run_jobs(jobs, out, key=key, workers=T.WORKERS.get(v, 6))


if __name__ == '__main__':
    v = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3] == 'plan':
        tot = 0; nv = 0
        for did in items():
            ann = json.loads((M.BENCH / 'annotations' / f'{did}.json').read_text()); p = plan(v, ann); tot += p['tokens']; nv += p['n_views']
            print(did, f"m={p['m']:.0f} Sw={p['Sw']:.2f} S={p['S']:.2f} views={p['n_views']} canvas={p['canvas']} chi={p['chi']:.3f} tokens/cmin={p['tokens']/max(p['cmin'],1):.2f}")
        print(v, BENCH_ARG, 'views', nv, 'tokens', tot)
    else:
        M.C.load_env('../config.env')
        run(v)
