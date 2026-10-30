#!/usr/bin/env python3
"""Phase 0: service-path checks, dry run, and freezing of phase-1 layouts/images.
usage: phase0.py calib | freeze_p1 | dry | pad | report"""
import json, sys, time, math
from pathlib import Path
from PIL import Image, ImageDraw
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import vspec as V, vcall as VC, vrun as R

ROOT = V.ROOT; OUT = ROOT / 'results'; OUT.mkdir(exist_ok=True)
S_LEVELS = [0.2, 0.3, 0.45, 0.65, 0.9, 1.3, 1.9, 2.8]
N0 = 8

CALIB = {'gpt61': [(1536, 1536), (2048, 2048), (2896, 2896)], 'gpt56': [(1536, 1536), (2048, 2048), (2896, 2896)],
         'opus55': [(448, 448), (1932, 1932)], 'gem38': [(1024, 1024)],
         'qwen3vl': [(128, 128), (256, 256), (512, 512), (1536, 1536), (1600, 1600), (2048, 2048)]}


def calib():
    out = OUT / 'phase0_calib.jsonl'
    have = {(json.loads(l)['model'], tuple(json.loads(l)['size'] or [0, 0])) for l in out.open()} if out.exists() else set()
    sysp = 'You are a helpful assistant.'; user = 'Reply with the single word OK.'
    for model, sizes in CALIB.items():
        for sz in [None] + sizes:
            key = (model, tuple(sz or [0, 0]))
            if key in have:
                continue
            imgs = []
            if sz:
                im = Image.new('RGB', sz, 'white'); d = ImageDraw.Draw(im)
                for k in range(0, max(sz), 64):
                    d.line([k, 0, k, sz[1]], fill=(150, 150, 150)); d.line([0, k, sz[0], k], fill=(150, 150, 150))
                imgs = [im]
            rec = dict(model=model, size=list(sz) if sz else None)
            try:
                if model == 'gem38' or imgs:
                    raw, u = VC.call(model, imgs, sysp, user)
                else:
                    raw, u = VC.call(model, [], sysp, user)
                rec.update(status='success', usage=u, raw=raw[:200], cost=VC.cost(model, u))
            except Exception as e:
                rec.update(status='api_error', error=f'{type(e).__name__}: {e}'[:400])
            with out.open('a') as f:
                f.write(json.dumps(rec) + '\n')
            print(model, sz, rec['status'], rec.get('usage', {}).get('input_tokens'), rec.get('usage', {}).get('provider'), rec.get('error', '')[:200], flush=True)


def p1_renders():
    rr = [(m, S) for m in V.ORDER for S in S_LEVELS]
    return rr + [('opus55', 0.3), ('opus55', 0.65)]


def freeze_p1():
    log = []; layouts = {}
    for i in range(12):
        lay, k = V.frozen_layout('P1', i, N0, p1_renders(), log)
        assert lay is not None, i
        layouts[i] = lay
    items, unreach = [], []
    for i, lay in layouts.items():
        for m in V.ORDER:
            for S in S_LEVELS:
                it, r = R.make_item('p1', lay, m, S, 'S')
                (items.append(it) if it else unreach.append(dict(i=i, model=m, S=S, why=r)))
    json.dump(items, open(ROOT / 'items_p1.json', 'w'), indent=1)
    json.dump(dict(layout_log=log, unreachable=unreach), open(ROOT / 'freeze_p1_log.json', 'w'), indent=1)
    print(len(items), 'phase-1 items;', len(unreach), 'unreachable;', 'k used:', [l['k'] for l in log])


def dry_items():
    spec = [(8, 0.45), (8, 0.9), (8, 1.9), (32, 1.0), (128, 0.8)]
    items = []
    for j, (n, S) in enumerate(spec):
        renders = [(m, S) for m in V.ORDER]
        lay, k = V.frozen_layout('P0', j, n, renders)
        for m in V.ORDER:
            it, r = R.make_item('p0', lay, m, S, 'dry')
            if it: items.append(it)
            else: print('dry unreachable', m, n, S, r)
    return items


def pad_items():
    items = []
    p1 = json.load(open(ROOT / 'items_p1.json'))
    lays = {}
    for i in range(12):
        lay, k = V.frozen_layout('P1', i, N0, p1_renders())
        for S in (0.3, 0.65):
            it, r = R.make_item('p0', lay, 'opus55', S, 'nopad', pad=False)
            items.append(it)
    return items


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'calib':
        calib()
    elif cmd == 'freeze_p1':
        freeze_p1()
    elif cmd == 'dry':
        R.run_items(dry_items(), OUT / 'phase0_dry.jsonl', workers=5)
    elif cmd == 'pad':
        R.run_items(pad_items(), OUT / 'phase0_pad.jsonl', workers=6)
