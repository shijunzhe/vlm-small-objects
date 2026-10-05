#!/usr/bin/env python3
"""FPC-sheets: real FloorPlanCAD symbols at increasing sheet area. For source drawings with >= 9 test blocks, the first
9 blocks (sorted by name) are tiled 3x3 (3000^2 px); the top-left 2x2 (2000^2) and the first block (1000^2) are nested
subsets, so k = 1, 2, 3 share symbols, style, and reference. One class per sheet (the KEEP class with the most instances
over the 9 blocks, at least 6, and at least 1 in block 0). Reference = median-area instance in block 0.
Each annotation lists block rectangles; scoring ignores predictions and labels within m/2 of any block edge, where
FloorPlanCAD's cropping cuts instances. Seed 2026; 60 sheets."""
import json, random, collections, subprocess, os
from pathlib import Path
import numpy as np
from PIL import Image
from build_fpc import KEEP, ROOT
B = ROOT / 'sheets'; CACHE = ROOT / 'blocks'; CACHE.mkdir(exist_ok=True)
smp = json.load(open(ROOT / 'samples.json'))['samples']
by_draw = collections.defaultdict(list)
for s in smp:
    by_draw[s['filepath'].split('/')[-1].split('-')[0]].append(s)
cand = []
for dw, L in sorted(by_draw.items()):
    if len(L) < 9: continue
    L = sorted(L, key=lambda s: s['filepath'])[:9]
    if any((s['metadata']['width'], s['metadata']['height']) != (1000, 1000) for s in L): continue
    cnt = collections.Counter(); first = collections.Counter()
    for bi, s in enumerate(L):
        for d in s['ground_truth']['detections']:
            if d['label'] in KEEP:
                cnt[d['label']] += 1
                if bi == 0: first[d['label']] += 1
    ok = [c for c in cnt if cnt[c] >= 6 and first[c] >= 1]
    if ok:
        c = max(ok, key=lambda c: (cnt[c], c)); cand.append((dw, L, c))
rng = random.Random(2026); rng.shuffle(cand); cand = sorted(cand[:60], key=lambda x: x[0])
print(len(cand), 'sheets', collections.Counter(c for _, _, c in cand))


def block(s):
    p = CACHE / s['filepath'].split('/')[-1]
    if not p.exists():
        subprocess.run(['curl', '-sL', '-o', str(p) + '.tmp', f"https://huggingface.co/datasets/Voxel51/FloorPlanCAD/resolve/main/{s['filepath']}"], check=True)
        im = Image.open(str(p) + '.tmp').convert('RGBA'); bg = Image.new('RGBA', im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
        bg.convert('RGB').save(p); os.remove(str(p) + '.tmp')
    return Image.open(p).convert('RGB')


items = []
for dw, L, c in cand:
    boxes_all = []
    for bi, s in enumerate(L):
        ox, oy = (bi % 3) * 1000, (bi // 3) * 1000
        for d in s['ground_truth']['detections']:
            if d['label'] == c:
                b = d['bounding_box']; boxes_all.append((bi, [ox + b[0] * 1000, oy + b[1] * 1000, b[2] * 1000, b[3] * 1000]))
    b0 = [b for bi, b in boxes_all if bi == 0]; ref = b0[int(np.argsort([b[2] * b[3] for b in b0])[len(b0) // 2])]
    for k, blocks in ((1, [0]), (2, [0, 1, 3, 4]), (3, list(range(9)))):
        W = H = 1000 * k; can = Image.new('RGB', (W, H), 'white'); rects = []
        for bi in blocks:
            r, col = divmod(bi, 3); x, y = col * 1000, r * 1000
            if x >= W or y >= H: continue
            can.paste(block(L[bi]), (x, y)); rects.append([x, y, x + 1000, y + 1000])
        iid = f's{dw}_{c}_k{k}'
        can.save(B / 'images' / f'{iid}.png')
        if k == 1:
            pad = 0.1 * max(ref[2], ref[3])
            can.crop((max(0, round(ref[0] - pad)), max(0, round(ref[1] - pad)), round(ref[0] + ref[2] + pad), round(ref[1] + ref[3] + pad))).save(B / 'templates' / f's{dw}_{c}.png')
        bx = [b for bi, b in boxes_all if bi in blocks]
        ann = {'id': iid, 'image': f'images/{iid}.png', 'template': f'templates/s{dw}_{c}.png', 'width': W, 'height': H,
               'template_size': [max(1, round(ref[2])), max(1, round(ref[3]))], 'count': len(bx),
               'points': [[b[0] + b[2] / 2, b[1] + b[3] / 2] for b in bx], 'boxes': bx, 'blocks': rects, 'k': k,
               'subset': f'fpc_k{k}', 'class': c, 'source': [s['filepath'] for s in L]}
        json.dump(ann, open(B / 'annotations' / f'{iid}.json', 'w')); items.append(iid)
json.dump({'name': 'FPC-sheets', 'items': items, 'license': 'FloorPlanCAD, CC BY-NC 4.0'}, open(B / 'manifest.json', 'w'), indent=1)
print(len(items), 'items')
