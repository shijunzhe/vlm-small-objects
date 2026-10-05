#!/usr/bin/env python3
"""FPC-300: a public second domain from FloorPlanCAD (test split, Voxel51/FloorPlanCAD on Hugging Face, 1000x1000 rasters).
Item = (image, thing class) with 3 to 80 instances; one item per image; at most 20 items per class; seed 2026.
Reference = crop of the instance with the median box area in that image and class (as a legend crop would be);
ground truth = box centres of all instances of the class. Same file format as REDP-X40."""
import json, random, collections, subprocess, os
from pathlib import Path
import numpy as np
from PIL import Image
ROOT = Path('../data/fpc'); B = ROOT / 'bench'
KEEP = ['double_door', 'sliding_door', 'single_door', 'window', 'chair', 'bath', 'table', 'half_height_cabinet', 'wardrobe', 'toilet',
        'bed', 'sink', 'bath_tub', 'tv_cabinet', 'urinal', 'high_cabinet', 'sofa', 'squat_toilet', 'bedside_cupboard', 'refrigerator',
        'washing_machine', 'gas_stove']
if __name__ == '__main__':
    smp = json.load(open(ROOT / 'samples.json'))['samples']
    cands = collections.defaultdict(list)
    for s in smp:
        W, H = s['metadata']['width'], s['metadata']['height']
        by = collections.defaultdict(list)
        for d in s['ground_truth']['detections']:
            if d['label'] in KEEP: by[d['label']].append(d['bounding_box'])
        for c, bx in by.items():
            if 3 <= len(bx) <= 80: cands[c].append((s['filepath'], W, H, bx))
    rng = random.Random(2026); used = set(); items = []
    order = sorted(cands, key=lambda c: len(cands[c]))   # rare classes first so they get their images
    for c in order:
        L = sorted(cands[c], key=lambda x: x[0]); rng.shuffle(L); k = 0
        for fp, W, H, bx in L:
            if fp in used or k >= 20: continue
            used.add(fp); k += 1; items.append((c, fp, W, H, bx))
    print(len(items), 'items;', collections.Counter(i[0] for i in items))
    man = []
    for c, fp, W, H, bx in items:
        stem = fp.split('/')[-1][:-4]; iid = f'f{stem}_{c}'
        ip = B / 'images' / f'{stem}.png'
        if not ip.exists():
            subprocess.run(['curl', '-sL', '-o', str(ip) + '.tmp', f'https://huggingface.co/datasets/Voxel51/FloorPlanCAD/resolve/main/{fp}'], check=True)
            im = Image.open(str(ip) + '.tmp').convert('RGBA'); bg = Image.new('RGBA', im.size, (255, 255, 255, 255)); bg.alpha_composite(im)
            bg.convert('RGB').save(ip); os.remove(str(ip) + '.tmp')
        img = Image.open(ip)
        boxes = [[b[0] * W, b[1] * H, b[2] * W, b[3] * H] for b in bx]   # x, y, w, h in pixels
        areas = [b[2] * b[3] for b in boxes]; ref = boxes[int(np.argsort(areas)[len(areas) // 2])]
        pad = 0.1 * max(ref[2], ref[3])
        tpl = img.crop((max(0, round(ref[0] - pad)), max(0, round(ref[1] - pad)), min(W, round(ref[0] + ref[2] + pad)), min(H, round(ref[1] + ref[3] + pad))))
        tpl.save(B / 'templates' / f'{iid}.png')
        ann = {'id': iid, 'image': f'images/{stem}.png', 'template': f'templates/{iid}.png', 'width': W, 'height': H,
               'template_size': [max(1, round(ref[2])), max(1, round(ref[3]))], 'count': len(boxes),
               'points': [[b[0] + b[2] / 2, b[1] + b[3] / 2] for b in boxes], 'boxes': boxes, 'subset': 'fpc', 'class': c, 'source': fp}
        json.dump(ann, open(B / 'annotations' / f'{iid}.json', 'w')); man.append(iid)
    rng2 = random.Random(7); dev = sorted(rng2.sample(man, 30))
    json.dump({'name': 'FPC-300', 'items': sorted(man), 'dev': dev, 'test': sorted(set(man) - set(dev)), 'license': 'FloorPlanCAD, CC BY-NC 4.0'}, open(B / 'manifest.json', 'w'), indent=1)
    print('done', len(man))

