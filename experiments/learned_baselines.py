#!/usr/bin/env python3
"""
X5: learned, training-free exemplar-based detectors as classical-vision references (CPU).
  owlv2 : OWLv2 image-guided one-shot detection (google/owlv2-base-patch16-ensemble), query = legend template.
Input: the same 1024-px tiles (symbol 44 px) as the VLM tile condition, with the same core-only rule.
All detections with score >= 0.02 are stored; the score threshold is chosen on the six tuning drawings only.
usage: python3 learned_baselines.py detect owlv2 [dids] | score owlv2
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import mech as M, ceilings as K
from x_small import x2_tiles

OUT = Path('results_x/learned'); OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)


def owlv2_model():
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    name = 'google/owlv2-base-patch16-ensemble'
    return Owlv2Processor.from_pretrained(name), Owlv2ForObjectDetection.from_pretrained(name).eval()


def detect(method, dids):
    proc, model = owlv2_model()
    out = OUT / f'{method}_dets.jsonl'
    done = {json.loads(l)['did'] for l in out.open()} if out.exists() else set()
    for did in dids:
        if did in done: continue
        t0 = time.time()
        ann, img, tpl = M.load(did)
        ts, z = x2_tiles(ann, img, 1024); q = M.scaled_template(tpl, z, 1)
        pts = []
        for t in ts:
            im = t['img']
            inputs = proc(images=im, query_images=q, return_tensors='pt')
            with torch.no_grad():
                o = model.image_guided_detection(**inputs)
            # raw per-box scores (the library post-processing rescales scores per image, which breaks a global threshold)
            L = max(im.size)
            sc = o.logits[0, :, 0]; bx = o.target_pred_boxes[0]   # raw logits (sigmoid saturates for image-guided queries)
            top = torch.topk(sc, k=min(300, sc.numel()))
            res = {'scores': top.values, 'boxes': [[(b[0] - b[2] / 2) * L, (b[1] - b[3] / 2) * L, (b[0] + b[2] / 2) * L, (b[1] + b[3] / 2) * L] for b in bx[top.indices].tolist()]}
            a, b, c, e = t['core_px']; x0, y0 = t['origin_z0']
            for box, sc in zip(res['boxes'], res['scores'].tolist()):
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if a <= cx < c and b <= cy < e:
                    pts.append([(x0 + cx) / z, (y0 + cy) / z, float(sc)])
        with out.open('a') as f:
            f.write(json.dumps({'did': did, 'points': pts, 'tiles': len(ts), 'sec': time.time() - t0}) + '\n')
        print(method, did, len(ts), 'tiles', len(pts), 'dets', f'{time.time() - t0:.0f}s', flush=True)


def nms(pts, r):
    pts = sorted(pts, key=lambda p: -p[2]); keep = []
    for p in pts:
        if all((p[0] - k[0]) ** 2 + (p[1] - k[1]) ** 2 > r * r for k in keep):
            keep.append(p)
    return keep


def score(method):
    import analyze_test as A
    dets = {}
    for l in open(OUT / f'{method}_dets.jsonl'):
        r = json.loads(l); dets[r['did']] = r['points']
    grid = np.round(np.arange(0.02, 0.9, 0.01), 2)
    def f1_at(d, th):
        a = A.ANN[d] if d in A.ANN else json.loads((M.BENCH / 'annotations' / f'{d}.json').read_text())
        tau = M.taus(a)[0]
        p = nms([x for x in dets[d] if x[2] >= th], tau)
        return A.metrics(d, [(x[0], x[1]) for x in p])
    dev = [d for d in K.DEV if d in dets]
    th = float(grid[int(np.argmax([np.mean([f1_at(d, t)[0] for d in dev]) for t in grid]))])
    per = {d: [f1_at(d, th)] for d in K.TEST if d in dets}
    json.dump({'threshold': th, 'per': {d: v[0].tolist() for d, v in per.items()}}, open(OUT / f'{method}_scores.json', 'w'), indent=1)
    m = np.mean([v[0] for v in per.values()], axis=0)
    print(f'{method}: threshold {th} (tuning set), test F1 {m[0]:.3f} recall {m[1]:.3f} precision {m[2]:.3f} (n={len(per)})')
    return per


if __name__ == '__main__':
    if sys.argv[1] == 'detect':
        dids = sys.argv[3].split(',') if len(sys.argv) > 3 else K.DEV + K.TEST
        detect(sys.argv[2], dids)
    else:
        score(sys.argv[2])
