#!/usr/bin/env python3
"""
Hybrid pilot (dev set): template matching proposes, the VLM verifies.

Candidates: multi-scale/rotation NCC candidates (results/tm_candidates) with score >= 0.5, at most the
top 300 per drawing (budget cap; fixed before any VLM call).
Verification: one candidate per call. Crop centered on the candidate, side = 3 x the template's long
side (native px), resized to 768x768; a thin red square (1.3 x template long side) marks the candidate.
The template is shown at the same scale as the crop. The VLM answers match true/false (+ confidence).
Pre-stated decision rule: accept iff match == true. Primary metric: strict F1 of the accepted centers.

usage: python3 hybrid_verify.py run <model>[,<model>...] [dids]
       python3 hybrid_verify.py score
"""
import json, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import mech as M

DEV = ['d001', 'd004', 'd008', 'd010', 'd017', 'd019']
FLOOR, CAP, CROP = 0.5, 300, 768
SYSTEM = """You are a precise engineering drawing analysis assistant.

You will see two images. Image 1 is a reference crop of ONE target symbol taken from the drawing's
legend. Image 2 is a small crop of the drawing, shown at the same scale as the reference, with a thin
red square marking ONE candidate location.

Decide whether the red square contains an instance of the reference symbol. Instances may be rotated or
slightly rescaled, but must have the reference's distinctive shape. Similar-looking symbols, parts of
other symbols, text, and plain linework are NOT matches.

Respond with ONLY a JSON object (no markdown, no commentary):
{"match": true or false, "confidence": <integer 0-100>}"""
USER = "Does the red square in image 2 contain an instance of the reference symbol in image 1? JSON only."


def candidates(did):
    c = json.load(open(f'../data/redp40/results/tm_candidates/{did}.json'))['candidates']
    c = sorted([x for x in c if x[2] >= FLOOR], key=lambda x: -x[2])[:CAP]
    return c


def crop_for(img, tpl, ann, x, y):
    tw, th = ann['template_size']; L = max(tw, th)
    side = 3 * L; f = CROP / side
    box = (round(x - side / 2), round(y - side / 2), round(x + side / 2), round(y + side / 2))
    cr = Image.new('RGB', (side, side), 'white')
    W, H = img.size
    sub = img.crop((max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3])))
    cr.paste(sub, (max(0, box[0]) - box[0], max(0, box[1]) - box[1]))
    cr = cr.resize((CROP, CROP), Image.LANCZOS)
    d = ImageDraw.Draw(cr); h = 0.65 * L * f; c = CROP / 2
    d.rectangle([c - h, c - h, c + h, c + h], outline=(230, 0, 0), width=2)
    t = tpl.resize((max(1, round(tpl.width * f)), max(1, round(tpl.height * f))), Image.LANCZOS)
    return t, cr


def run(models, dids):
    M.C.load_env('../config.env')
    for model in models:
        out = M.OUT / f'hyb_{model}.jsonl'
        done = set()
        if out.exists():
            for l in out.open():
                r = json.loads(l)
                if r['status'] in ('success', 'parse_error'):
                    done.add((r['did'], r['k']))
        lock = threading.Lock()
        for did in dids:
            ann, img, tpl = M.load(did)
            cand = candidates(did)
            todo = [(k, x) for k, x in enumerate(cand) if (did, k) not in done]
            print(f'{model} {did}: {len(cand)} candidates, {len(todo)} to go', flush=True)

            def one(job):
                k, (x, y, sc, v) = job
                t, cr = crop_for(img, tpl, ann, x, y)
                rec = {'model': model, 'did': did, 'k': k, 'x': x, 'y': y, 'score': sc}
                for attempt in range(3):
                    try:
                        raw, usage = M.C.call(model, [t, cr], SYSTEM, USER)
                        p = M.C.parse(raw)
                        mt = p.get('match') if isinstance(p, dict) else None
                        if isinstance(mt, str):
                            mt = mt.strip().lower() == 'true'
                        rec.update(status='success' if isinstance(mt, bool) else 'parse_error', match=mt,
                                   conf=p.get('confidence') if isinstance(p, dict) else None, usage=usage, raw=raw[:300])
                        break
                    except Exception as e:
                        rec.update(status='api_error', error=f'{type(e).__name__}: {e}'[:200]); time.sleep(5 * 2 ** attempt)
                with lock, out.open('a') as f:
                    f.write(json.dumps(rec) + '\n')
                return rec

            with ThreadPoolExecutor(8) as ex:
                for r in ex.map(one, todo):
                    if r['status'] != 'success':
                        print('  ', did, r['k'], r['status'], r.get('error', r.get('raw', ''))[:120], flush=True)


def score():
    ev = json.load(open('../data/redp40/results/tm_eval.json'))
    print(f'{"drawing":8s}' + ''.join(f'{x:>10s}' for x in ['TM LOO', 'cand bound', 'cand raw']))
    base = {}
    for did in DEV:
        ann = json.loads((M.BENCH / 'annotations' / f'{did}.json').read_text()); tau, _ = M.taus(ann)
        cand = candidates(did)
        tp, n, g = M.match([(x, y) for x, y, s, v in cand], ann['points'], tau)
        R = tp / g
        base[did] = (ev[did]['loo']['f1_sym'], 2 * R / (1 + R), 2 * tp / (n + g))
        print(f'{did:8s}' + ''.join(f'{v:10.2f}' for v in base[did]))
    B = np.array(list(base.values())).mean(0)
    print(f'{"mean":8s}' + ''.join(f'{v:10.2f}' for v in B))
    for f in sorted(M.OUT.glob('hyb_*.jsonl')):
        model = f.stem[4:]
        recs = {}
        for l in f.open():
            r = json.loads(l); recs[(r['did'], r['k'])] = r
        per, acc_rate, fails, usage = {}, [], 0, [0, 0]
        for did in DEV:
            ann = json.loads((M.BENCH / 'annotations' / f'{did}.json').read_text()); tau, _ = M.taus(ann)
            rs = [r for (d, k), r in recs.items() if d == did]
            if len(rs) < len(candidates(did)):
                continue
            fails += sum(r['status'] != 'success' for r in rs)
            acc = [(r['x'], r['y']) for r in rs if r.get('match') is True]
            acc_rate += [r.get('match') is True for r in rs]
            for r in rs:
                u = r.get('usage') or {}
                usage[0] += u.get('input_tokens', 0) or 0; usage[1] += u.get('output_tokens', 0) or 0
            tp, n, g = M.match(acc, ann['points'], tau)
            per[did] = (2 * tp / (n + g), tp / g, tp / max(n, 1), abs(n - g) / g)
        if not per:
            continue
        V = np.array(list(per.values())).mean(0)
        print(f'\n{model}: hybrid strict F1 {V[0]:.2f}  recall {V[1]:.2f}  precision {V[2]:.2f}  |relerr| {V[3]:.2f}  '
              f'(drawings {len(per)}, accept rate {np.mean(acc_rate):.2f}, failed calls {fails}, tokens in/out {usage[0]}/{usage[1]})')
        print('   per drawing F1: ' + ', '.join(f'{d} {v[0]:.2f}' for d, v in per.items()))


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run(sys.argv[2].split(','), sys.argv[3].split(',') if len(sys.argv) > 3 else DEV)
    else:
        score()
