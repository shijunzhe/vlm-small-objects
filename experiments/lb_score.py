#!/usr/bin/env python3
"""X5 scoring: learned exemplar detectors (CountGD, GeCo2, OWLv2) run on Modal GPUs (modal_lb/lb_modal.py) on the same
1024-px tiles as the VLM tile condition. Detections are NMS-merged within tau; the score threshold is chosen on the six
REDP-X40 tuning drawings only (GeCo2 scores are relative to the tile maximum, as in its reference demo) and applied
unchanged to the 34 test drawings and to REDP-10. A per-drawing oracle threshold is reported as an upper bound."""
import json
from pathlib import Path
import numpy as np
import mech as M, ceilings as K
import analyze_test as A

R = Path('results_x/learned')
B10 = Path('../data/redp10/redp10')
I10 = sorted(p.stem for p in (B10 / 'annotations').glob('r*.json'))
for d in I10:
    A.ANN[d] = json.loads((B10 / 'annotations' / f'{d}.json').read_text()); A.SUB[d] = 'redp10'


def nms(pts, r):
    pts = sorted(pts, key=lambda p: -p[2]); keep = []
    for p in pts:
        if all((p[0] - k[0]) ** 2 + (p[1] - k[1]) ** 2 > r * r for k in keep): keep.append(p)
    return keep


def score_of(method, p):
    return p[2] / max(p[5], 1e-9) if method == 'geco2' else p[2]


def load(method, bench):
    out = {}
    f = R / f'{method}_{bench}.jsonl'
    if f.exists():
        for l in open(f):
            r = json.loads(l)
            pts = sorted(([p[0], p[1], score_of(method, p)] for p in r['points']), key=lambda p: -p[2])
            out[r['did']] = pts[:2000]   # label-free cap (top 2000 by score), as in lb_score2
    return out


def f1_at(dets, d, th):
    tau = M.taus(A.ANN[d])[0]
    return A.metrics(d, [(x[0], x[1]) for x in nms([p for p in dets[d] if p[2] >= th], tau)])


def main():
    out = {}
    for m in ('countgd', 'geco2', 'owlv2'):
        dx = load(m, 'redp_x40'); d10 = load(m, 'redp10')
        if not all(d in dx for d in K.DEV): continue
        sc = [p[2] for d in K.DEV for p in dx[d]]
        grid = np.unique(np.quantile(sc, np.linspace(0.0, 0.99, 50)))
        th = float(max(grid, key=lambda t: np.mean([f1_at(dx, d, t)[0] for d in K.DEV])))
        per = {d: [f1_at(dx, d, th)] for d in K.TEST if d in dx}
        per10 = {d: [f1_at(d10, d, th)] for d in I10 if d in d10}
        def orac(dets, d):
            s = [p[2] for p in dets[d]]
            return max(f1_at(dets, d, t)[0] for t in np.unique(np.quantile(s, np.linspace(0, 0.99, 25)))) if s else 0.0
        oracle = float(np.mean([orac(dx, d) for d in per])); oracle10 = float(np.mean([orac(d10, d) for d in per10])) if per10 else None
        s = A.summary(per, K.TEST)[0]; sh = A.summary(per, [d for d in K.TEST if A.SUB[d] == 'sheet'])[0]; vw = A.summary(per, [d for d in K.TEST if A.SUB[d] == 'view'])[0]
        s10 = A.summary(per10, I10)[0] if per10 else None
        print(f'{m:8s} thr {th:.3f} | REDP-X40 test F1 {s[0]:.3f} (sheets {sh[0]:.2f}, views {vw[0]:.2f}) R {s[1]:.2f} P {s[2]:.2f} oracle {oracle:.2f} | REDP-10 F1 {s10[0] if s10 is not None else float("nan"):.2f} oracle {oracle10 if oracle10 is not None else float("nan"):.2f}', flush=True)
        out[m] = {'threshold': th, 'f1': [s[0], sh[0], vw[0]], 'recall': s[1], 'precision': s[2], 'oracle': oracle,
                  'redp10_f1': s10[0] if s10 is not None else None, 'redp10_oracle': oracle10,
                  'per': {d: v[0].tolist() for d, v in per.items()}, 'per10': {d: v[0].tolist() for d, v in per10.items()}}
    if not I10 and (R / 'lb_scores.json').exists():   # REDP-10 annotations are not released: keep the shipped REDP-10 scores
        old = json.load(open(R / 'lb_scores.json'))
        for m in out:
            if m in old:
                for k in ('redp10_f1', 'redp10_oracle', 'per10'): out[m][k] = old[m][k]
    json.dump(out, open(R / 'lb_scores.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
