#!/usr/bin/env python3
"""
Classical baseline for REDP-X40: multi-scale, multi-rotation normalized
cross-correlation template matching (OpenCV TM_CCOEFF_NORMED) with the legend
crop as the template, followed by non-maximum suppression.

Stage 1 (this script, --detect): for every drawing, compute all candidate
detections (center, score, variant) above a low floor score. Stored in
results/tm_candidates/dNNN.json.
Stage 2 (--evaluate): choose the score threshold
  * leave-one-drawing-out (LOO): threshold that maximizes mean F1 on the other
    39 drawings (the honest, deployable setting), and
  * oracle: best threshold per drawing (an upper bound, not deployable),
then report count error and F1 under both tolerances used for REDP-X40:
  tau_sym  = 0.5 * sqrt(template_w * template_h)   (native pixels)
  tau_1600 = 30 px in the 1600-px evaluation space used for the VLMs
             (= 30 / min(1, 1600 / long_side) native pixels)

Working resolution: drawings are processed at scale s = min(1, 4096/long_side),
raised if needed so that the template's short side stays >= 12 px.
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
BENCH = HERE / "redp_x40"
OUT = HERE / "results"
SCALES = (0.8, 0.9, 1.0, 1.1, 1.25)
ROTS = (0, 90, 180, 270)
FLOOR = 0.30
MAX_CAND = 3000


def load(did):
    ann = json.loads((BENCH / "annotations" / f"{did}.json").read_text())
    img = cv2.imread(str(BENCH / ann["image"]), cv2.IMREAD_GRAYSCALE)
    tpl = cv2.imread(str(BENCH / ann["template"]), cv2.IMREAD_GRAYSCALE)
    return ann, img, tpl


def rot(t, deg):
    return {0: t, 90: cv2.rotate(t, cv2.ROTATE_90_CLOCKWISE),
            180: cv2.rotate(t, cv2.ROTATE_180), 270: cv2.rotate(t, cv2.ROTATE_90_COUNTERCLOCKWISE)}[deg]


def detect(did):
    ann, img, tpl = load(did)
    H, W = img.shape
    th, tw = tpl.shape
    s = min(1.0, 4096 / max(W, H))
    s = max(s, min(1.0, 12 / min(th, tw)))
    im = cv2.resize(img, (round(W * s), round(H * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
    best = np.full(im.shape, -1.0, np.float32)
    bvar = np.full(im.shape, -1, np.int16)
    variants = []
    for sc in SCALES:
        k = s * sc
        t = cv2.resize(tpl, (max(3, round(tw * k)), max(3, round(th * k))),
                       interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
        for r in ROTS:
            tt = rot(t, r)
            if tt.shape[0] >= im.shape[0] or tt.shape[1] >= im.shape[1] or tt.std() < 1e-3:
                continue
            res = cv2.matchTemplate(im, tt, cv2.TM_CCOEFF_NORMED)
            res = np.nan_to_num(res, nan=-1.0)
            hh, ww = tt.shape
            # map top-left response to center coordinates
            y0, x0 = hh // 2, ww // 2
            region = best[y0:y0 + res.shape[0], x0:x0 + res.shape[1]]
            vreg = bvar[y0:y0 + res.shape[0], x0:x0 + res.shape[1]]
            m = res > region
            region[m] = res[m]
            vreg[m] = len(variants)
            variants.append({"scale": sc, "rot": r, "w": ww, "h": hh})
    # NMS: local maxima within a window of the template's short side
    rad = max(2, int(0.5 * min(th, tw) * s))
    dil = cv2.dilate(best, np.ones((2 * rad + 1, 2 * rad + 1), np.uint8))
    peaks = np.argwhere((best >= dil) & (best >= FLOOR))
    scores = best[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(-scores)[:MAX_CAND]
    cands, taken = [], []
    for idx in order:
        y, x = peaks[idx]
        # greedy suppression of plateaus / near-duplicates
        if any((x - a) ** 2 + (y - b) ** 2 < rad ** 2 for a, b in taken[-400:]):
            continue
        taken.append((x, y))
        cands.append([round(x / s, 1), round(y / s, 1), round(float(best[y, x]), 4), int(bvar[y, x])])
    return {"id": did, "work_scale": s, "nms_radius_native": rad / s, "variants": variants,
            "candidates": cands}


def f1_count(pred, gt, tau):
    n, g = len(pred), len(gt)
    if n == 0 and g == 0:
        return 1.0, 0
    if n == 0 or g == 0:
        return 0.0, 0
    P, G = np.asarray(pred, float), np.asarray(gt, float)
    D = np.sqrt(((P[:, None, :] - G[None, :, :]) ** 2).sum(-1))
    r, c = linear_sum_assignment(D)
    tp = int((D[r, c] <= tau).sum())
    return 2 * tp / (n + g), tp


def taus(ann):
    tw, th = ann["template_size"]
    L = max(ann["width"], ann["height"])
    return {"sym": 0.5 * math.sqrt(tw * th), "1600": 30 / min(1.0, 1600 / L)}


def evaluate():
    anns = {p.stem: json.loads(p.read_text()) for p in sorted((BENCH / "annotations").glob("d*.json"))}
    cands = {d: json.loads((OUT / "tm_candidates" / f"{d}.json").read_text()) for d in anns}
    grid = np.round(np.arange(0.30, 0.991, 0.01), 2)
    # per drawing, per threshold: F1(sym), F1(1600), count
    table = {}
    for d, a in anns.items():
        gt = a["points"]
        T = taus(a)
        rows = []
        for t in grid:
            pred = [c[:2] for c in cands[d]["candidates"] if c[2] >= t]
            f_sym, _ = f1_count(pred, gt, T["sym"])
            f_16, _ = f1_count(pred, gt, T["1600"])
            rows.append((f_sym, f_16, len(pred)))
        table[d] = np.array(rows)
    ids = sorted(anns)
    res = {}
    for d in ids:
        others = [o for o in ids if o != d]
        mean_f = np.mean([table[o][:, 0] for o in others], axis=0)
        k_loo = int(np.argmax(mean_f))
        k_or = int(np.argmax(table[d][:, 0]))
        g = anns[d]["count"]
        res[d] = {"subset": anns[d]["subset"], "gt": g,
                  "loo": {"thr": float(grid[k_loo]), "n": int(table[d][k_loo, 2]),
                          "f1_sym": float(table[d][k_loo, 0]), "f1_1600": float(table[d][k_loo, 1])},
                  "oracle": {"thr": float(grid[k_or]), "n": int(table[d][k_or, 2]),
                             "f1_sym": float(table[d][k_or, 0]), "f1_1600": float(table[d][k_or, 1])}}
        for m in ("loo", "oracle"):
            r = res[d][m]
            r["abs_err"] = abs(r["n"] - g)
            r["rel_err"] = (r["n"] - g) / g
    (OUT / "tm_eval.json").write_text(json.dumps(res, indent=1))

    def summ(sub, m):
        v = [r[m] | {"gt": r["gt"]} for r in res.values() if sub in (None, r["subset"])]
        return {"n": len(v), "MAE": np.mean([x["abs_err"] for x in v]),
                "MRE": np.mean([abs(x["rel_err"]) for x in v]),
                "total_rel": sum(x["n"] - x["gt"] for x in v) / sum(x["gt"] for x in v),
                "F1_sym": np.mean([x["f1_sym"] for x in v]),
                "F1_1600": np.mean([x["f1_1600"] for x in v]),
                "exact": sum(x["abs_err"] == 0 for x in v)}
    lines = ["| Subset | Threshold | MAE | mean |rel. err| | F1 (tau_sym) | F1 (30 px @1600) | exact |",
             "|---|---|---|---|---|---|---|"]
    for sub in (None, "sheet", "view"):
        for m in ("loo", "oracle"):
            s = summ(sub, m)
            lines.append(f"| {sub or 'all'} | {m} | {s['MAE']:.1f} | {100*s['MRE']:.0f}% | "
                         f"{s['F1_sym']:.3f} | {s['F1_1600']:.3f} | {s['exact']}/{s['n']} |")
    txt = "\n".join(lines)
    (OUT / "tm_summary.md").write_text(txt + "\n")
    print(txt)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--detect", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    if a.detect:
        (OUT / "tm_candidates").mkdir(parents=True, exist_ok=True)
        ids = [a.only] if a.only else sorted(p.stem for p in (BENCH / "annotations").glob("d*.json"))
        for d in ids:
            p = OUT / "tm_candidates" / f"{d}.json"
            if p.exists() and not a.only:
                continue
            t0 = time.time()
            r = detect(d)
            p.write_text(json.dumps(r))
            print(f"{d}: {len(r['candidates'])} candidates, scale {r['work_scale']:.2f}, {time.time()-t0:.0f}s",
                  flush=True)
    if a.evaluate:
        evaluate()
