#!/usr/bin/env python3
"""
Whole-drawing prompt on REDP-X40 (construction drawings, legend-crop exemplar, no grid).

The paper uses PROMPTS["counts_first"] only (imported by main_test.py, gen_test.py, full_extra.py and
x4_vendor_highres.py); the list-first variant and the runner below belong to a later study and are kept for completeness.

counts_first = the REDP-X prompt used in the paper (07c_test_redp_x.py): "count"
               is written before the "shapes" list.
list_first   = identical prompt with the two JSON fields swapped and the user
               message asking for the list before the count.
Both keep the sentence "The count must equal the length of shapes".

Drawings are downscaled to 1600 px on the long side, as in the paper.

Usage:
  python3 e2_redpx_order.py --vendor claude --runs 3
  python3 e2_redpx_order.py --vendor qwen3vl --runs 3 --workers 4
  python3 e2_redpx_order.py --vendor gpt --runs 1 --max_items 1     # smoke
"""
import argparse
import json
from pathlib import Path

from PIL import Image

import arr_common as C

HERE = Path(__file__).resolve().parent

SYSTEM_COUNTS_FIRST = """\
You are a precise engineering drawing analysis assistant.

You will be shown a reference image of ONE target symbol, followed by a full
engineering drawing. Your task is to count every instance of that target
symbol in the drawing.

The drawing image is {canvas_w}x{canvas_h} pixels. All coordinates you output
must be in this image's pixel space, where (0,0) is the top-left corner.

Respond with ONLY a JSON object (no markdown, no backticks, no commentary). Format:
{{
  "target_type": "target_symbol",
  "count": <integer>,
  "shapes": [
    {{"type": "target_symbol", "cx": <pixel_x>, "cy": <pixel_y>}},
    ...
  ]
}}

The "shapes" array should contain one entry per instance of the target symbol
you found, with its center pixel coordinate. The "count" must equal the length
of "shapes". Count only symbols matching the reference image; do NOT count
other symbols, text labels, or drawing linework.
"""

SYSTEM_LIST_FIRST = """\
You are a precise engineering drawing analysis assistant.

You will be shown a reference image of ONE target symbol, followed by a full
engineering drawing. Your task is to count every instance of that target
symbol in the drawing.

The drawing image is {canvas_w}x{canvas_h} pixels. All coordinates you output
must be in this image's pixel space, where (0,0) is the top-left corner.

Respond with ONLY a JSON object (no markdown, no backticks, no commentary). Format:
{{
  "target_type": "target_symbol",
  "shapes": [
    {{"type": "target_symbol", "cx": <pixel_x>, "cy": <pixel_y>}},
    ...
  ],
  "count": <integer>
}}

The "shapes" array should contain one entry per instance of the target symbol
you found, with its center pixel coordinate. The "count" must equal the length
of "shapes". Count only symbols matching the reference image; do NOT count
other symbols, text labels, or drawing linework.
"""

USER_COUNTS_FIRST = """\
The first image is the reference crop of the target symbol (taken from this
drawing's legend). The second image is the full drawing.

Count every instance of the target symbol in the full drawing. Instances may
vary slightly in scale or rotation but share the reference's distinctive shape.

Respond with JSON only. Be thorough; do not miss any instances, do not invent ones."""

USER_LIST_FIRST = USER_COUNTS_FIRST.replace(
    "Count every instance of the target symbol in the full drawing.",
    "List every instance of the target symbol in the full drawing, then count the list.")

PROMPTS = {"counts_first": (SYSTEM_COUNTS_FIRST, USER_COUNTS_FIRST),
           "list_first": (SYSTEM_LIST_FIRST, USER_LIST_FIRST)}


def check_counts_first_matches_paper():
    """counts_first must equal the paper's REDP-X no-grid prompt (07c), byte for byte."""
    p = HERE.parent / "07c_test_redp_x.py"
    if not p.exists():
        print("  (07c_test_redp_x.py not found; prompt identity check skipped)")
        return
    src, ns = p.read_text(), {}
    for name in ("SYSTEM_TEMPLATE", "USER_TEMPLATE"):
        start = src.index(f'{name} = """')
        end = src.index('"""', start + len(name) + 7) + 3
        exec(src[start:end], ns)
    aids = C.REAL.build_spatial_aids_section(False)
    a = ns["SYSTEM_TEMPLATE"].format(canvas_w=1600, canvas_h=900, spatial_aids_section=aids)
    b = SYSTEM_COUNTS_FIRST.format(canvas_w=1600, canvas_h=900)
    assert a == b, "counts_first system prompt differs from the paper's REDP-X prompt"
    assert ns["USER_TEMPLATE"] == USER_COUNTS_FIRST, "counts_first user prompt differs"
    print("  counts_first prompt == paper REDP-X prompt (verified)")


def load_items(image_dir, gt_dir, max_side=1600):
    items = []
    for g in sorted(Path(gt_dir).glob("gt_redpx_*.json")):
        gt = json.loads(g.read_text())
        img = Image.open(Path(image_dir) / gt["image_file"]).convert("RGB")
        w, h = img.size
        s = min(1.0, max_side / max(w, h))
        if s < 1:
            img = img.resize((round(w * s), round(h * s)), Image.LANCZOS)
        ref = Image.open(Path(image_dir) / gt["template_file"]).convert("RGB")
        pts = [{"cx": round(p["cx"] * s), "cy": round(p["cy"] * s)} for p in gt["shapes"]]
        items.append({"item_id": gt["drawing_id"], "img": img, "ref": ref,
                      "gt_count": len(pts), "gt_points": pts, "scale": s})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vendor", required=True, choices=list(C.ALL_VENDORS))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--image_dir", default=str(HERE.parent / "redpx_images"))
    ap.add_argument("--gt_dir", default=str(HERE.parent / "ground_truth_redpx"))
    ap.add_argument("--out_dir", default=str(HERE / "results" / "e2"))
    ap.add_argument("--max_items", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    C.load_env()
    check_counts_first_matches_paper()

    items = load_items(args.image_dir, args.gt_dir)[: args.max_items]
    by_id = {it["item_id"]: it for it in items}
    jobs = [{"item_id": it["item_id"], "order": o, "run": r, "vendor": args.vendor}
            for r in range(args.runs) for it in items for o in PROMPTS]

    def work(job):
        it = by_id[job["item_id"]]
        system, user = PROMPTS[job["order"]]
        system = system.format(canvas_w=it["img"].size[0], canvas_h=it["img"].size[1])
        raw, usage = C.call(job["vendor"], [it["ref"], it["img"]], system, user)
        parsed = C.parse(raw)
        shapes = parsed.get("shapes", []) if isinstance(parsed, dict) else []
        if not isinstance(shapes, list):
            shapes = []
        return {**job, "exp": "e2", "dataset": "redpx", "model": C.ALL_VENDORS[job["vendor"]],
                "status": "parse_error" if parsed.get("parse_error") else "success",
                "declared_count": parsed.get("count"), "list_count": len(shapes),
                "shapes": shapes, "gt_count": it["gt_count"], "gt_points": it["gt_points"],
                "canvas": list(it["img"].size), "scale": it["scale"],
                "usage": usage, "raw": raw}

    out = Path(args.out_dir) / f"{args.vendor}.jsonl"
    C.run_jobs(jobs, out, work, key_fn=lambda r: (r["item_id"], r["order"], r["run"]),
               workers=args.workers)


if __name__ == "__main__":
    main()
