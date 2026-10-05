"""
03_test_anchor_only.py

Single-image pipeline: overlay sparse internal anchors (semi-transparent
magenta crosshairs) on each test image and query the VLM. The model's
output coordinates are in the original image frame, so no remapping is
needed. No ruler, no slicing, no NMS.

Default configuration:
  - 100 px anchor grid (no padding; grid starts at image (0, 0))
  - Semi-transparent magenta crosshair, opacity = 120 / 255
  - Model: claude-sonnet-4-6

Usage:
    python 03_test_anchor_only.py
    python 03_test_anchor_only.py --no_anchor          # ablation: no anchor
    python 03_test_anchor_only.py --anchor_interval 150

"""

import argparse
import base64
import json
import math
import os
import re
import time
from io import BytesIO
from pathlib import Path

import anthropic
from PIL import Image, ImageDraw


# ═══════════════════════════════════════════════════════════════════════
#  Anchor rendering: overlay sparse crosshair markers
# ═══════════════════════════════════════════════════════════════════════

def add_sparse_anchors(img: Image.Image, anchor_interval: int = 100,
                       anchor_color: str = "#FF00FF", anchor_opacity: int = 120,
                       crosshair_size: int = 16) -> Image.Image:
    """
    Overlay sparse semi-transparent crosshair markers (no text labels).
    Markers are placed on a regular grid starting at (0, 0); the last
    row/column may be partially off-canvas.
    """
    img = img.copy()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if anchor_color.startswith("#"):
        r = int(anchor_color[1:3], 16)
        g = int(anchor_color[3:5], 16)
        b = int(anchor_color[5:7], 16)
    else:
        r, g, b = 255, 0, 255

    color_rgba = (r, g, b, anchor_opacity)
    w, h = img.size
    half = crosshair_size // 2

    for x in range(0, w, anchor_interval):
        for y in range(0, h, anchor_interval):
            # Skip the origin to avoid the visual corner of the canvas
            if x == 0 and y == 0:
                continue
            draw.line([(x - half, y), (x + half, y)], fill=color_rgba, width=1)
            draw.line([(x, y - half), (x, y + half)], fill=color_rgba, width=1)

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return img.convert("RGB")


# ═══════════════════════════════════════════════════════════════════════
#  Prompt construction
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are a precise visual counting and localization assistant.

Given an image containing geometric shapes, identify and count every shape.

The image is {canvas_w}x{canvas_h} pixels. All coordinates you output must be in this image's pixel space, where (0,0) is the top-left corner. Output raw pixel coordinates as you see them.
{spatial_aids_section}
Respond with ONLY a JSON object (no markdown, no backticks, no commentary). Example format:
{{
  "counts_by_type": {{
    "circle": 5,
    "triangle": 3,
    "square": 0,
    "star": 2,
    "pentagon": 4
  }},
  "shapes": [
    {{"type": "circle", "cx": 412, "cy": 318}},
    {{"type": "triangle", "cx": 875, "cy": 642}}
  ]
}}

Only include shape types that actually appear in the image; use 0 for absent types or omit them.
Shape types to recognize: circle, triangle, square, star, pentagon.
Locate every shape carefully. Count systematically; do not miss any and do not invent shapes that are not there.
"""

USER_MESSAGE = "Count every geometric shape in this image. Output counts by type and each shape's pixel-coordinate center. Respond with JSON only."


def build_spatial_aids(has_anchor: bool) -> str:
    """Returns an empty string when no anchor is used, so the prompt matches the baseline exactly."""
    if not has_anchor:
        return ""
    return ("\nSpatial aids present in the image:\n"
            "- Faint semi-transparent magenta '+' crosshair markers placed at a regular grid "
            "inside the image, providing nearby visual reference points. "
            "Crosshairs are reference markers only — they are NOT shapes to count.\n"
            "Use the crosshair reference points to estimate each shape's pixel coordinate as accurately as possible.\n")


# ═══════════════════════════════════════════════════════════════════════
#  VLM call
# ═══════════════════════════════════════════════════════════════════════

def encode_pil_image(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def query_lvm(client, img: Image.Image, model: str,
              has_anchor: bool = True) -> tuple:
    """Single-image query. Returns (parsed_dict, usage_dict)."""
    canvas_w, canvas_h = img.size
    b64 = encode_pil_image(img)

    spatial_section = build_spatial_aids(has_anchor)
    system = SYSTEM_PROMPT.format(
        canvas_w=canvas_w, canvas_h=canvas_h,
        spatial_aids_section=spatial_section,
    )

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                {"type": "text", "text": USER_MESSAGE},
            ],
        }],
    )
    raw = response.content[0].text.strip()
    parsed = extract_json(raw)
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "actual_model": getattr(response, "model", None),  # snapshot served by the API
        "stop_reason": getattr(response, "stop_reason", None),
    }
    cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0)
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0)
    if cache_creation:
        usage["cache_creation_input_tokens"] = cache_creation
    if cache_read:
        usage["cache_read_input_tokens"] = cache_read
    return parsed, usage


# ═══════════════════════════════════════════════════════════════════════
#  JSON extraction (robust parser shared with 05)
# ═══════════════════════════════════════════════════════════════════════

def extract_json(text: str) -> dict:
    """Extract a JSON object from the model's raw output. Returns the full raw_response on failure."""
    last_error = None

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        last_error = e

    # 2. Strip markdown fences
    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = re.sub(r"```", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        last_error = e

    # 3. First { ... } region
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            last_error = e

    # 4. Brace-depth scan
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1])
                    if isinstance(obj, dict) and ("counts_by_type" in obj or "shapes" in obj):
                        return obj
                except json.JSONDecodeError as e:
                    last_error = e
                start = None

    # 5. Strip trailing commas
    cleaned2 = re.sub(r",\s*([\]}])", r"\1", cleaned)
    try:
        return json.loads(cleaned2)
    except json.JSONDecodeError as e:
        last_error = e

    # 6. Truncation recovery
    if '"shapes"' in cleaned:
        try:
            truncated = re.sub(r",\s*\{[^{}]*$", "", cleaned)
            open_braces = truncated.count("{") - truncated.count("}")
            open_brackets = truncated.count("[") - truncated.count("]")
            patched = truncated + "]" * open_brackets + "}" * open_braces
            return json.loads(patched)
        except json.JSONDecodeError as e:
            last_error = e

    # 7. Regex fallback
    if '"shapes"' in cleaned:
        try:
            counts_match = re.search(r'"counts_by_type"\s*:\s*(\{[^{}]*\})', cleaned)
            counts_by_type = {}
            if counts_match:
                try:
                    counts_by_type = json.loads(counts_match.group(1))
                except json.JSONDecodeError:
                    pass

            shape_pattern = re.compile(
                r'\{\s*"type"\s*:\s*"(\w+)"\s*,\s*"cx"\s*:\s*(-?\d+)\s*,\s*"cy"\s*:\s*(-?\d+)\s*\}'
            )
            shapes = [
                {"type": m.group(1), "cx": int(m.group(2)), "cy": int(m.group(3))}
                for m in shape_pattern.finditer(cleaned)
            ]
            if shapes or counts_by_type:
                return {
                    "counts_by_type": counts_by_type or {
                        s["type"]: sum(1 for x in shapes if x["type"] == s["type"]) for s in shapes
                    },
                    "shapes": shapes,
                    "_parser_recovered": True,
                }
        except Exception as e:
            last_error = e

    err_msg = f"{type(last_error).__name__}: {last_error}" if last_error else "unknown"
    return {
        "raw_response": text,
        "parse_error": True,
        "raw_length": len(text),
        "decode_error": err_msg,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Per-image processing
# ═══════════════════════════════════════════════════════════════════════

def process_one_image(client, img: Image.Image, model: str, args) -> tuple:
    """Run anchor + VLM on one image. Returns (result_dict, rendered_image)."""
    if args.no_anchor:
        final_img = img
    else:
        final_img = add_sparse_anchors(
            img,
            anchor_interval=args.anchor_interval,
            anchor_color=args.anchor_color,
            anchor_opacity=args.anchor_opacity,
        )

    result, usage = query_lvm(client, final_img, model, has_anchor=not args.no_anchor)

    if result.get("parse_error"):
        result["status"] = "parse_error"
        result["failure_reason"] = "JSON parse failed"
        result["usage"] = usage
        return result, final_img

    # Output coordinates are already in the original image frame (no padding to undo)
    counts_by_type = {}
    for s in result.get("shapes", []):
        counts_by_type[s["type"]] = counts_by_type.get(s["type"], 0) + 1

    return {
        "counts_by_type": counts_by_type,
        "shapes": result.get("shapes", []),
        "usage": usage,
    }, final_img


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Anchor-only counting test")
    parser.add_argument("--model", default="claude-sonnet-4-6",
                        help="model id (default: claude-sonnet-4-6)")
    parser.add_argument("--images_dir", default="test_images")
    parser.add_argument("--gt_dir", default="ground_truth")
    parser.add_argument("--delay", type=float, default=2.0)

    parser.add_argument("--anchor_interval", type=int, default=100,
                        help="anchor grid spacing (px)")
    parser.add_argument("--anchor_color", default="#FF00FF")
    parser.add_argument("--anchor_opacity", type=int, default=120,
                        help="anchor alpha (0-255)")
    parser.add_argument("--no_anchor", action="store_true",
                        help="ablation: do not draw the anchor")

    parser.add_argument("--no_save_processed", action="store_true",
                        help="do not save the rendered image sent to the VLM")
    parser.add_argument("--max_images", type=int, default=None,
                        help="run on at most N images (default: all)")

    args = parser.parse_args()

    # Output-directory model tag, e.g. "claude-sonnet-4-20250514" -> "sonnet_4"
    model_tag = args.model.replace("claude-", "").replace("-", "_")
    parts = model_tag.split("_")
    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) > 4:
        model_tag = "_".join(parts[:-1])

    # Naming convention:
    #   results_anchor_{model_tag}/         (default: anchor on)
    #   results_anchor_noaid_{model_tag}/   (--no_anchor ablation)
    if args.no_anchor:
        tag = f"anchor_noaid_{model_tag}"
    else:
        tag = f"anchor_{model_tag}"
    results_dir = Path(f"results_{tag}")
    results_dir.mkdir(exist_ok=True)

    client = anthropic.Anthropic()
    images = sorted(Path(args.images_dir).glob("test_*.png"))
    if args.max_images:
        images = images[:args.max_images]
    print(f"Found {len(images)} test images")
    print(f"Model: {model_tag}")
    if args.no_anchor:
        print(f"Anchor: DISABLED (--no_anchor)")
    else:
        print(f"Anchor: {args.anchor_interval}px grid, opacity={args.anchor_opacity}")
    print()

    config = vars(args).copy()
    config["model_tag"] = model_tag
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_results = []
    gts = {}
    for gt_path in sorted(Path(args.gt_dir).glob("gt_*.json")):
        with open(gt_path) as f:
            g = json.load(f)
            gts[g["image_id"]] = g

    processed_dir = results_dir / "processed"
    if not args.no_save_processed:
        processed_dir.mkdir(exist_ok=True)

    for idx, img_path in enumerate(images):
        img_id = int(img_path.stem.replace("test_", ""))
        gt = gts.get(img_id, {"counts_by_type": {}, "shapes": []})

        print(f"  [->] processing {img_path.name} ... ", end="", flush=True)
        try:
            img = Image.open(img_path)

            result, rendered_img = process_one_image(client, img, args.model, args)

            # Save the actual image sent to the VLM (kept by default for debugging)
            if not args.no_save_processed:
                rendered_img.save(str(processed_dir / f"processed_{img_id:03d}.png"))

            if result.get("parse_error"):
                result["status"] = "parse_error"
                result["failure_reason"] = result.get("failure_reason", "JSON parse failed")
                raw = result.get("raw_response", "")
                raw_len = result.get("raw_length", len(raw))
                print(f"PARSE_ERROR (len={raw_len})")
                decode_err = result.get("decode_error")
                if decode_err:
                    print(f"    decode error: {decode_err}")
                if raw:
                    head = raw[:150].replace("\n", "\\n")
                    tail = raw[-150:].replace("\n", "\\n") if len(raw) > 300 else ""
                    print(f"    head: {head}")
                    if tail:
                        print(f"    tail: {tail}")
            else:
                result["status"] = "success"
                pred_types = result.get("counts_by_type", {})
                gt_types = gt.get("counts_by_type", {})
                all_t = sorted(set(gt_types.keys()) | set(pred_types.keys()))
                per_type_err = 0
                type_diffs = []
                for t in all_t:
                    p = pred_types.get(t, 0)
                    g = gt_types.get(t, 0)
                    per_type_err += abs(p - g)
                    if p != g:
                        type_diffs.append(f"{t}:{p}/{g}")
                if per_type_err == 0:
                    status = "OK"
                else:
                    diffs_str = " ".join(type_diffs)
                    status = f"ERR err={per_type_err} ({diffs_str})"
                print(status)
        except Exception as e:
            result = {
                "status": "api_error",
                "failure_reason": str(type(e).__name__),
                "error": str(e),
            }
            print(f"API_ERROR: {e}")

        result["image_id"] = img_id
        result["image_file"] = str(img_path)
        result["method"] = tag
        result["model"] = args.model
        all_results.append(result)

        if idx < len(images) - 1:
            time.sleep(args.delay)

    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDone -> {results_dir}/")


if __name__ == "__main__":
    main()
