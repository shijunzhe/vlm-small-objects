"""
_shared.py

Shared utilities for the cross-model experiments. Reuses the prompt and
parser from 03_test_anchor_only.py while abstracting the LVM call so that
GPT and Gemini scripts only need to provide their own query_vlm function.

Usage:
    python _shared.py [args]

"""

import importlib.util
import json
import sys
import time
from pathlib import Path
from PIL import Image

# Load module 03 (filename starts with a digit, so use importlib)
_grid_test_dir = Path(__file__).parent
_spec = importlib.util.spec_from_file_location(
    "_anchor_only", _grid_test_dir / "anchor_only.py"
)
_anchor_only = importlib.util.module_from_spec(_spec)
sys.modules["_anchor_only"] = _anchor_only
_spec.loader.exec_module(_anchor_only)

# Re-export for downstream scripts
SYSTEM_PROMPT = _anchor_only.SYSTEM_PROMPT
USER_MESSAGE = _anchor_only.USER_MESSAGE
build_spatial_aids = _anchor_only.build_spatial_aids
extract_json = _anchor_only.extract_json
add_sparse_anchors = _anchor_only.add_sparse_anchors
encode_pil_image = _anchor_only.encode_pil_image


def build_prompt(img: Image.Image, has_anchor: bool) -> tuple:
    """Build the same system + user prompt as in 03.

    Returns: (system_str, user_str)
    """
    canvas_w, canvas_h = img.size
    spatial_section = build_spatial_aids(has_anchor)
    system = SYSTEM_PROMPT.format(
        canvas_w=canvas_w, canvas_h=canvas_h,
        spatial_aids_section=spatial_section,
    )
    return system, USER_MESSAGE


def render_image(img: Image.Image, anchor_interval: int = None,
                 anchor_color: str = "#FF00FF",
                 anchor_opacity: int = 120) -> Image.Image:
    """Return the final image to send to the LVM, depending on anchor settings.

    anchor_interval=None -> baseline (no anchors).
    """
    if anchor_interval is None:
        return img
    return add_sparse_anchors(
        img,
        anchor_interval=anchor_interval,
        anchor_color=anchor_color,
        anchor_opacity=anchor_opacity,
    )


def post_process(parsed: dict) -> dict:
    """Extract counts_by_type and shapes from parsed JSON (matches 03 behavior)."""
    if parsed.get("parse_error"):
        return {
            "status": "parse_error",
            "failure_reason": "JSON parse failure",
            "raw_response": parsed.get("raw_response", ""),
            "decode_error": parsed.get("decode_error"),
        }
    counts_by_type = {}
    for s in parsed.get("shapes", []):
        t = s.get("type")
        if t:
            counts_by_type[t] = counts_by_type.get(t, 0) + 1
    return {
        "status": "success",
        "counts_by_type": counts_by_type,
        "shapes": parsed.get("shapes", []),
    }


def load_test_data(images_dir: Path, gt_dir: Path, max_images: int = None):
    """Load test images and ground truth. Returns (images_list, gts_dict)."""
    images = sorted(images_dir.glob("test_*.png"))
    if max_images:
        images = images[:max_images]
    gts = {}
    for gt_path in sorted(gt_dir.glob("gt_*.json")):
        with open(gt_path) as f:
            g = json.load(f)
            gts[g["image_id"]] = g
    return images, gts


def run_experiment(
    query_vlm_fn,
    config_name: str,
    model_id: str,
    images: list,
    gts: dict,
    results_dir: Path,
    anchor_interval: int = None,  # None = baseline
    anchor_color: str = "#FF00FF",
    anchor_opacity: int = 120,
    delay: float = 0.3,
    save_processed: bool = True,
):
    """Run one experiment configuration. query_vlm_fn is the model-specific call,
    with signature: query_vlm_fn(img: PIL.Image, system: str, user: str, model: str) -> dict.
    The returned dict must contain: parsed (matching the 03 JSON structure) plus usage
    (input/output tokens, etc.).
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    if save_processed:
        processed_dir = results_dir / "processed"
        processed_dir.mkdir(exist_ok=True)

    config = {
        "config_name": config_name,
        "model": model_id,
        "anchor_interval": anchor_interval,
        "anchor_color": anchor_color,
        "anchor_opacity": anchor_opacity,
    }
    with open(results_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    has_anchor = anchor_interval is not None
    all_results = []

    for idx, img_path in enumerate(images):
        img_id = int(img_path.stem.replace("test_", ""))
        gt = gts.get(img_id, {"counts_by_type": {}, "shapes": []})

        print(f"  [->] {img_path.name} ... ", end="", flush=True)
        try:
            img = Image.open(img_path)
            rendered = render_image(img, anchor_interval, anchor_color, anchor_opacity)

            if save_processed:
                rendered.save(str(processed_dir / f"processed_{img_id:03d}.png"))

            system, user = build_prompt(rendered, has_anchor)

            # Model-specific call
            api_result = query_vlm_fn(rendered, system, user, model_id)
            parsed = api_result["parsed"]
            usage = api_result["usage"]

            processed_result = post_process(parsed)
            processed_result["usage"] = usage

            if processed_result["status"] == "parse_error":
                print(f"PARSE_ERROR (raw len={len(processed_result.get('raw_response', ''))})")
            else:
                pred_counts = processed_result["counts_by_type"]
                gt_counts = gt.get("counts_by_type", {})
                all_t = sorted(set(gt_counts.keys()) | set(pred_counts.keys()))
                err = sum(abs(pred_counts.get(t, 0) - gt_counts.get(t, 0)) for t in all_t)
                if err == 0:
                    print("OK")
                else:
                    diffs = " ".join(
                        f"{t}:{pred_counts.get(t, 0)}/{gt_counts.get(t, 0)}"
                        for t in all_t
                        if pred_counts.get(t, 0) != gt_counts.get(t, 0)
                    )
                    print(f"ERR err={err} ({diffs})")
        except Exception as e:
            processed_result = {
                "status": "api_error",
                "failure_reason": type(e).__name__,
                "error": str(e),
            }
            print(f"API_ERROR: {e}")

        processed_result["image_id"] = img_id
        processed_result["image_file"] = str(img_path)
        processed_result["method"] = config_name
        processed_result["model"] = model_id
        if anchor_interval is not None:
            processed_result["anchor_interval"] = anchor_interval

        all_results.append(processed_result)

        if idx < len(images) - 1:
            time.sleep(delay)

    with open(results_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results
