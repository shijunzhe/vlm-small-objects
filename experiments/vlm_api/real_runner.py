#!/usr/bin/env python3
"""
07_test_real_drawing.py

Real electrical drawing test script. Reuses the cross_model three-model API
calling framework, but the task changes from synthetic shape counting to real
electrical symbol counting. Each call asks for one symbol type only.

Usage:
    python 07_test_real_drawing.py [args]

"""

import argparse
import base64
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path
from PIL import Image

# Load shared module from cross_model (reuse add_sparse_anchors + extract_json)
script_dir = Path(__file__).parent
cross_model_dir = script_dir
sys.path.insert(0, str(cross_model_dir))
import _shared  # loads the 03 module


# ═══════════════════════════════════════════════════════════════════════
# Symbol descriptions: text description + reference image filename
# ═══════════════════════════════════════════════════════════════════════

SYMBOL_INFO = {
    "Step light": {
        "description": "small rectangular fixture with horizontal lines and arrow indicator, often labeled '3'",
        "reference_file": "step_light_symbol.jpeg",
    },
    "Weatherproof GFCI duplex receptacle": {
        "description": "circle with horizontal line through it, labeled 'WP/GFCI' below",
        "reference_file": "Weatherproof_GFCI_duplex_receptacle_symbol.jpeg",
    },
    "Switch": {
        "description": "small dollar-sign-like symbol with subscripts (e.g., '$' or '$$')",
        "reference_file": "switch_symbol.jpeg",
    },
    "Wall mounted cylinder light": {
        "description": "small filled black semi-circle on a wall edge",
        "reference_file": "wall_mounted_cylinder_light_symbol.jpeg",
    },
    "GFCI outlet": {
        "description": "small filled circle inside a larger circle (concentric)",
        "reference_file": "GFCI_outlet_symbol.jpeg",
    },
    "Speaker": {
        "description": "circle with radiating short lines (sun/star-like)",
        "reference_file": "spearker_symbol.jpeg",
    },
    "Hose bibb": {
        "description": "vertical line with a small horizontal cross at top, labeled 'HB'",
        "reference_file": "hose_bibb_symbol.jpeg",
    },
    "Ceiling mounted light": {
        "description": "small circle with crosshair pattern (plus sign through it)",
        "reference_file": "Ceiling_mounted_light_symbol.jpeg",
    },
    "Fan": {
        "description": "circle with X-shaped fan blades inside, often dashed outer ring",
        "reference_file": "Fan_symbol.jpeg",
    },
    "Wall mounted light": {
        "description": "small circle with rays extending outward (sun-like)",
        "reference_file": "Wall_mounted_light_symbol.jpeg",
    },
    "Water connection": {
        "description": "triangle with W inside, labeled 'W'",
        "reference_file": "water_connection_symbol.jpeg",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Prompt construction
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_TEMPLATE = """\
You are a precise electrical drawing analysis assistant.

You will be shown an architectural floor plan with electrical symbols.
Your task is to count every instance of a SPECIFIC symbol type in the drawing.

The image is {canvas_w}x{canvas_h} pixels. All coordinates you output must be
in this image's pixel space, where (0,0) is the top-left corner.
{spatial_aids_section}
Respond with ONLY a JSON object (no markdown, no backticks, no commentary). Format:
{{
  "target_type": "<the symbol you were asked to count>",
  "count": <integer>,
  "shapes": [
    {{"type": "<target_type>", "cx": <pixel_x>, "cy": <pixel_y>}},
    ...
  ]
}}

The "shapes" array should contain one entry per instance of the target symbol you found,
with its center pixel coordinate. The "count" must equal the length of "shapes".
Only output the target symbol type, do NOT include other symbols.
"""

USER_TEMPLATE = """\
Count every instance of '{target_type}' in this electrical floor plan.

Visual description of {target_type}:
{description}

{reference_note}Respond with JSON only. Be thorough; do not miss any instances, do not invent ones."""


def build_spatial_aids_section(has_anchor: bool) -> str:
    if not has_anchor:
        return ""
    return ("\nSpatial aids present in the image:\n"
            "- Faint semi-transparent magenta '+' crosshair markers placed at a regular grid "
            "inside the image, providing nearby visual reference points. "
            "Crosshairs are reference markers only; they are NOT symbols to count.\n"
            "Use the crosshair reference points to estimate each symbol's pixel coordinate as accurately as possible.\n")


def build_prompts(canvas_w: int, canvas_h: int, target_type: str,
                  has_anchor: bool, has_reference: bool) -> tuple:
    """Assemble system + user prompt."""
    system = SYSTEM_TEMPLATE.format(
        canvas_w=canvas_w, canvas_h=canvas_h,
        spatial_aids_section=build_spatial_aids_section(has_anchor),
    )
    info = SYMBOL_INFO.get(target_type, {})
    desc = info.get("description", target_type)
    ref_note = ("A reference image of this symbol is provided below, before the floor plan.\n"
                if has_reference else "")
    user = USER_TEMPLATE.format(
        target_type=target_type,
        description=desc,
        reference_note=ref_note,
    )
    return system, user


# ═══════════════════════════════════════════════════════════════════════
# Three-model API calls - all accept (main image + optional reference image)
# ═══════════════════════════════════════════════════════════════════════

def encode_pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def encode_pil_to_dataurl(img: Image.Image) -> str:
    return f"data:image/png;base64,{encode_pil_to_b64(img)}"


def query_claude(main_img: Image.Image, ref_img, system: str, user: str, model: str) -> dict:
    """Claude API. ref_img can be None or PIL Image."""
    import anthropic
    client = anthropic.Anthropic()

    content = []
    if ref_img is not None:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png",
                       "data": encode_pil_to_b64(ref_img)},
        })
    content.append({
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png",
                   "data": encode_pil_to_b64(main_img)},
    })
    content.append({"type": "text", "text": user})

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=system,
        messages=[{"role": "user", "content": content}],
    )
    raw = response.content[0].text.strip()
    parsed = _shared.extract_json(raw)
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "actual_model": response.model,
        "stop_reason": response.stop_reason,
        "raw_response_length": len(raw),
    }
    if parsed.get("parse_error"):
        usage["raw_response"] = raw
    return {"parsed": parsed, "usage": usage}


def query_gpt(main_img: Image.Image, ref_img, system: str, user: str, model: str) -> dict:
    import openai
    client = openai.OpenAI()

    content = []
    if ref_img is not None:
        content.append({"type": "image_url",
                        "image_url": {"url": encode_pil_to_dataurl(ref_img)}})
    content.append({"type": "image_url",
                    "image_url": {"url": encode_pil_to_dataurl(main_img)}})
    content.append({"type": "text", "text": user})

    response = client.chat.completions.create(
        model=model,
        max_completion_tokens=8192,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        reasoning_effort="none",  # Setup A: pure vision
    )
    raw = response.choices[0].message.content.strip() if response.choices[0].message.content else ""
    parsed = _shared.extract_json(raw)
    usage = {
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "actual_model": response.model,
        "stop_reason": response.choices[0].finish_reason,
        "raw_response_length": len(raw),
    }
    if hasattr(response.usage, "completion_tokens_details"):
        details = response.usage.completion_tokens_details
        if details and getattr(details, "reasoning_tokens", None):
            usage["reasoning_tokens"] = details.reasoning_tokens
    if parsed.get("parse_error"):
        usage["raw_response"] = raw
    return {"parsed": parsed, "usage": usage}


def query_gemini(main_img: Image.Image, ref_img, system: str, user: str, model: str) -> dict:
    from google import genai
    from google.genai import types
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)

    parts = []
    if ref_img is not None:
        ref_buf = BytesIO()
        ref_img.save(ref_buf, format="PNG")
        parts.append(types.Part.from_bytes(data=ref_buf.getvalue(), mime_type="image/png"))
    main_buf = BytesIO()
    main_img.save(main_buf, format="PNG")
    parts.append(types.Part.from_bytes(data=main_buf.getvalue(), mime_type="image/png"))
    parts.append(user)

    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=8192,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = client.models.generate_content(
        model=model, contents=parts, config=config,
    )
    raw = ""
    if response.candidates and response.candidates[0].content:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                raw += part.text
    raw = raw.strip()
    parsed = _shared.extract_json(raw)
    usage_meta = response.usage_metadata
    usage = {
        "input_tokens": usage_meta.prompt_token_count if usage_meta else 0,
        "output_tokens": usage_meta.candidates_token_count if usage_meta else 0,
        "actual_model": getattr(response, "model_version", model),
        "stop_reason": str(response.candidates[0].finish_reason) if response.candidates else None,
        "raw_response_length": len(raw),
    }
    if usage_meta and getattr(usage_meta, "thoughts_token_count", None):
        usage["thinking_tokens"] = usage_meta.thoughts_token_count
    if parsed.get("parse_error"):
        usage["raw_response"] = raw
    return {"parsed": parsed, "usage": usage}


MODEL_REGISTRY = {
    "claude": {"fn": query_claude, "default_id": "claude-sonnet-4-6"},
    "gpt": {"fn": query_gpt, "default_id": "gpt-5.4"},
    "gemini": {"fn": query_gemini, "default_id": "gemini-2.5-flash"},
}


# ═══════════════════════════════════════════════════════════════════════
# Single (model, type, config) experiment runner
# ═══════════════════════════════════════════════════════════════════════

def run_single_experiment(model_key: str, model_id: str, target_type: str,
                          anchor_interval, main_img: Image.Image,
                          ref_img, gt_count: int) -> dict:
    """Run one experiment. anchor_interval=None means baseline."""
    has_anchor = anchor_interval is not None

    # Render main image (optionally adding anchor)
    if has_anchor:
        rendered = _shared.add_sparse_anchors(
            main_img, anchor_interval=anchor_interval,
            anchor_color="#FF00FF", anchor_opacity=120,
        )
    else:
        rendered = main_img

    canvas_w, canvas_h = rendered.size
    system, user = build_prompts(canvas_w, canvas_h, target_type,
                                 has_anchor=has_anchor, has_reference=(ref_img is not None))

    query_fn = MODEL_REGISTRY[model_key]["fn"]

    print(f"  [->] {model_key}/{target_type}/{'anchor_'+str(anchor_interval) if has_anchor else 'baseline'} ... ",
          end="", flush=True)
    try:
        api_result = query_fn(rendered, ref_img, system, user, model_id)
        parsed = api_result["parsed"]
        usage = api_result["usage"]

        if parsed.get("parse_error"):
            print(f"PARSE_ERROR")
            return {
                "status": "parse_error",
                "target_type": target_type,
                "gt_count": gt_count,
                "pred_count": None,
                "usage": usage,
                "rendered_image": rendered,
            }

        pred_count = parsed.get("count", len(parsed.get("shapes", [])))
        shapes = parsed.get("shapes", [])
        err = abs(pred_count - gt_count)
        sym = "OK" if err == 0 else "ERR"
        print(f"{sym} pred={pred_count}/gt={gt_count} (err={err})")

        return {
            "status": "success",
            "target_type": target_type,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "shapes": shapes,
            "usage": usage,
            "rendered_image": rendered,
        }
    except Exception as e:
        print(f"API_ERROR: {type(e).__name__}: {e}")
        return {
            "status": "api_error",
            "target_type": target_type,
            "gt_count": gt_count,
            "error": str(e),
            "rendered_image": rendered,
        }


# ═══════════════════════════════════════════════════════════════════════
# Main program
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="real_drawing/data/Electric_plan.jpeg",
                        help="Path to electrical drawing")
    parser.add_argument("--gt", default="real_drawing/data/ground_truth_real/gt_real_000.json",
                        help="Path to GT JSON")
    parser.add_argument("--model", required=True, choices=["claude", "gpt", "gemini"],
                        help="Model to test")
    parser.add_argument("--model_id", default=None,
                        help="Model version (defaults to MODEL_REGISTRY entry)")
    parser.add_argument("--type", default=None,
                        help="Test a single symbol type. Mutually exclusive with --all_types")
    parser.add_argument("--all_types", action="store_true",
                        help="Run all 11 types")
    parser.add_argument("--config", choices=["baseline", "anchor_100", "anchor_150", "anchor_200"],
                        default=None, help="Run a single config. If unspecified, runs baseline + anchor_150")
    parser.add_argument("--reference_dir", default=None,
                        help="Directory containing the 11 symbol reference icons (enables few-shot)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory for results (defaults to results_real_<model>)")
    parser.add_argument("--no_save_processed", action="store_true",
                        help="Do not save anchor-rendered images")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="Interval between API calls (seconds)")

    args = parser.parse_args()

    if not args.type and not args.all_types:
        parser.error("Must specify --type <name> or --all_types")
    if args.type and args.all_types:
        parser.error("--type and --all_types are mutually exclusive")

    # Load main image + GT
    main_img = Image.open(args.image)
    if main_img.mode != "RGB":
        main_img = main_img.convert("RGB")
    with open(args.gt) as f:
        gt = json.load(f)
    print(f"Main image: {args.image} ({main_img.size})")
    print(f"GT: {args.gt}, {len(gt['shapes'])} keypoints, {len(gt['counts_by_type'])} types")
    print()

    # Decide which types to test
    if args.all_types:
        target_types = list(SYMBOL_INFO.keys())
    else:
        target_types = [args.type]
        if args.type not in SYMBOL_INFO:
            parser.error(f"Unknown type: {args.type}. Valid: {list(SYMBOL_INFO.keys())}")

    # Decide which configs to test
    if args.config:
        configs = [args.config]
    else:
        configs = ["baseline", "anchor_150"]

    config_to_interval = {
        "baseline": None,
        "anchor_100": 100,
        "anchor_150": 150,
        "anchor_200": 200,
    }

    # Model setup
    model_id = args.model_id or MODEL_REGISTRY[args.model]["default_id"]

    # Output directory
    output_dir = Path(args.output_dir or f"real_drawing/results/results_real_{args.model}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_save_processed:
        (output_dir / "processed").mkdir(exist_ok=True)

    # Load reference images (once, reused)
    reference_imgs = {}
    if args.reference_dir:
        ref_dir = Path(args.reference_dir)
        for sym_name, info in SYMBOL_INFO.items():
            ref_path = ref_dir / info["reference_file"]
            if ref_path.exists():
                ref = Image.open(ref_path)
                if ref.mode != "RGB":
                    ref = ref.convert("RGB")
                reference_imgs[sym_name] = ref
            else:
                print(f"WARN Reference image not found: {ref_path}")

    # Run experiments
    print(f"Model: {args.model} ({model_id})")
    print(f"Configs: {configs}")
    print(f"Types: {len(target_types)}")
    print(f"Few-shot ref: {'yes' if reference_imgs else 'no'}")
    print(f"Output: {output_dir}")
    print(f"=" * 70)

    all_results = []
    for cfg in configs:
        interval = config_to_interval[cfg]
        for target_type in target_types:
            gt_count = gt["counts_by_type"].get(target_type, 0)
            ref_img = reference_imgs.get(target_type) if reference_imgs else None

            result = run_single_experiment(
                model_key=args.model, model_id=model_id,
                target_type=target_type, anchor_interval=interval,
                main_img=main_img, ref_img=ref_img, gt_count=gt_count,
            )

            # Save processed image
            rendered = result.pop("rendered_image", None)
            if rendered is not None and not args.no_save_processed:
                # Filename: processed_<config>_<type_safe>.png
                type_safe = target_type.replace(" ", "_").replace("/", "_")
                out_path = output_dir / "processed" / f"processed_{cfg}_{type_safe}.png"
                rendered.save(str(out_path))

            result["config"] = cfg
            result["model_key"] = args.model
            result["model_id"] = model_id
            all_results.append(result)

            time.sleep(args.delay)

    # Save
    out_json = output_dir / "all_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)

    # Brief summary
    print()
    print(f"=" * 70)
    print(f"Result summary:")
    print()
    for cfg in configs:
        cfg_results = [r for r in all_results if r["config"] == cfg]
        success = [r for r in cfg_results if r["status"] == "success"]
        total_err = sum(abs(r["pred_count"] - r["gt_count"]) for r in success)
        n_correct = sum(1 for r in success if r["pred_count"] == r["gt_count"])
        print(f"  {cfg}: {len(success)}/{len(cfg_results)} succeeded, "
              f"total err={total_err}, {n_correct}/{len(success)} types correct")
        for r in success:
            err = abs(r["pred_count"] - r["gt_count"])
            mark = "OK" if err == 0 else "ERR"
            print(f"    {mark} {r['target_type']:50s} pred={r['pred_count']:3d} gt={r['gt_count']:3d}")
        print()

    print(f"Done -> {out_json}")


if __name__ == "__main__":
    main()
