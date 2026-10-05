#!/usr/bin/env python3
"""
arr_common.py  --  shared plumbing for the ARR extension experiments (E2-E5).

Design rules (same as E1):
  * Closed vendors (claude / gpt / gemini) are called through the release
    functions in real_drawing/07_test_real_drawing.py, which take a main image
    and an optional reference image, so API settings are byte-identical to the
    published Setup A (8192 output tokens, GPT reasoning_effort="none", Gemini
    thinking_budget=0, Claude without extended thinking).
  * Reasoning-on variants (E5) are separate, explicitly named vendors.
  * Open-weight models are reached through OpenRouter's OpenAI-compatible API
    with the same message layout as the release GPT call (images, then text)
    and the same 8192-token output limit; the serving provider is logged.
  * Every record stores the raw text, the declared count(s), the model's own
    list, the stop reason and token usage. Nothing is scored at collection time.
  * Output is JSONL, appended as results arrive, so runs resume after interruption.
"""

import base64
import importlib.util
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
RELEASE_ROOT = HERE


# ── config.env (keys are read, never printed) ───────────────────────────
def load_env(path=None):
    for p in [path, HERE / "config.env", HERE.parent / "config.env", HERE.parent.parent / "config.env"]:
        if p and Path(p).exists():
            for line in Path(p).read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip('"').strip("'")
                if v and not os.environ.get(k.strip()):
                    os.environ[k.strip()] = v
            return str(p)
    return None


# ── Release modules (read-only) ─────────────────────────────────────────
sys.path.insert(0, str(HERE))
import _shared  # noqa: E402

AO = sys.modules["_anchor_only"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REAL = _load("real_runner", HERE / "real_runner.py")

# Capture raw text per thread: the release functions parse with _shared.extract_json.
_TLS = threading.local()
_orig_extract = _shared.extract_json


def _capturing_extract(text):
    _TLS.raw = text
    return _orig_extract(text)


_shared.extract_json = _capturing_extract
AO.extract_json = _capturing_extract
REAL._shared.extract_json = _capturing_extract


# ── Vendors ─────────────────────────────────────────────────────────────
CLOSED = {
    "claude": "claude-sonnet-4-6",
    "gpt": "gpt-5.4",
    "gemini": "gemini-2.5-flash",
}
OPEN = {
    "qwen3vl": "qwen/qwen3-vl-235b-a22b-instruct",
    "gemma4": "google/gemma-4-31b-it",
}
REASONING = {
    "claude_think": "claude-sonnet-4-6",
    "gpt_reason": "gpt-5.4",
    "qwen3vl_think": "qwen/qwen3-vl-235b-a22b-thinking",
}
ALL_VENDORS = {**CLOSED, **OPEN, **REASONING}

# Pin one OpenRouter provider per open model so that all calls hit the same
# serving stack (providers can differ in quantization and image preprocessing).
PROVIDER_PIN = {
    "qwen/qwen3-vl-235b-a22b-instruct": "Alibaba",
    "qwen/qwen3-vl-235b-a22b-thinking": "Alibaba",
    "google/gemma-4-31b-it": "CoreWeave",
}

CLAUDE_THINK_BUDGET = 4096
MAX_TOKENS = 8192


def _b64(img):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _call_openai_compatible(client, model, images, system, user, extra=None, max_tokens=MAX_TOKENS):
    content = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_b64(im)}"}}
               for im in images]
    content.append({"type": "text", "text": user})
    kwargs = dict(model=model, messages=[{"role": "system", "content": system},
                                         {"role": "user", "content": content}])
    kwargs.update(extra or {})
    kwargs.setdefault("max_tokens", max_tokens)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return client.chat.completions.create(**kwargs)


_clients = {}
_clients_lock = threading.Lock()


def _client(kind):
    with _clients_lock:
        if kind not in _clients:
            if kind == "openrouter":
                import openai
                _clients[kind] = openai.OpenAI(base_url="https://openrouter.ai/api/v1",
                                               api_key=os.environ["OPENROUTER_API_KEY"],
                                               timeout=600)
            elif kind == "openai":
                import openai
                _clients[kind] = openai.OpenAI(timeout=600)
            elif kind == "anthropic":
                import anthropic
                _clients[kind] = anthropic.Anthropic(timeout=600)
        return _clients[kind]


def call(vendor, images, system, user):
    """images: [main] or [ref, main] (reference first, as in the release).
    Returns (raw_text, usage)."""
    _TLS.raw = None
    model = ALL_VENDORS[vendor]
    if vendor in CLOSED:
        main = images[-1]
        ref = images[0] if len(images) == 2 else None
        out = REAL.MODEL_REGISTRY[vendor]["fn"](main, ref, system, user, model)
        usage = {k: v for k, v in out["usage"].items() if k != "raw_response"}
        return _TLS.raw, usage

    if vendor in OPEN or vendor == "qwen3vl_think":
        pin = PROVIDER_PIN.get(model)
        extra = {"extra_body": {"provider": {"order": [pin], "allow_fallbacks": False}}} if pin else None
        resp = _call_openai_compatible(_client("openrouter"), model, images, system, user, extra=extra)
        ch = resp.choices[0]
        raw = (ch.message.content or "").strip()
        u = resp.usage
        usage = {"input_tokens": getattr(u, "prompt_tokens", None),
                 "output_tokens": getattr(u, "completion_tokens", None),
                 "actual_model": resp.model, "stop_reason": ch.finish_reason,
                 "provider": getattr(resp, "provider", None)}
        det = getattr(u, "completion_tokens_details", None)
        if det is not None and getattr(det, "reasoning_tokens", None):
            usage["reasoning_tokens"] = det.reasoning_tokens
        return raw, usage

    if vendor == "gpt_reason":
        resp = _call_openai_compatible(
            _client("openai"), model, images, system, user,
            extra={"reasoning_effort": "medium", "max_completion_tokens": 32768,
                   "max_tokens": None})
        ch = resp.choices[0]
        raw = (ch.message.content or "").strip()
        usage = {"input_tokens": resp.usage.prompt_tokens,
                 "output_tokens": resp.usage.completion_tokens,
                 "actual_model": resp.model, "stop_reason": ch.finish_reason,
                 "reasoning_effort": "medium"}
        det = getattr(resp.usage, "completion_tokens_details", None)
        if det is not None and getattr(det, "reasoning_tokens", None):
            usage["reasoning_tokens"] = det.reasoning_tokens
        return raw, usage

    if vendor == "claude_think":
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                "data": _b64(im)}} for im in images]
        content.append({"type": "text", "text": user})
        resp = _client("anthropic").messages.create(
            model=model, max_tokens=MAX_TOKENS + CLAUDE_THINK_BUDGET, system=system,
            thinking={"type": "enabled", "budget_tokens": CLAUDE_THINK_BUDGET},
            messages=[{"role": "user", "content": content}])
        raw = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
        usage = {"input_tokens": resp.usage.input_tokens,
                 "output_tokens": resp.usage.output_tokens,
                 "actual_model": resp.model, "stop_reason": resp.stop_reason,
                 "thinking_budget": CLAUDE_THINK_BUDGET}
        return raw, usage

    raise ValueError(vendor)


def parse(raw):
    """Release parser (with its recovery path); never raises."""
    if raw is None:
        return {"parse_error": True}
    try:
        return _orig_extract(raw)
    except Exception as e:  # pragma: no cover
        return {"parse_error": True, "error": str(e)}


# ── Resumable, threaded JSONL runner ─────────────────────────────────────
def run_jobs(jobs, out_path, work_fn, key_fn, workers=4, delay=0.0, max_retries=3):
    """jobs: list of dicts. work_fn(job) -> record dict (must include 'status').
    Records with status success/parse_error are considered done; api errors are
    retried with backoff and, if still failing, written with status api_error
    (and retried on the next invocation)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") in ("success", "parse_error"):
                done.add(key_fn(r))
    todo = [j for j in jobs if key_fn(j) not in done]
    print(f"{out_path}: {len(done)} done, {len(todo)} to go", flush=True)
    lock = threading.Lock()
    counter = {"n": 0}

    def _one(job):
        rec = None
        for attempt in range(max_retries):
            try:
                rec = work_fn(job)
                break
            except Exception as e:
                rec = {**{k: v for k, v in job.items() if not k.startswith("_")},
                       "status": "api_error", "error": f"{type(e).__name__}: {e}"[:500]}
                time.sleep(min(60, 5 * 2 ** attempt))
        rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with lock:
            with out_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            counter["n"] += 1
            if counter["n"] % 10 == 0 or rec["status"] != "success":
                print(f"  [{counter['n']}/{len(todo)}] {rec.get('status')} "
                      f"{ {k: rec.get(k) for k in ('item_id', 'order', 'grid', 'vendor')} } "
                      f"{rec.get('error', '')[:120]}", flush=True)
        if delay:
            time.sleep(delay)
        return rec

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(as_completed([ex.submit(_one, j) for j in todo]))
    return out_path
