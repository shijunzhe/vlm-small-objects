#!/usr/bin/env python3
"""Model calls for the visual-spec experiment: vendor-default reasoning (no reasoning parameters sent),
output cap 16,384 (Gemini: includes thinking), Qwen pinned to one OpenRouter provider without fallbacks.
Keys come from the repository's config.env (see config.env.example) and are never printed."""
import base64, io, os, json, time, threading
from pathlib import Path

MAX_OUT = 16384
QWEN_PROVIDER = os.environ.get('VSPEC_QWEN_PROVIDER', 'DeepInfra')   # Alibaba no longer serves this model on OpenRouter (phase 0, 2026-10-25); user chose DeepInfra (fp8)
IDS = {'gpt61': 'gpt-6.1-sol', 'gpt56': 'gpt-5.6-sol', 'opus55': 'claude-opus-5-5',
       'gem38': 'gemini-3.8-flash', 'gem4': 'gemini-4-argon', 'qwen3vl': 'qwen/qwen3-vl-235b-a22b-instruct'}
# USD per million tokens (input, output), checked 2026-10-25; Qwen uses the cost OpenRouter reports when present
PRICE = {'gpt61': (2.0, 10.0), 'gpt56': (4.0, 20.0), 'opus55': (4.0, 20.0), 'gem38': (0.75, 3.75), 'gem4': (2.0, 10.0),   # gem4: introductory price until 2026-11-30
         'qwen3vl': (0.5, 2.0)}
_lock = threading.Lock(); _clients = {}
_env_loaded = False


def load_env():
    global _env_loaded
    if _env_loaded:
        return
    for line in (Path(__file__).resolve().parents[2] / 'config.env').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1); os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    _env_loaded = True


def _png(im):
    b = io.BytesIO(); im.save(b, format='PNG'); return b.getvalue()


def _client(kind):
    load_env()
    with _lock:
        if kind not in _clients:
            if kind == 'anthropic':
                import anthropic; _clients[kind] = anthropic.Anthropic(timeout=900)
            elif kind == 'openai':
                import openai; _clients[kind] = openai.OpenAI(timeout=900)
            elif kind == 'openrouter':
                import openai; _clients[kind] = openai.OpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ['OPENROUTER_API_KEY'], timeout=900)
            else:
                from google import genai; from google.genai import types as _t
                _clients[kind] = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options=_t.HttpOptions(timeout=900000))
        return _clients[kind]


def cost(model, u):
    if u.get('cost') is not None:
        return float(u['cost'])
    pi, po = PRICE[model]
    out = (u.get('output_tokens') or 0) + (u.get('thinking_tokens') or 0 if model in ('gem38', 'gem4') else 0)
    return ((u.get('input_tokens') or 0) * pi + out * po) / 1e6


def call(model, images, system, user):
    """Returns (raw_text, usage). usage has input/output tokens, stop_reason, and provider for Qwen."""
    mid = IDS[model]
    if model == 'opus55':
        content = [{'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png',
                                                 'data': base64.standard_b64encode(_png(im)).decode()}} for im in images]
        content.append({'type': 'text', 'text': user})
        r = _client('anthropic').messages.create(model=mid, max_tokens=MAX_OUT, system=system,
                                                 messages=[{'role': 'user', 'content': content}])
        raw = ''.join(b.text for b in r.content if getattr(b, 'type', '') == 'text').strip()
        kinds = [getattr(b, 'type', '') for b in r.content]
        return raw, {'input_tokens': r.usage.input_tokens, 'output_tokens': r.usage.output_tokens,
                     'actual_model': r.model, 'stop_reason': r.stop_reason, 'block_types': kinds}
    if model in ('gpt61', 'gpt56'):
        content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(_png(im)).decode(),
                                                       'detail': 'auto'}} for im in images]
        content.append({'type': 'text', 'text': user})
        r = _client('openai').chat.completions.create(model=mid, max_completion_tokens=MAX_OUT,
                                                      messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': content}])
        ch = r.choices[0]
        u = {'input_tokens': r.usage.prompt_tokens, 'output_tokens': r.usage.completion_tokens,
             'actual_model': r.model, 'stop_reason': ch.finish_reason}
        det = getattr(r.usage, 'completion_tokens_details', None)
        if det is not None and getattr(det, 'reasoning_tokens', None) is not None:
            u['reasoning_tokens'] = det.reasoning_tokens
        return (ch.message.content or '').strip(), u
    if model == 'qwen3vl':
        content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(_png(im)).decode()}} for im in images]
        content.append({'type': 'text', 'text': user})
        r = _client('openrouter').chat.completions.create(
            model=mid, max_tokens=MAX_OUT, messages=[{'role': 'system', 'content': system}, {'role': 'user', 'content': content}],
            extra_body={'provider': {'order': [QWEN_PROVIDER], 'allow_fallbacks': False}, 'usage': {'include': True}})
        ch = r.choices[0]; uu = r.usage
        u = {'input_tokens': getattr(uu, 'prompt_tokens', None), 'output_tokens': getattr(uu, 'completion_tokens', None),
             'actual_model': r.model, 'stop_reason': ch.finish_reason, 'provider': getattr(r, 'provider', None),
             'cost': getattr(uu, 'cost', None)}
        return (ch.message.content or '').strip(), u
    if model in ('gem38', 'gem4'):
        from google.genai import types
        parts = [types.Part.from_bytes(data=_png(im), mime_type='image/png') for im in images] + [user]
        cfg = types.GenerateContentConfig(system_instruction=system, max_output_tokens=MAX_OUT)
        r = _client('gemini').models.generate_content(model=mid, contents=parts, config=cfg)
        raw = ''.join(p.text for p in r.candidates[0].content.parts if getattr(p, 'text', None) and not getattr(p, 'thought', False)) \
            if r.candidates and r.candidates[0].content and r.candidates[0].content.parts else ''
        um = r.usage_metadata
        return raw.strip(), {'input_tokens': um.prompt_token_count, 'output_tokens': um.candidates_token_count,
                             'thinking_tokens': getattr(um, 'thoughts_token_count', None), 'actual_model': mid,
                             'stop_reason': str(r.candidates[0].finish_reason) if r.candidates else None}
    raise KeyError(model)


def truncated(model, u):
    s = str(u.get('stop_reason') or '').lower()
    return ('length' in s) or ('max_tokens' in s) or ('max_tokens' in s.replace(' ', '_'))
