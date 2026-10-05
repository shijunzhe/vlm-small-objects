#!/usr/bin/env python3
"""
X4: vendor high-resolution modes on the whole drawing (dev set, 2 runs).
  gpt_original : GPT-5.4, detail="original" (up to ~10k 32-px patches instead of 2.5k).
                 Drawing sent at native size if <= 6000 px and <= 10k patches, else downscaled to fit.
  gemini_high  : Gemini 2.5 Flash, media_resolution=HIGH (pan-and-scan tiling, ~258 tokens per tile).
Same whole-drawing prompt as E2 counts-first (pixel coordinates in the sent image's space;
Gemini additionally decoded as 0-1000 and the better decoding is reported per drawing set).
"""
import base64, io, json, math, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import mech as M
sys.path.insert(0, 'vlm_api')
import arr_common as C, e2_redpx_order as E2

DIDS = ('d001', 'd004', 'd008', 'd010', 'd017', 'd019')
OUT = M.OUT / 'x4_vendor_highres.jsonl'


def b64(im):
    b = io.BytesIO(); im.save(b, format='PNG'); return base64.b64encode(b.getvalue()).decode()


def fit_gpt_original(img):
    W, H = img.size
    s = min(1.0, 6000 / max(W, H), math.sqrt(10000 * 1024 / (W * H)) * 0.98)
    return img.resize((round(W * s), round(H * s)), Image.LANCZOS) if s < 1 else img, s


def call(mode, did, run):
    ann, img, tpl = M.load(did)
    if mode == 'gpt_original':
        im, s = fit_gpt_original(img)
    else:
        s = min(1.0, 3072 / max(img.size))
        im = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS) if s < 1 else img
    sysp, user = E2.PROMPTS['counts_first']
    sysp = sysp.format(canvas_w=im.size[0], canvas_h=im.size[1])
    if mode == 'gpt_original':
        import openai
        r = openai.OpenAI(timeout=600).chat.completions.create(
            model='gpt-5.4', max_completion_tokens=8192, reasoning_effort='none',
            messages=[{'role': 'system', 'content': sysp},
                      {'role': 'user', 'content': [
                          {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64(tpl), 'detail': 'original'}},
                          {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64(im), 'detail': 'original'}},
                          {'type': 'text', 'text': user}]}])
        raw = r.choices[0].message.content or ''
        usage = {'input_tokens': r.usage.prompt_tokens, 'output_tokens': r.usage.completion_tokens,
                 'stop_reason': r.choices[0].finish_reason}
    else:
        from google import genai
        from google.genai import types
        cl = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
        parts = [types.Part.from_bytes(data=base64.b64decode(b64(tpl)), mime_type='image/png'),
                 types.Part.from_bytes(data=base64.b64decode(b64(im)), mime_type='image/png'), user]
        r = cl.models.generate_content(model='gemini-2.5-flash', contents=parts, config=types.GenerateContentConfig(
            system_instruction=sysp, max_output_tokens=8192, thinking_config=types.ThinkingConfig(thinking_budget=0),
            media_resolution='MEDIA_RESOLUTION_HIGH'))
        raw = ''.join(p.text for p in r.candidates[0].content.parts if getattr(p, 'text', None)) if r.candidates and r.candidates[0].content else ''
        usage = {'input_tokens': r.usage_metadata.prompt_token_count, 'output_tokens': r.usage_metadata.candidates_token_count,
                 'stop_reason': str(r.candidates[0].finish_reason) if r.candidates else None}
    p = C.parse(raw)
    sh = p.get('shapes', []) if isinstance(p, dict) else []
    return {'mode': mode, 'did': did, 'run': run, 'scale': s, 'sent_size': list(im.size),
            'status': 'parse_error' if p.get('parse_error') else 'success', 'declared': p.get('count'),
            'shapes': sh if isinstance(sh, list) else [], 'usage': usage, 'raw': raw}


if __name__ == '__main__':
    C.load_env('../config.env')
    jobs = [(m, d, k) for m in ('gpt_original', 'gemini_high') for d in DIDS for k in range(2)]
    with ThreadPoolExecutor(6) as ex, OUT.open('a') as f:
        for r in ex.map(lambda j: call(*j), jobs):
            f.write(json.dumps(r) + '\n'); f.flush()
            print(r['mode'], r['did'], r['run'], r['status'], len(r['shapes']), r['usage'].get('input_tokens'), r['usage'].get('stop_reason'), flush=True)
