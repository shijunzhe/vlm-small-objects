"""X4b: Gemini 2.5 Flash media_resolution=HIGH only applies to single-image requests, so the legend crop
is placed in a labeled band above the drawing in ONE image. Whole drawing (<=3072 px), dev set, 2 runs.
Coordinates requested as 0-1000 normalized over the whole composite image; mapped back to native."""
import base64, io, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw
import mech as M
sys.path.insert(0, 'vlm_api')
import arr_common as C
DIDS = ('d001', 'd004', 'd008', 'd010', 'd017', 'd019')
OUT = M.OUT / 'x4b_gemini_high.jsonl'
SYS = """You are a precise engineering drawing analysis assistant.

The image has a white band at the top containing a box labeled REFERENCE with ONE target symbol
(taken from the drawing's legend). Below the band is a full engineering drawing. Count every
instance of the target symbol in the drawing below the band; do not count the reference itself.

Report coordinates normalized to the whole image: x and y each range from 0 to 1000, where (0,0) is
the top-left corner and (1000,1000) the bottom-right corner of the whole image.

Respond with ONLY a JSON object (no markdown, no backticks, no commentary). Format:
{"target_type": "target_symbol", "count": <integer>, "shapes": [{"type": "target_symbol", "cx": <x_0_to_1000>, "cy": <y_0_to_1000>}, ...]}
The "count" must equal the length of "shapes". Count only symbols matching the reference; do NOT count other symbols, text labels, or drawing linework."""
USER = "Count every instance of the reference symbol in the drawing below the band. Respond with JSON only. Be thorough; do not miss any instances, do not invent ones."

def composite(img, tpl, s):
    im = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS) if s < 1 else img
    t = tpl
    band = max(t.height + 40, 80)
    can = Image.new('RGB', (max(im.width, t.width + 40), im.height + band), 'white')
    can.paste(im, (0, band))
    d = ImageDraw.Draw(can)
    d.rectangle([10, 10, 30 + t.width, 30 + t.height], outline=(0, 0, 0), width=2)
    can.paste(t, (20, 20)); d.text((40 + t.width, 20), 'REFERENCE', fill=(0, 0, 0))
    d.line([0, band - 2, can.width, band - 2], fill=(0, 0, 0), width=2)
    return can, band

def one(job):
    did, run = job
    from google import genai
    from google.genai import types
    ann, img, tpl = M.load(did)
    s = min(1.0, 3072 / max(img.size))
    can, band = composite(img, tpl, s)
    b = io.BytesIO(); can.save(b, format='PNG')
    cl = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    r = cl.models.generate_content(model='gemini-2.5-flash', contents=[types.Part.from_bytes(data=b.getvalue(), mime_type='image/png'), USER],
        config=types.GenerateContentConfig(system_instruction=SYS, max_output_tokens=8192, thinking_config=types.ThinkingConfig(thinking_budget=0), media_resolution='MEDIA_RESOLUTION_HIGH'))
    raw = ''.join(p.text for p in r.candidates[0].content.parts if getattr(p, 'text', None)) if r.candidates and r.candidates[0].content else ''
    p = C.parse(raw); sh = p.get('shapes', []) if isinstance(p, dict) else []
    pts = []
    for q in sh if isinstance(sh, list) else []:
        x, y = M.fnum(q.get('cx')), M.fnum(q.get('cy'))
        if x is None or y is None: continue
        X, Y = x / 1000 * can.width, y / 1000 * can.height - band
        if Y >= 0: pts.append([X / s, Y / s])
    return {'mode': 'gemini_high_composite', 'did': did, 'run': run, 'status': 'parse_error' if p.get('parse_error') else 'success',
            'pts_native': pts, 'n_raw': len(sh) if isinstance(sh, list) else 0, 'input_tokens': r.usage_metadata.prompt_token_count, 'raw': raw}

if __name__ == '__main__':
    C.load_env('../config.env')
    with ThreadPoolExecutor(6) as ex, OUT.open('a') as f:
        for r in ex.map(one, [(d, k) for d in DIDS for k in range(2)]):
            f.write(json.dumps(r) + '\n'); f.flush(); print(r['did'], r['run'], r['status'], len(r['pts_native']), r['input_tokens'], flush=True)
