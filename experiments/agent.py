#!/usr/bin/env python3
"""
X4: agentic zoom. The model sees the legend crop and an overview of the drawing (long side <= 1600 px) and may call
zoom(x0, y0, x1, y1) (overview pixels) up to MAX_ZOOMS times; each call returns that region cropped from the NATIVE
drawing and resampled so its long side is min(1024, 2 x native size). Reasoning is enabled. The final answer lists
centres in overview pixels. We log every zoom region and all billed tokens.
usage: python3 agent.py run <vendor> [run] [dids] | score
vendors: gpt56 (reasoning medium), sonnet55 (adaptive thinking, high effort), gem38 (dynamic thinking)
"""
import base64, io, json, math, os, sys, threading, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import mech as M, ceilings as K

OUT = Path('results_x/agent'); OUT.mkdir(parents=True, exist_ok=True)
MAX_ZOOMS = 20
ZOOM_SRC = os.environ.get('ZOOM_SRC', 'native')   # 'overview': ablation, the zoom is cut from the overview itself (no new pixels)
SYS = """You are counting one target symbol on an engineering drawing.
You are given (1) a reference crop of the target symbol taken from the drawing's legend and (2) an overview of the
whole drawing, which is {w}x{h} pixels. Symbols may be too small to recognize in the overview.
You can call the tool zoom(x0, y0, x1, y1) with a rectangle in OVERVIEW pixel coordinates; it returns that region cut
from the original high-resolution drawing. You may call zoom at most {z} times in total, so plan your coverage.
Count every instance of the target symbol in the whole drawing. Instances may be rotated or slightly rescaled.
When you are done, reply with ONLY a JSON object, coordinates in OVERVIEW pixels:
{{"count": <integer>, "shapes": [{{"cx": <x>, "cy": <y>}}, ...]}}"""
USER = "Here are the reference symbol and the overview. Find and locate every instance of the reference symbol."
TOOL_DESC = "Return the given rectangle of the overview, cut from the original high-resolution drawing."
FINAL = "Your zoom budget is used up or you have finished. Reply now with ONLY the final JSON object."


def b64(im):
    b = io.BytesIO(); im.save(b, format='PNG'); return base64.b64encode(b.getvalue()).decode()


def setup(did):
    ann, img, tpl = M.load(did)
    W, H = img.size; s = min(1.0, 1600 / max(W, H))
    ov = img.resize((round(W * s), round(H * s)), Image.LANCZOS) if s < 1 else img
    return ann, img, tpl, ov, s


def do_zoom(img, s, ov, args):
    x0, y0, x1, y1 = [float(args.get(k, 0)) for k in ('x0', 'y0', 'x1', 'y1')]
    x0, x1 = sorted((max(0, x0), min(ov.width, x1))); y0, y1 = sorted((max(0, y0), min(ov.height, y1)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None, 'Empty rectangle; give x0 < x1 and y0 < y1 inside the overview.', None
    box = (round(x0 / s), round(y0 / s), round(x1 / s), round(y1 / s))
    crop = img.crop(box); L = max(crop.size); out_L = min(1024, 2 * L); f = out_L / L
    size = (max(1, round(crop.width * f)), max(1, round(crop.height * f)))
    if ZOOM_SRC == 'overview':   # same output size and framing, but only the information already in the overview
        crop = ov.crop((round(x0), round(y0), max(round(x0) + 1, round(x1)), max(round(y0) + 1, round(y1)))).resize(size, Image.LANCZOS)
    elif ZOOM_SRC == 'ovcrop':   # crop of the overview at overview scale: less content per view, no new pixels, no more tokens per symbol
        crop = ov.crop((round(x0), round(y0), max(round(x0) + 1, round(x1)), max(round(y0) + 1, round(y1))))
        cw, chh = crop.size
        txt = (f'Zoom of overview region [{x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}], shown at {cw}x{chh} px in the top-left of a white canvas. '
               f'Crop pixel (u, v) corresponds to overview pixel ({x0:.0f} + u*{(x1 - x0) / cw:.4f}, {y0:.0f} + v*{(y1 - y0) / chh:.4f}).')
        if max(cw, chh) < 768:   # small images hurt independently of content (tuning-set check); pad, do not rescale
            can = Image.new('RGB', (max(768, cw), max(768, chh)), 'white'); can.paste(crop, (0, 0)); crop = can
        return crop, txt, {'overview_box': [x0, y0, x1, y1], 'native_box': list(box), 'shown': [cw, chh], 'scale': s}
    else:
        crop = crop.resize(size, Image.LANCZOS)
    txt = (f'Zoom of overview region [{x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}], shown at {crop.width}x{crop.height} px. '
           f'Crop pixel (u, v) corresponds to overview pixel ({x0:.0f} + u*{(x1 - x0) / crop.width:.4f}, {y0:.0f} + v*{(y1 - y0) / crop.height:.4f}).')
    return crop, txt, {'overview_box': [x0, y0, x1, y1], 'native_box': list(box), 'shown': list(crop.size), 'scale': f}


def run_openai(model, sysp, tpl, ov, img, s, log):
    import openai
    cl = openai.OpenAI(timeout=900)
    tools = [{'type': 'function', 'name': 'zoom', 'description': TOOL_DESC, 'parameters': {
        'type': 'object', 'properties': {k: {'type': 'number'} for k in ('x0', 'y0', 'x1', 'y1')},
        'required': ['x0', 'y0', 'x1', 'y1'], 'additionalProperties': False}, 'strict': True}]
    im = lambda x: {'type': 'input_image', 'image_url': 'data:image/png;base64,' + b64(x), 'detail': 'auto'}
    items = [{'role': 'user', 'content': [im(tpl), im(ov), {'type': 'input_text', 'text': USER}]}]
    nudged = False
    while True:
        last = log['zooms_used'] >= MAX_ZOOMS
        if last and not nudged:
            items.append({'role': 'user', 'content': [{'type': 'input_text', 'text': FINAL}]}); nudged = True
        r = cl.responses.create(model=model, instructions=sysp, input=items, tools=tools, tool_choice='none' if last else 'auto',
                                reasoning={'effort': 'medium'}, max_output_tokens=32000)
        log['in'] += r.usage.input_tokens; log['out'] += r.usage.output_tokens; log['turns'] += 1
        calls = [o for o in r.output if o.type == 'function_call']
        if calls and not last:
            items += [o.model_dump(exclude_none=True) for o in r.output]
            follow = []
            for fc in calls:
                args = json.loads(fc.arguments or '{}')
                crop, txt, meta = do_zoom(img, s, ov, args) if log['zooms_used'] < MAX_ZOOMS else (None, 'Zoom budget exhausted; give the final answer.', None)
                if meta: log['zooms'].append(meta); log['zooms_used'] += 1
                items.append({'type': 'function_call_output', 'call_id': fc.call_id, 'output': txt + (' The image follows.' if crop else '')})
                if crop: follow += [{'type': 'input_text', 'text': txt}, im(crop)]
            if follow:
                items.append({'role': 'user', 'content': follow})
            continue
        text = r.output_text or ''
        if not text.strip() and not nudged:
            items += [o.model_dump(exclude_none=True) for o in r.output]
            items.append({'role': 'user', 'content': [{'type': 'input_text', 'text': FINAL}]}); nudged = True; log['zooms_used'] = MAX_ZOOMS
            continue
        return text


def run_anthropic(model, sysp, tpl, ov, img, s, log):
    import anthropic
    cl = anthropic.Anthropic(timeout=900)
    tools = [{'name': 'zoom', 'description': TOOL_DESC, 'input_schema': {
        'type': 'object', 'properties': {k: {'type': 'number'} for k in ('x0', 'y0', 'x1', 'y1')}, 'required': ['x0', 'y0', 'x1', 'y1']}}]
    im = lambda x: {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': b64(x)}}
    msgs = [{'role': 'user', 'content': [im(tpl), im(ov), {'type': 'text', 'text': USER}]}]
    while True:
        last = log['zooms_used'] >= MAX_ZOOMS
        kw = dict(model=model, max_tokens=32000, system=sysp, messages=msgs, tools=tools,
                  thinking={'type': 'adaptive'}, extra_body={'output_config': {'effort': 'high'}})
        if last:
            kw['tool_choice'] = {'type': 'none'}
            if msgs[-1]['role'] == 'user' and isinstance(msgs[-1]['content'], list) and not any(c.get('type') == 'text' and c.get('text') == FINAL for c in msgs[-1]['content'] if isinstance(c, dict)):
                msgs[-1]['content'].append({'type': 'text', 'text': FINAL})
        r = cl.messages.create(**kw)
        log['in'] += r.usage.input_tokens; log['out'] += r.usage.output_tokens; log['turns'] += 1
        uses = [b for b in r.content if b.type == 'tool_use']
        if uses and not last:
            msgs.append({'role': 'assistant', 'content': [b.model_dump(exclude_none=True) for b in r.content]})
            res = []
            for u in uses:
                crop, txt, meta = do_zoom(img, s, ov, u.input) if log['zooms_used'] < MAX_ZOOMS else (None, 'Zoom budget exhausted; give the final answer.', None)
                if meta: log['zooms'].append(meta); log['zooms_used'] += 1
                res.append({'type': 'tool_result', 'tool_use_id': u.id, 'content': ([im(crop)] if crop else []) + [{'type': 'text', 'text': txt}]})
            msgs.append({'role': 'user', 'content': res})
            continue
        return ''.join(b.text for b in r.content if b.type == 'text')


def run_gemini(model, sysp, tpl, ov, img, s, log):
    from google import genai
    from google.genai import types
    cl = genai.Client(api_key=os.environ['GEMINI_API_KEY'], http_options=types.HttpOptions(timeout=600000))
    fd = types.FunctionDeclaration(name='zoom', description=TOOL_DESC, parameters=types.Schema(
        type='OBJECT', properties={k: types.Schema(type='NUMBER') for k in ('x0', 'y0', 'x1', 'y1')}, required=['x0', 'y0', 'x1', 'y1']))
    part = lambda x: types.Part.from_bytes(data=base64.b64decode(b64(x)), mime_type='image/png')
    contents = [types.Content(role='user', parts=[part(tpl), part(ov), types.Part.from_text(text=USER)])]
    while True:
        last = log['zooms_used'] >= MAX_ZOOMS
        cfg = types.GenerateContentConfig(system_instruction=sysp, max_output_tokens=65536,
                                          thinking_config=types.ThinkingConfig(thinking_budget=-1),
                                          tools=[types.Tool(function_declarations=[fd])],
                                          tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode='NONE' if last else 'AUTO')),
                                          automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        if last and not getattr(log, 'nudged', False) and not log.get('nudged'):
            contents.append(types.Content(role='user', parts=[types.Part.from_text(text=FINAL)])); log['nudged'] = True
        r = cl.models.generate_content(model=model, contents=contents, config=cfg)
        um = r.usage_metadata
        log['in'] += um.prompt_token_count or 0; log['out'] += (um.candidates_token_count or 0) + (getattr(um, 'thoughts_token_count', 0) or 0); log['turns'] += 1
        cont = r.candidates[0].content if r.candidates else None
        calls = [p.function_call for p in (cont.parts if cont and cont.parts else []) if getattr(p, 'function_call', None)]
        if calls and not last:
            contents.append(cont)
            parts = []
            for fc in calls:
                crop, txt, meta = do_zoom(img, s, ov, dict(fc.args or {})) if log['zooms_used'] < MAX_ZOOMS else (None, 'Zoom budget exhausted; give the final answer.', None)
                if meta: log['zooms'].append(meta); log['zooms_used'] += 1
                parts.append(types.Part.from_function_response(name='zoom', response={'result': txt}))
                if crop: parts += [types.Part.from_text(text=txt), part(crop)]
            contents.append(types.Content(role='user', parts=parts))
            continue
        text = ''.join(p.text for p in (cont.parts if cont and cont.parts else []) if getattr(p, 'text', None) and not getattr(p, 'thought', False))
        if not text.strip() and not log.get('retried'):
            log['retried'] = True; log['zooms_used'] = max(log['zooms_used'], MAX_ZOOMS)
            if cont: contents.append(cont)
            continue
        return text


SYS_N1000 = SYS.replace("the tool zoom(x0, y0, x1, y1) with a rectangle in OVERVIEW pixel coordinates", "the tool zoom(x0, y0, x1, y1) with a rectangle in coordinates normalized to 0-1000 over the overview (0,0 top-left, 1000,1000 bottom-right)").replace(
    "reply with ONLY a JSON object, coordinates in OVERVIEW pixels:", "reply with ONLY a JSON object, coordinates normalized to 0-1000 over the overview:")


def run_openrouter(model, sysp, tpl, ov, img, s, log):
    """Open-weight zoom agent (Qwen3-VL via OpenRouter, pinned provider). Qwen's native convention is 0-1000, so the
    tool and the answer use normalized coordinates; zoom arguments are mapped to overview pixels here."""
    import openai
    cl = openai.OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=900)
    tools = [{'type': 'function', 'function': {'name': 'zoom', 'description': TOOL_DESC.replace('of the overview', 'of the overview (0-1000 normalized)'),
              'parameters': {'type': 'object', 'properties': {k: {'type': 'number'} for k in ('x0', 'y0', 'x1', 'y1')}, 'required': ['x0', 'y0', 'x1', 'y1']}}}]
    im = lambda x: {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64(x)}}
    msgs = [{'role': 'system', 'content': SYS_N1000.format(w=ov.width, h=ov.height, z=MAX_ZOOMS)},
            {'role': 'user', 'content': [im(tpl), im(ov), {'type': 'text', 'text': USER}]}]
    extra = {'provider': {'order': ['Alibaba'], 'allow_fallbacks': False}}
    while True:
        last = log['zooms_used'] >= MAX_ZOOMS
        if last and msgs[-1]['role'] != 'user':
            msgs.append({'role': 'user', 'content': FINAL})
        kw = dict(model=model, messages=msgs, max_tokens=16000, extra_body=extra)
        if not last: kw['tools'] = tools
        r = cl.chat.completions.create(**kw)
        u = r.usage; log['in'] += u.prompt_tokens or 0; log['out'] += u.completion_tokens or 0; log['turns'] += 1
        mm = r.choices[0].message
        calls = mm.tool_calls or []
        if calls and not last:
            msgs.append({'role': 'assistant', 'content': mm.content or '', 'tool_calls': [c.model_dump() for c in calls]})
            follow = []
            for c in calls:
                try: a = json.loads(c.function.arguments or '{}')
                except Exception: a = {}
                a = {k: float(a.get(k, 0)) * (ov.width if k[0] == 'x' else ov.height) / 1000 for k in ('x0', 'y0', 'x1', 'y1')}
                crop, txt, meta = do_zoom(img, s, ov, a) if log['zooms_used'] < MAX_ZOOMS else (None, 'Zoom budget exhausted; give the final answer.', None)
                if meta: log['zooms'].append(meta); log['zooms_used'] += 1
                txt = txt.split(' Crop pixel')[0] if crop else txt
                msgs.append({'role': 'tool', 'tool_call_id': c.id, 'content': txt + (' The image follows.' if crop else '')})
                if crop: follow += [{'type': 'text', 'text': txt}, im(crop)]
            if follow: msgs.append({'role': 'user', 'content': follow})
            continue
        text = mm.content or ''
        if not text.strip() and not log.get('retried'):
            log['retried'] = True; log['zooms_used'] = MAX_ZOOMS; msgs.append({'role': 'assistant', 'content': ''}); continue
        return text


RUNNERS = {'qwen3vl': (run_openrouter, 'qwen/qwen3-vl-235b-a22b-instruct'), 'qwen3vlt': (run_openrouter, 'qwen/qwen3-vl-235b-a22b-thinking'), 'gpt56': (run_openai, 'gpt-5.6-sol'), 'sonnet55': (run_anthropic, 'claude-sonnet-5-5'), 'gem38': (run_gemini, 'gemini-3.8-flash')}


def run(vendor, run_id=0, dids=None):
    fn, model = RUNNERS[vendor]
    out = OUT / f'{vendor}{ {"overview": "_ovzoom", "ovcrop": "_ovcrop"}.get(ZOOM_SRC, "")}_run{run_id}.jsonl'
    done = {json.loads(l)['did'] for l in out.open() if json.loads(l)['status'] in ('success', 'parse_error')} if out.exists() else set()
    lock = threading.Lock()

    def one(did):
        ann, img, tpl, ov, s = setup(did)
        log = {'in': 0, 'out': 0, 'turns': 0, 'zooms_used': 0, 'zooms': []}
        rec = {'did': did, 'vendor': vendor, 'run': run_id, 'scale': s, 'overview': list(ov.size), 'zoom_src': ZOOM_SRC}
        for a in range(2):
            try:
                raw = fn(model, SYS.format(w=ov.width, h=ov.height, z=MAX_ZOOMS), tpl, ov, img, s, log)
                p = M.C.parse(raw); sh = p.get('shapes', []) if isinstance(p, dict) else []
                rec.update(status='parse_error' if p.get('parse_error') else 'success', shapes=sh if isinstance(sh, list) else [], raw=raw[:4000]); break
            except Exception as e:
                rec.update(status='api_error', error=f'{type(e).__name__}: {e}'[:300]); time.sleep(20)
                log = {'in': 0, 'out': 0, 'turns': 0, 'zooms_used': 0, 'zooms': []}
        rec.update(log=log)
        with lock, out.open('a') as f:
            f.write(json.dumps(rec) + '\n')
        print(vendor, did, rec['status'], 'zooms', log['zooms_used'], 'tokens', log['in'], log['out'], rec.get('error', '')[:120], flush=True)
        return rec
    todo = [d for d in (dids or K.TEST) if d not in done]
    with ThreadPoolExecutor(int(os.environ.get("AG_WORKERS", "3"))) as ex:
        list(ex.map(one, todo))


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        M.C.load_env('../config.env')
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 0, sys.argv[4].split(',') if len(sys.argv) > 4 else None)
