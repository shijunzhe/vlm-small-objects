#!/usr/bin/env python3
"""2026-10-27 check for Table 1: Gemini 4 Argon mixes coordinate conventions under the pixel-coordinate whole-image
prompt (some answers in pixels, most in 0-1000). This reruns the whole-image call with the same prompt except that
the coordinate sentences ask for 0-1000 normalized coordinates, the convention the tiles (D, CS) already use for Gemini.
usage: python3 gem4_a1n.py <redp|fpc>   -> A1n_gem4_run0.jsonl next to A1_gem4_run0.jsonl"""
import sys, glob, json, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
bench = sys.argv[1]
if bench == 'fpc':
    sys.argv = ['fpc_vlm.py', 'sheets']; import fpc_vlm  # noqa: F401
    import main_test as T
    T.TEST = sorted(Path(p).stem for p in glob.glob('../data/fpc/sheets/annotations/*_k3.json'))
else:
    sys.argv = ['gen_test.py']; import gen_test  # noqa: F401
    import main_test as T
import mech as M
import gen_common as G
sys.path.insert(0, 'vlm_api')
import e2_redpx_order as E2
M.C.load_env('../config.env')
SYS0, USER = E2.PROMPTS['counts_first']
OLD = ("The drawing image is {canvas_w}x{canvas_h} pixels. All coordinates you output\n"
       "must be in this image's pixel space, where (0,0) is the top-left corner.")
NEW = ("Report coordinates normalized to the drawing image: x and y each range from 0 to 1000,\n"
       "where (0,0) is the top-left corner and (1000,1000) is the bottom-right corner of the image.")
assert SYS0.count(OLD) == 1 and SYS0.count('{{"type": "target_symbol", "cx": <pixel_x>, "cy": <pixel_y>}}') == 1
SYS = SYS0.replace(OLD, NEW).replace('<pixel_x>', '<x_0_to_1000>').replace('<pixel_y>', '<y_0_to_1000>').replace(
    'with its center pixel coordinate', 'with its center coordinate')
out = T.OUT / 'A1n_gem4_run0.jsonl'
done = {json.loads(l)['did'] for l in out.open() if json.loads(l)['status'] in ('success', 'parse_error')} if out.exists() else set()


def one(did):
    ann, img, tpl = M.load(did); W, H = img.size
    s = min(1.0, 1600 / max(W, H)); im = img.resize((round(W * s), round(H * s)), Image.LANCZOS) if s < 1 else img
    rec = {'did': did, 'model': 'gem4', 'run': 0, 'scale': s, 'sent_size': list(im.size), 'coord': 'n1000'}
    for attempt in range(3):
        try:
            raw, u = G.call('gem4', [tpl, im], SYS.format(), USER); p = M.C.parse(raw)
            sh = p.get('shapes', []) if isinstance(p, dict) else []
            rec.update(status='parse_error' if p.get('parse_error') else 'success', declared=p.get('count'),
                       shapes=sh if isinstance(sh, list) else [], usage=u, raw=raw); break
        except Exception as e:
            rec.update(status='api_error', error=f'{type(e).__name__}: {e}'[:300]); time.sleep(10 * 2 ** attempt)
    return rec


with ThreadPoolExecutor(4) as ex, out.open('a') as f:
    for r in ex.map(one, [d for d in T.TEST if d not in done]):
        f.write(json.dumps(r) + '\n'); f.flush(); print('A1n gem4', r['did'], r['status'], len(r.get('shapes', [])), flush=True)
