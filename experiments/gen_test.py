#!/usr/bin/env python3
"""
G3: new-generation models on the 34 test drawings, same protocols as generation 1 (main_test.py).
usage: python3 gen_test.py <cond> <vendor> [runs]
  cond A1    : whole drawing, long side <= 1600, pixel coordinates requested (vendors in gen_common.VENDORS)
  cond A1nat : gpt56orig only: native drawing (downscaled only above 30k patches), detail=original
  cond D     : sonnet55 / gpt56: 1024-px tiles, symbol 44 px, pixel coords; gem38: fixed-grid FOV with
               symbol = 2.5 tokens on a ~33x33 grid, 0-1000 coords
  cond H     : propose-and-verify (same candidates and prompt as hybrid_verify.py)
Outputs: results_gen/<cond>_<vendor>_run<r>.jsonl and results_gen/hyb_<vendor>.jsonl
"""
import json, math, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
import mech as M
import gen_common as G
import main_test as T

OUT = Path('results_gen'); OUT.mkdir(exist_ok=True)
T.OUT = OUT
_orig_call = M.C.call
def dispatch(vendor, images, system, user):
    return G.call(vendor, images, system, user) if vendor in G.VENDORS else _orig_call(vendor, images, system, user)
M.C.call = dispatch
G.REASONING['gemini-3.8-flash'] = {'thinking_budget': 0}
T.TILE_IFACE.update({'sonnet55': ('px', 'px'), 'gpt56': ('px', 'px'), 'gem38': ('n1000', 'n1000'), 'gpt61': ('px', 'px'), 'opus55': ('px', 'px')})
T.FOV_K.update({'gem38': 2.5 * 16 / math.sqrt(1090)})   # fov_tiles uses F = 16*m/k  ->  F = sqrt(1090)*m/2.5
T.WORKERS.update({'sonnet55': 6, 'gpt56': 6, 'gem38': 10, 'gpt61': 8, 'opus55': 6}); T.XTILE.update({'sonnet55', 'gpt56', 'gpt61', 'opus55'})


def whole_native(vendor, run):
    sys.path.insert(0, 'vlm_api')
    import e2_redpx_order as E2
    out = OUT / f'A1nat_{vendor}_run{run}.jsonl'
    done = {json.loads(l)['did'] for l in out.open() if json.loads(l)['status'] in ('success', 'parse_error')} if out.exists() else set()

    def one(did):
        ann, img, tpl = M.load(did); W, H = img.size
        s = 1.0
        while math.ceil(W * s / 32) * math.ceil(H * s / 32) > 30000:
            s *= 0.99
        im = img.resize((round(W * s), round(H * s)), Image.LANCZOS) if s < 1 else img
        sysp, user = E2.PROMPTS['counts_first']; sysp = sysp.format(canvas_w=im.size[0], canvas_h=im.size[1])
        rec = {'did': did, 'model': vendor, 'run': run, 'scale': s, 'sent_size': list(im.size)}
        for attempt in range(3):
            try:
                raw, u = G.call(vendor, [tpl, im], sysp, user); p = M.C.parse(raw)
                sh = p.get('shapes', []) if isinstance(p, dict) else []
                rec.update(status='parse_error' if p.get('parse_error') else 'success', declared=p.get('count'),
                           shapes=sh if isinstance(sh, list) else [], usage=u, raw=raw); break
            except Exception as e:
                rec.update(status='api_error', error=f'{type(e).__name__}: {e}'[:300]); time.sleep(10 * 2 ** attempt)
        return rec
    with ThreadPoolExecutor(2) as ex, out.open('a') as f:
        for r in ex.map(one, [d for d in T.TEST if d not in done]):
            f.write(json.dumps(r) + '\n'); f.flush(); print('A1nat', vendor, run, r['did'], r['status'], len(r.get('shapes', [])), flush=True)


if __name__ == '__main__':
    M.C.load_env('../config.env')
    cond, vendor = sys.argv[1], sys.argv[2]
    runs = [int(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else [0]
    for r in runs:
        if cond == 'A1':
            T.whole(vendor, r)
        elif cond == 'A1nat':
            whole_native(vendor, r)
        elif cond == 'D':
            T.tiles('D', vendor, r)
        elif cond == 'H':
            import hybrid_verify as HV
            HV.M.OUT = OUT
            HV.run([vendor], T.TEST)
