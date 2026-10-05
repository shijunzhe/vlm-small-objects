#!/usr/bin/env python3
"""G1: billing-based token calibration of new-generation models (blank images; cost does not depend on content)."""
import json, mech as M, gen_common as G
from PIL import Image
M.C.load_env('../config.env')
G.REASONING['gemini-3.8-flash'] = {'thinking_budget': 0}
SIZES = [(16, 16), (512, 512), (1024, 1024), (1536, 1536), (2048, 2048), (2576, 1449), (3072, 3072), (1600, 1000), (4096, 2048), (6000, 4000)]
out = []
for v in ('sonnet55', 'opus55', 'gpt55', 'gpt56', 'gpt56orig', 'gem38', 'gem38uh'):
    for w, h in SIZES:
        if v != 'gpt56orig' and w * h > 4096 * 2048:
            continue
        try:
            raw, u = G.call(v, [Image.new('RGB', (w, h), 'white')], 'Reply with the single word OK.', 'OK?')
            rec = {'vendor': v, 'w': w, 'h': h, 'input_tokens': u['input_tokens']}
        except Exception as e:
            rec = {'vendor': v, 'w': w, 'h': h, 'error': str(e)[:200]}
        print(rec, flush=True); out.append(rec)
json.dump(out, open('gen_calib.json', 'w'), indent=1)
