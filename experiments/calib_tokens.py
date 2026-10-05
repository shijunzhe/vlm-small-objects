#!/usr/bin/env python3
"""Step 0: measure how many input tokens each model spends on an image of a given size.
Blank white images (token cost does not depend on content). A 16x16 image gives the
near-zero baseline (text prompt + minimal image cost)."""
import json, sys, os
sys.path.insert(0, 'vlm_api')
os.environ.setdefault('RELEASE_ROOT', 'vlm_api')
import arr_common as C
from PIL import Image
C.load_env('../config.env')
SIZES = [(16,16),(256,256),(512,512),(768,768),(1024,1024),(1536,1536),(2048,2048),(3072,3072),(1600,1000),(4096,2048)]
out = []
for v in ['claude','gpt','gemini','qwen3vl','gemma4']:
    for (w,h) in SIZES:
        img = Image.new('RGB',(w,h),'white')
        try:
            raw, u = C.call(v, [img], 'Reply with the single word OK.', 'OK?')
            rec = {'vendor':v,'w':w,'h':h,'input_tokens':u.get('input_tokens'),'provider':u.get('provider')}
        except Exception as e:
            rec = {'vendor':v,'w':w,'h':h,'error':str(e)[:200]}
        print(rec, flush=True); out.append(rec)
json.dump(out, open('calib_tokens.json','w'), indent=1)
