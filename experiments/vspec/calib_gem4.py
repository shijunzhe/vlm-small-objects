import json, sys
from PIL import Image, ImageDraw
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import vcall as VC
out = open(__import__('os').path.dirname(__import__('os').path.abspath(__file__)) + '/results/phase0_calib_gem4.jsonl', 'a')
for sz in [None, (256, 256), (512, 512), (1024, 1024), (1536, 1536), (2048, 2048), (3072, 3072), (2576, 1449)]:
    imgs = []
    if sz:
        im = Image.new('RGB', sz, 'white'); d = ImageDraw.Draw(im)
        for k in range(0, max(sz), 64):
            d.line([k, 0, k, sz[1]], fill=(150, 150, 150)); d.line([0, k, sz[0], k], fill=(150, 150, 150))
        imgs = [im]
    rec = dict(model='gem4', size=list(sz) if sz else None)
    try:
        raw, u = VC.call('gem4', imgs, 'You are a helpful assistant.', 'Reply with the single word OK.')
        rec.update(status='success', usage=u, raw=raw[:100], cost=VC.cost('gem4', u))
    except Exception as e:
        rec.update(status='api_error', error=f'{type(e).__name__}: {e}'[:400])
    out.write(json.dumps(rec) + '\n'); out.flush()
    print(sz, rec['status'], rec.get('usage', {}).get('input_tokens'), rec.get('usage', {}).get('thinking_tokens'), rec.get('error', '')[:200], flush=True)
