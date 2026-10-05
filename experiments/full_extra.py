"""'full' level (E2 counts-first protocol: whole drawing at 1600 px, legend crop) for REDP-X40 drawings beyond d010."""
import json, sys, os
sys.path.insert(0,'vlm_api'); os.environ.setdefault('RELEASE_ROOT','vlm_api')
import arr_common as C, e2_redpx_order as E2
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
Image.MAX_IMAGE_PIXELS=None
C.load_env('../config.env')
model=sys.argv[1]; dids=sys.argv[2].split(','); runs=int(sys.argv[3]) if len(sys.argv)>3 else 3
B='../data/redp40/redp_x40/'
out=open(f'results/{model}_full_extra.jsonl','a')
def one(job):
    did,run=job
    a=json.load(open(B+f'annotations/{did}.json')); img=Image.open(B+a['image']).convert('RGB'); W,H=img.size
    s=min(1.0,1600/max(W,H)); im=img.resize((round(W*s),round(H*s)),Image.LANCZOS) if s<1 else img
    ref=Image.open(B+a['template']).convert('RGB')
    sysp,user=E2.PROMPTS['counts_first']; sysp=sysp.format(canvas_w=im.size[0],canvas_h=im.size[1])
    raw,u=C.call(model,[ref,im],sysp,user); p=C.parse(raw); sh=p.get('shapes',[]) if isinstance(p,dict) else []
    return {'did':did,'run':run,'scale':s,'status':'parse_error' if p.get('parse_error') else 'success','declared':p.get('count'),'shapes':sh,'usage':u,'raw':raw}
with ThreadPoolExecutor(6) as ex:
    for r in ex.map(one,[(d,k) for d in dids for k in range(runs)]):
        out.write(json.dumps(r)+'\n'); out.flush(); print(r['did'],r['run'],r['status'],len(r['shapes']))
