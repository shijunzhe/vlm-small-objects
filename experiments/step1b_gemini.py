"""Gemini coordinate calibration at a cell it can perceive (FOV=S, up=1, S_tok~0.92)."""
import json, math
from collections import defaultdict
import mech as M
M.C.load_env('../config.env')
OUTP = M.OUT / 'step1b_gemini.jsonl'
jobs=[]
for did in ['d001','d004','d010','d017','d019']:
    ann,img,tpl=M.load(did); ts=M.tiles_for(ann,img,'S',1); z0=ts[0]['z0']
    def n_in(t):
        a,b,c,e=t['core_px']; x0,y0=t['origin_z0']
        return sum(a<=(gx*z0-x0)<c and b<=(gy*z0-y0)<e for gx,gy in ann['points'])
    ts.sort(key=n_in, reverse=True); tplz=M.scaled_template(tpl,z0,1)
    for t in ts[:2]:
        for coord in ('px','n1000'):
            jobs.append({'model':'gemini','coord':coord,'tpl':tplz,'tile':t,'meta':{'did':did,'model':'gemini','coord':coord,'i':t['i'],'j':t['j'],'cell':'S1'}})
M.run_jobs(jobs, OUTP, key=lambda r:(r['coord'],r['did'],r['i'],r['j']), workers=8)
anns={}
res=defaultdict(lambda:[0,0,0])
for l in open(OUTP):
    r=json.loads(l)
    if r['status']!='success': continue
    a_=anns.setdefault(r['did'],json.loads((M.BENCH/'annotations'/f"{r['did']}.json").read_text()))
    a,b,c,e=r['core_px']; x0,y0=r['origin_z0']; z0,up=r['z0'],r['up']
    gt=[(gx,gy) for gx,gy in a_['points'] if a<=(gx*z0-x0)*up<c and b<=(gy*z0-y0)*up<e]
    tsym,_=M.taus(a_)
    for dec in ('px','n1000','n1000_yx'):
        rr=dict(r,decode=dec); tp,n_p,n_g=M.match(M.to_native(rr),gt,tsym); v=res[(r['coord'],dec)]; v[0]+=tp;v[1]+=n_p;v[2]+=n_g
for k,v in sorted(res.items()): print('gemini prompt=%s decode=%s F1_sym=%.2f (pred %d / gt %d)'%(k[0],k[1],2*v[0]/max(v[1]+v[2],1),v[1],v[2]))
