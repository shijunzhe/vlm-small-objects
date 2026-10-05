import json, numpy as np
from collections import defaultdict
import mech as M
anns={}
def A(d):
    if d not in anns: anns[d]=json.loads((M.BENCH/'annotations'/f'{d}.json').read_text())
    return anns[d]
def cellstats(files, cells):
    out={}
    for c in cells:
        vals=[]
        for f in files:
            seen={}
            for l in open(f):
                r=json.loads(l)
                if r['cell']==c: seen[(r['did'],r['i'],r['j'])]=r
            if not seen: continue
            per=defaultdict(lambda:{'ne':[], 'fpe':0,'tiles':0})
            for k,r in seen.items():
                if r['status']!='success': continue
                a=A(r['did']); Aa,B,Cc,E=r['core_px']; x0,y0=r['origin_z0']; z0,up=r['z0'],r['up']
                ng=sum(Aa<=(gx*z0-x0)*up<Cc and B<=(gy*z0-y0)*up<E for gx,gy in a['points'])
                pts=M.to_native(r); p=per[r['did']]; p['tiles']+=1
                if ng: p['ne']+=pts
                else: p['fpe']+=len(pts)
            for did,p in per.items():
                a=A(did); tsym,t16=M.taus(a)
                allp=p['ne']  # FPs on empty tiles counted separately
                tp,n_p,n_g=M.match(p['ne'],a['points'],tsym)
                ntot=n_p+p['fpe']
                vals.append((did, 2*tp/(ntot+n_g), tp/n_g, tp/max(n_p,1), p['fpe']/n_g, abs(ntot-n_g)/n_g, p['tiles']))
        if vals:
            V=np.array([v[1:] for v in vals])
            out[c]=dict(zip(['f1','rec','prec_ne','fp_empty','relerr','tiles'],V.mean(0)))
            out[c]['per']= {v[0]:round(v[1],2) for v in vals}
    return out
for model,sfiles in [('gpt',['step2_gpt_run0.jsonl','step2_gpt_run1.jsonl']),('claude',['step2_claude_run0.jsonl'])]:
    s=cellstats([M.OUT/f for f in sfiles],['S2','L2n'])
    x=cellstats([M.OUT/f'x2_{model}.jsonl'],['T512','T768','T1024'])
    print(f'=== {model}  (symbol 44 px; S2 = 768 px tiles upsampled from 22 px)')
    for c,v in list(x.items())+list(s.items()):
        print(f"{c:6s} F1_sym {v['f1']:.2f}  recall {v['rec']:.2f}  prec(non-empty tiles) {v['prec_ne']:.2f}  FP on empty tiles/gt {v['fp_empty']:.2f}  |relerr| {v['relerr']:.2f}  tiles/drawing {v['tiles']:.0f}  per-drawing {v['per']}")
