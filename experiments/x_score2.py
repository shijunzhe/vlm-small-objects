import json, numpy as np
from collections import defaultdict
import mech as M
from x_score import cellstats
def num(v): return isinstance(v,(int,float)) and not isinstance(v,bool)
DEV=['d001','d004','d008','d010','d017','d019']
def full_metrics(model):
    """default whole-drawing condition (E2 counts-first for d001-d010; pilot full_extra for d017/d019)"""
    runs={}
    for l in open(f'vlm_api/results/e2/{model}.jsonl'):
        r=json.loads(l)
        if r['status']=='success' and r['order']=='counts_first':
            s=r['scale']; runs.setdefault(f"d{r['item_id']:03d}",[]).append([(q['cx']/s,q['cy']/s) for q in r['shapes'] if isinstance(q,dict) and num(q.get('cx')) and num(q.get('cy'))])
    try:
        for l in open(f'results/{model}_full_extra.jsonl'):
            r=json.loads(l)
            if r['status']=='success':
                s=r['scale']; dec=1
                pts=[]
                for q in r['shapes']:
                    if isinstance(q,dict) and num(q.get('cx')) and num(q.get('cy')): pts.append((q['cx']/s,q['cy']/s))
                runs.setdefault(r['did'],[]).append(pts)
    except FileNotFoundError: pass
    return runs
def summarize(runs, label):
    v=[]
    for did in DEV:
        if did not in runs: continue
        a=json.loads((M.BENCH/'annotations'/f'{did}.json').read_text()); tsym,_=M.taus(a); g=a['count']
        m=[]
        for p in runs[did]:
            tp,n_p,n_g=M.match(p,a['points'],tsym); m.append((2*tp/(n_p+n_g), tp/n_g, tp/max(n_p,1), abs(n_p-g)/g))
        v.append(np.mean(m,axis=0))
    V=np.mean(v,axis=0); print(f"{label:34s} F1 {V[0]:.2f}  recall {V[1]:.2f}  prec {V[2]:.2f}  |relerr| {V[3]:.2f}  (n={len(v)} drawings)")
# X4: vendor high-res whole drawing
x4=defaultdict(dict)
for l in open(M.OUT/'x4_vendor_highres.jsonl'):
    r=json.loads(l)
    if r['status']=='success':
        s=r['scale']; x4[r['mode']].setdefault(r['did'],[]).append([(q['cx']/s,q['cy']/s) for q in r['shapes'] if isinstance(q,dict) and num(q.get('cx')) and num(q.get('cy'))])
for l in open(M.OUT/'x4b_gemini_high.jsonl'):
    r=json.loads(l)
    if r['status']=='success': x4['gemini_high'].setdefault(r['did'],[]).append([tuple(p) for p in r['pts_native']])
print('--- whole drawing: default vs vendor high-resolution mode (dev set, 2-3 runs)')
summarize(full_metrics('gpt'),'GPT-5.4 default (1600px, auto)')
summarize(x4['gpt_original'],'GPT-5.4 detail=original')
summarize(full_metrics('gemini'),'Gemini default (258 tok)')
summarize(x4['gemini_high'],'Gemini HIGH (single composite img)')
summarize(full_metrics('claude'),'Claude default (1600 -> resized)')
summarize(full_metrics('gemma4'),'Gemma default')
print('--- tiles (dev set)')
for model,files,cells in [('gpt',['x2_gpt.jsonl','x2_gpt_run1.jsonl'],['T768','T1024']),('claude',['x2_claude.jsonl','x2_claude_run1.jsonl'],['T768','T1024'])]:
    for f in files:
        s=cellstats([M.OUT/f],cells)
        for c,v in s.items(): print(f"{model} {f:22s} {c:6s} F1 {v['f1']:.2f} recall {v['rec']:.2f} prec_ne {v['prec_ne']:.2f} FPempty {v['fp_empty']:.2f} |relerr| {v['relerr']:.2f}")
for model in ['gemini','gemma4']:
    s=cellstats([M.OUT/f'x3_{model}_run0.jsonl'],['k1.0','k1.5','k2.5'])
    for c,v in s.items(): print(f"{model} FOV sweep {c:5s} F1 {v['f1']:.2f} recall {v['rec']:.2f} prec_ne {v['prec_ne']:.2f} FPempty {v['fp_empty']:.2f} |relerr| {v['relerr']:.2f} tiles/dwg {v['tiles']:.0f} drawings {len(v['per'])}")
