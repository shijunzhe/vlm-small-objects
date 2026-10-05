import json, math, numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
import ceilings as K, analyze_test as A, x_new_score as X
rows=json.load(open('syn/analysis_robust.json'))['rows']
C1=json.load(open('ceilings.json')); C2=json.load(open('gen_ceilings.json'))
G1={'gpt54':('gpt','px'),'sonnet46':('claude','px_claude'),'gemini25':('gemini','n1000'),'gemma4':('gemma4','n1000'),'qwen3vl':('qwen3vl','n1000')}
def fit(v):
    rs=[r for r in rows if r['v']==v and r['exp'] in ('res','cap','area')]
    x1=np.log([r['S'] for r in rs]); x2=np.log([r['n'] for r in rs]); y=np.array([r['recall'] for r in rs])
    def nll(p):
        z=p[0]+p[1]*x1-p[2]*x2; q=1/(1+np.exp(-z)); q=np.clip(q,1e-4,1-1e-4)
        return -np.sum(y*np.log(q)+(1-y)*np.log(1-q))
    return minimize(nll,[0,1,0.3],method='Nelder-Mead').x
def pred(p,S,n): return 1/(1+np.exp(-(p[0]+p[1]*np.log(S)-p[2]*np.log(n))))
allo=[];allp=[];allc=[]
for name in list(G1)+['gpt56','sonnet55','gem38']:
    p=fit(name)
    if name in G1:
        k,dec=G1[name]; per=A.mean_per(A.score_whole(k,dec)); C=lambda d: C1[f'{d}|A1|{k}']
    else:
        per=A.mean_per(X.at(A.score_whole,name,'n1000' if name=='gem38' else 'px')); C=lambda d: C2[f'{d}|{name}']
    ds=[d for d in K.TEST if d in per]
    S=np.array([max(C(d)['sym_tokens'],0.05) for d in ds]); n=np.array([A.ANN[d]['count'] for d in ds])
    obs=np.array([per[d][1] for d in ds]); pr=pred(p,S,n); ce=np.array([C(d)['ceiling_f1'] for d in ds])
    r=spearmanr(pr,obs)[0]; rc=spearmanr(ce,obs)[0]
    print(f'{name:9s} params a={p[0]:+.2f} b={p[1]:.2f} c={p[2]:.2f} | rho(pred,obs recall) {r:+.2f}  rho(ceil) {rc:+.2f} | mean obs {obs.mean():.2f} pred {pr.mean():.2f} MAE {np.abs(pr-obs).mean():.2f}')
    allo+=list(obs); allp+=list(pr)
allo=np.array(allo); allp=np.array(allp)
print('pooled over models: r', np.corrcoef(allp,allo)[0,1], 'R2', 1-((allo-allp)**2).sum()/((allo-allo.mean())**2).sum())
