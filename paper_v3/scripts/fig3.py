"""Draft figure: where a safe split pays. Descriptive only (no fit).
Pairs: whole call vs split whose views never zoom out (S_view >= S_whole).
Drawings/floor plans: first-round model-specific tiles (D) vs whole (A1); FSC-147: 2x2 tiles (T) vs upsampled whole (U).
Unified reader (FORMAT_AWARE + POINT_XY), regenerated 2026-10-27."""
import json, collections, numpy as np, matplotlib
matplotlib.use('Agg'); matplotlib.rcParams['pdf.fonttype']=42
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import sys
from pathlib import Path
PAPER=Path(__file__).resolve().parent.parent
R=json.load(open(sys.argv[1]))   # results_law/law_data.json of the unified-reader copy
SB=[0,0.5,1,1.5,3,np.inf]; NB=[0,4,16,32,96,np.inf]
SL=['<0.5','[0.5, 1)','[1, 1.5)','[1.5, 3)','≥3']; NL=['≤3','4 to 15','16 to 31','32 to 95','≥96']
def b(x,B): return next(i for i in range(len(B)-1) if B[i]<=x<B[i+1])
pair=collections.defaultdict(dict)
for r in R:
    if r['ds'] in ('redp','fpc_bench','fpc_sheets') and r['config'] in ('A1','D'): pair[(r['ds'],r['item'],r['model'])][r['config']]=r
    if r['ds']=='fsc' and r['config'] in ('U','T'): pair[(r['ds'],r['item'],r['model'])]['A1' if r['config']=='U' else 'D']=r
def clus(k):
    """bootstrap cluster: the image; for mosaics the source drawing (its three nested areas share blocks)"""
    return (k[0], k[1].rsplit('_k',1)[0] if k[0]=='fpc_sheets' else k[1])
def cboot(v,rng,B=4000):
    """95% interval of the mean over pairs, resampling clusters (images) with replacement"""
    cl=collections.defaultdict(list)
    for c,d in v: cl[c].append(d)
    sums=np.array([sum(x) for x in cl.values()]); cnt=np.array([len(x) for x in cl.values()])
    bs=[]
    for _ in range(B):
        idx=rng.integers(0,len(sums),len(sums)); bs.append(sums[idx].sum()/cnt[idx].sum())
    return np.percentile(bs,[2.5,97.5])
def grid(keep):
    cell=collections.defaultdict(list)
    for k,v in pair.items():
        if 'A1' in v and 'D' in v:
            S,n,_=v['A1']['calls'][0]; safe=min(c[0] for c in v['D']['calls'])>=S*0.999
            if safe==keep: cell[(b(S,SB),b(n,NB))].append((clus(k),v['D']['obs']-v['A1']['obs']))
    M=np.full((5,5),np.nan); N=np.zeros((5,5),int); SIG=np.zeros((5,5),int); CI={}
    rng=np.random.default_rng(1)
    for (i,j),v in cell.items():
        N[j,i]=len(v)
        if len(v)>=10:
            M[j,i]=np.mean([d for _,d in v]); lo,hi=cboot(v,rng); CI[(j,i)]=(float(lo),float(hi)); SIG[j,i]=1 if lo>0 else (-1 if hi<0 else 0)
    return M,N,SIG,CI
cmap=LinearSegmentedColormap.from_list('div',['#e34948','#f0efec','#2a78d6'])
norm=TwoSlopeNorm(vmin=-0.25,vcenter=0,vmax=0.5)
plt.rcParams.update({'font.size':8,'font.family':'serif'})
fig,axs=plt.subplots(1,2,figsize=(6.9,2.45),sharey=True)
out={}
for ax,keep,title in zip(axs,(True,False),('(a) Splits that never zoom out','(b) Splits that zoom out')):
    M,N,SIG,CI=grid(keep); out[title]={'mean_gain':[[None if np.isnan(x) else round(float(x),3) for x in row] for row in M],'n':N.tolist(),'sig':SIG.tolist(),'ci':{f'{k[0]},{k[1]}':[round(a,3),round(b,3)] for k,(a,b) in CI.items()}}
    im=ax.imshow(M,origin='lower',cmap=cmap,norm=norm,aspect='auto',interpolation='nearest')
    for j in range(5):
        for i in range(5):
            if not np.isnan(M[j,i]): ax.text(i,j,f'{M[j,i]:+.2f}'+('*' if SIG[j,i]!=0 else '')+f'\n({N[j,i]})',ha='center',va='center',fontsize=6.5,color='#1a1a19',fontweight='bold' if SIG[j,i]!=0 else 'normal')
    ax.set_xticks(range(5)); ax.set_xticklabels(SL); ax.set_xlabel('S of the whole-image call')
    ax.axvline(2.5,color='#55554f',lw=1,ls='--'); ax.set_title(title,fontsize=8.5)
axs[0].set_yticks(range(5)); axs[0].set_yticklabels(NL); axs[0].set_ylabel('targets in the\nwhole-image call')
cb=fig.colorbar(im,ax=axs,fraction=0.03,pad=0.02); cb.set_label('recall change, split minus whole-image call',fontsize=7)
fig.subplots_adjust(left=0.12,right=0.86,bottom=0.2,top=0.9,wspace=0.06)
fig.savefig(PAPER/'figures/fig3.png',dpi=220,bbox_inches='tight',pad_inches=0.03); fig.savefig(PAPER/'figures/fig3.pdf',dpi=600,bbox_inches='tight',pad_inches=0.03)
json.dump(out,open(PAPER/'generated/fig3_cells.json','w'),indent=1)
