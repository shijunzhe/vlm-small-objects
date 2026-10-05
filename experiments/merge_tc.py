import json, collections
O=json.load(open('results_law/theory_check.json')); T=json.load(open('results_law/theory_check_thm1.json'))
out=[]
for o in O:
    if o['family'] in ('Thm1','outside'): continue
    if '(agent)' in o['rule']: o=dict(o, family='policy:'+o['family'], verdict='policy-'+o['verdict'])
    if o['rule'].startswith('single tiles') and 'sonnet55' in o['label'] and any(k in o['rule'] for k in ('4 tiles','8 tiles')): o=dict(o, family='R+M')
    if o['rule'].startswith('tiles of upsampled') and 'gem38' in o['label']: o=dict(o, family='R+M')
    out.append(o)
for t in T:
    out.append(dict(t, rule='tiles (S_tile >= S_whole) - whole, interior targets, hit recall' if t['family']=='Thm1' else 'tiles (S_tile < S_whole) - whole (no prediction)'))
json.dump(out, open('results_law/theory_check_final.json','w'), indent=1)
tested=[o for o in out if o['verdict'] in ('supported','consistent','violated')]
print(len(tested), collections.Counter(o['verdict'] for o in tested))
