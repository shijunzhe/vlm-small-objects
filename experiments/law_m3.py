#!/usr/bin/env python3
"""M3 configuration choice with stronger baselines and bootstrap intervals (over items, clustered by item across models)."""
import json, numpy as np
from collections import defaultdict
P = json.load(open('results_law/law_test_preds.json')); F = json.load(open('results_law/law_fit.json'))
R = json.load(open('results_law/law_data.json'))
DOM = {'redp': 'redp', 'fpc_bench': 'fpc', 'fpc_sheets': 'fpc', 'fsc': 'fsc'}
CFG = {'redp': ('A1', 'D'), 'fpc_bench': ('A1', 'D'), 'fpc_sheets': ('A1', 'D'), 'fsc': ('W', 'U', 'T')}
# per-model rule learned on tuning items (mean observed recall per config); fallback: tiles
tune = defaultdict(lambda: defaultdict(list))
for r in R:
    if r['split'] == 'tune' and r['ds'] != 'syn' and r['config'] in ('A1', 'D', 'W', 'U', 'T'):
        tune[(DOM[r['ds']], r['model'])][r['config']].append(r['obs'])
idx = defaultdict(dict)
for r in P: idx[(r['ds'], r['model'], r['item'])][r['config']] = r
rng = np.random.default_rng(7); out = {}
for ds, cfgs in CFG.items():
    rows = [(k, c) for k, c in idx.items() if k[0] == ds and all(x in c for x in cfgs)]
    def rule_tune(model):
        t = tune.get((DOM[ds], model))
        if t and all(x in t for x in cfgs): return max(cfgs, key=lambda x: np.mean(t[x]))
        return cfgs[-1]
    test_mean = defaultdict(lambda: defaultdict(list))
    for k, c in rows:
        for x in cfgs: test_mean[k[1]][x].append(c[x]['obs'])
    rule_oracle = {v: max(cfgs, key=lambda x: np.mean(d[x])) for v, d in test_mean.items()}
    choices = {'law': [max(cfgs, key=lambda x: c[x]['pred']) for k, c in rows],
               'model_rule_tune': [rule_tune(k[1]) for k, c in rows],
               'model_rule_test_oracle': [rule_oracle[k[1]] for k, c in rows],
               **{f'always_{x}': [x] * len(rows) for x in cfgs}}
    regret = {n: np.array([max(c[x]['obs'] for x in cfgs) - c[ch]['obs'] for (k, c), ch in zip(rows, chs)]) for n, chs in choices.items()}
    items = sorted({k[2] for k, c in rows}); pos = defaultdict(list)
    for j, (k, c) in enumerate(rows): pos[k[2]].append(j)
    def boot(a, b):
        d = a - b; bs = []
        for _ in range(2000):
            sel = np.concatenate([pos[items[i]] for i in rng.integers(0, len(items), len(items))]); bs.append(d[sel].mean())
        return [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out[ds] = {'n': len(rows), 'regret': {n: float(r.mean()) for n, r in regret.items()},
               'law_minus': {n: boot(regret['law'], regret[n]) for n in regret if n != 'law'},
               'rule_tune': {v: rule_tune(v) for v in sorted({k[1] for k, c in rows})}}
    print(ds, out[ds]['n'], {n: round(v, 3) for n, v in out[ds]['regret'].items()})
    print('   law - other:', {n: [round(x, 3) for x in v] for n, v in out[ds]['law_minus'].items()})
    print('   tuned rule:', out[ds]['rule_tune'])
json.dump(out, open('results_law/law_m3.json', 'w'), indent=1)
