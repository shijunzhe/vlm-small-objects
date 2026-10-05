"""F1 of the safe tiling with the union rule (each tile keeps its answers in its overlapping core), as in Thm 1. Existing calls only."""
import json, sys, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'theory_v8')
import syn_assume as A
from output_form import load, all_points, inside, boot, items
from scipy.optimize import linear_sum_assignment
def match(P, G, tau):
    if len(P) == 0: return 0
    d = np.hypot(P[:, None, 0] - G[None, :, 0], P[:, None, 1] - G[None, :, 1]); C = np.where(d <= tau, 0, 1)
    r, c = linear_sum_assignment(C); return int((C[r, c] == 0).sum())
rng = np.random.default_rng(11); res = {}
for v in ['gpt56', 'gpt54', 'sonnet55', 'qwen3vl', 'gem38', 'gemma4', 'opus55', 'gpt61']:
    recs = load(v); dF1 = []; dTP = []; dE = []; agg = np.zeros(4)
    for pid, it in items.items():
        tau = it['m'] / 2; G = np.array(it['points'], float); n = len(G)
        tiles = [c for c in A.calls_for(it) if c['kind'][0] == 'T']
        if (pid, 'P') not in recs or not all((pid, c['kind']) in recs for c in tiles): continue
        Pw = all_points(recs[(pid, 'P')]); Pw = Pw[inside(Pw, [0, 0, A.SIDE, A.SIDE])]
        Pd = np.vstack([all_points(recs[(pid, c['kind'])])[inside(all_points(recs[(pid, c['kind'])]), c['core'])] for c in tiles])
        tw, td = match(Pw, G, tau), match(Pd, G, tau); ew, ed = len(Pw) - tw, len(Pd) - td
        dF1.append((pid, 2 * td / (n + len(Pd)) - 2 * tw / (n + len(Pw)))); dTP.append((pid, (td - tw) / n)); dE.append((pid, (ed - ew) / n))
        agg += [tw, ew, td, ed]
    tw, ew, td, ed = agg; n = 576; rho = (n + ew) / tw
    res[v] = {'dF1': boot(dF1, rng), 'dTP': boot(dTP, rng), 'dE': boot(dE, rng), 'agg': agg.tolist(), 'rho_w': rho, 'rate': (ed - ew) / (td - tw)}
    print(v, 'dF1 %+.3f [%+.3f,%+.3f]' % res[v]['dF1'], 'dTP %+.3f' % res[v]['dTP'][0], 'dE/n %+.3f' % res[v]['dE'][0], 'agg', agg, 'rho_w %.2f rate %.2f' % (rho, res[v]['rate']))
json.dump(res, open('theory_v8/f1_union.json', 'w'), indent=1)
