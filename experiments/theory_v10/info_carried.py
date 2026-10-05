"""Lower bound on the bits about target positions each whole-image answer carried in the assumption battery (Thm capacity,
slot prior of syn.render: a tau-disk can receive the jittered centre of at most one slot (2(tau+P/6) < P), so p_k <= 1/(K)_k). Existing calls only."""
import json, math, sys, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'theory_v8')
import syn_assume as A
from output_form import load, all_points, inside
from f1_union import match
def lg(x): return math.lgamma(x + 1) / math.log(2)
def ff(n, k): return lg(n) - lg(n - k)
def K_of(W, H, m):
    p = max(int(1.6 * m), m + 6); return len(range(p // 2 + 4, W - p // 2 - 4, p)) * len(range(p // 2 + 4, H - p // 2 - 4, p))
items = {it['id']: it for it in json.load(open(A.ROOT / 'items.json'))}
res = {}
for v in ['gpt61', 'opus55', 'sonnet55', 'gpt56', 'gem38', 'gpt54', 'qwen3vl', 'gemma4']:
    recs = load(v); rows = []
    for pid, it in items.items():
        G = np.array(it['points'], float); tau = it['m'] / 2; n = len(G); K = K_of(2048, 2048, it['m'])
        P = all_points(recs[(pid, 'P')]); P = P[inside(P, [0, 0, 2048, 2048])]; tp = match(P, G, tau); R = len(P)
        I = ff(K, tp) - ff(n, tp) - (lg(R) - lg(tp) - lg(R - tp)) - math.log2(n + 1)
        rows.append({'m': it['m'], 'I': I, 'Lfull': ff(K, n) - ff(n, n) - math.log2(n + 1), 'recall': tp / n})
    res[v] = {str(m): {'I': float(np.mean([r['I'] for r in rows if r['m'] == m])), 'Lfull': float(np.mean([r['Lfull'] for r in rows if r['m'] == m])),
                       'recall': float(np.mean([r['recall'] for r in rows if r['m'] == m]))} for m in (12, 20, 32)}
json.dump(res, open('theory_v10/info_carried.json', 'w'), indent=1)
print(json.dumps({v: {m: round(d['I']) for m, d in r.items()} for v, r in res.items()}))
