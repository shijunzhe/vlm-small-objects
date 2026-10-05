#!/usr/bin/env python3
"""Numerical checks of the closed-form propositions (zero cost)."""
import json, math, ast, numpy as np, mech as M, ceilings as K
rng = np.random.default_rng(0)

def p_hit(rho):
    """Area of a disk of radius rho centred in a unit square (= P(uniform point within rho of the centre))."""
    if rho <= 0.5: return math.pi * rho**2
    if rho >= math.sqrt(2) / 2: return 1.0
    return math.pi * rho**2 - 4 * (rho**2 * math.acos(1 / (2 * rho)) - 0.5 * math.sqrt(rho**2 - 0.25))

print('=== P2 localization: closed form vs Monte Carlo (single isolated instance)')
for S in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5):
    rho = S / 2                       # tau = m/2, t = m/S
    u = rng.uniform(-0.5, 0.5, (200000, 2)); mc = np.mean(np.hypot(u[:, 0], u[:, 1]) <= rho)
    print(f'  S={S:.2f} tokens/symbol  closed form {p_hit(rho):.4f}  MC {mc:.4f}')

print('\n=== P3 merging/undercount: presence-only observer, Poisson instances at lambda per token')
for lam in (0.05, 0.2, 0.5, 1.0):
    pred = (1 - math.exp(-lam)) / lam
    cells = 2000; n = rng.poisson(lam, cells); mc = (n > 0).sum() / n.sum()
    print(f'  lambda={lam:.2f}  predicted count ratio {pred:.3f}  MC {mc:.3f}')

print('\n=== P3 on FSC-147 (E4 outputs): observed list/gt vs lambda = instances per visual token')
man = json.load(open('vlm_api/data_e4/fsc147/manifest.json')); items = {it['item_id']: it for it in man['items']}
GRID = {'gemini': 256, 'gemma4': 280}
for model in ('gpt', 'claude', 'gemini', 'gemma4', 'qwen3vl'):
    rows = []
    for l in open(f'vlm_api/results/e4/fsc147/{model}_run0.jsonl'):
        r = json.loads(l)
        if r['status'] != 'success': continue
        w, h = ast.literal_eval(r['canvas']) if isinstance(r['canvas'], str) else r['canvas']
        if model in GRID: ntok = GRID[model]
        elif model == 'claude':
            rw, rh = M.claude_resized(w, h); ntok = math.ceil(rw / 28) * math.ceil(rh / 28)
        else: ntok = math.ceil(w / 32) * math.ceil(h / 32)
        g = int(r['gt_count']); lam = g / ntok
        rows.append((lam, int(r['list_count']) / g))
    R = np.array(rows)
    out = []
    for a, b in ((0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 9)):
        s = (R[:, 0] >= a) & (R[:, 0] < b)
        if s.sum() >= 5:
            out.append(f'[{a},{b}) n={s.sum():3d} obs {np.median(R[s,1]):.2f} pred {np.median([(1-math.exp(-x))/x for x in R[s,0]]):.2f}')
    print(f'  {model:8s} ' + ' | '.join(out))
