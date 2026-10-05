#!/usr/bin/env python3
"""G2: a priori ceilings for the new-generation models from their calibrated geometry (gen_calib.json).
A1 = whole drawing at long side <= 1600 (same protocol as generation 1); gpt56orig = native drawing,
downscaled only if it exceeds 30k patches. Token side in native px:
  sonnet55/opus55 : 28/s (high-res tier: 1600x1600 = 3364 tokens <= 4784, no resize)
  gpt55/gpt56     : 32/s (1600x1600 = 2500 patches, below the measured auto budget of >= 8192 patches)
  gpt56orig       : 32/s', s' = min(1, sqrt(30000*1024/(W*H)))
  gem38 / gem38uh : long side / sqrt(N) with N = 1090 / 2200 measured tokens (fixed grid)
D recipes: sonnet55 28*m/44, gpt56 32*m/44 (1024-px tiles, symbol 44 px); gem38 FOV = sqrt(1090)*m/2.5.
Also reports the generation-1 values for the same conditions."""
import json, math, numpy as np, mech as M, ceilings as K
CEIL = json.load(open('ceilings.json'))
def side(v, a):
    W, H = a['width'], a['height']; L = max(W, H); m = min(a['template_size']); s = min(1.0, 1600 / L)
    if v in ('sonnet55', 'opus55'): return 28 / s
    if v in ('gpt55', 'gpt56'): return 32 / s
    if v == 'gpt56orig':
        so = min(1.0, math.sqrt(30000 * 1024 / (W * H)) * 0.98); return 32 / so
    if v == 'gem38': return L / math.sqrt(1090)
    if v == 'gem38uh': return L / math.sqrt(2200)
    if v == 'D_sonnet55': return 28 * m / 44
    if v == 'D_gpt56': return 32 * m / 44
    if v == 'D_gem38': return max(96, round(math.sqrt(1090) * m / 2.5)) / math.sqrt(1090)
out = {}
for d in K.TEST:
    a = json.loads((M.BENCH / 'annotations' / f'{d}.json').read_text()); tau = M.taus(a)[0]
    for v in ('sonnet55', 'opus55', 'gpt55', 'gpt56', 'gpt56orig', 'gem38', 'gem38uh', 'D_sonnet55', 'D_gpt56', 'D_gem38'):
        t = side(v, a); f, r = K.ideal_observer(a['points'], t, tau, n_off=32)
        out[f'{d}|{v}'] = {'ceiling_f1': round(f, 3), 'weak': round(2 * r / (1 + r), 3), 'sym_tokens': round(min(a['template_size']) / t, 2), 'subset': a['subset']}
json.dump(out, open('gen_ceilings.json', 'w'), indent=1)
print(f'{"condition":11s} {"strong ceiling all/sheets/views":32s} {"weak (resolvability)":22s} median sym tokens')
for v in ('sonnet55', 'opus55', 'gpt55', 'gpt56', 'gpt56orig', 'gem38', 'gem38uh', 'D_sonnet55', 'D_gpt56', 'D_gem38'):
    g = lambda sub, k: np.mean([out[f'{d}|{v}'][k] for d in K.TEST if sub in ('all', out[f'{d}|{v}']['subset'])])
    print(f'{v:11s} {g("all","ceiling_f1"):.2f} / {g("sheet","ceiling_f1"):.2f} / {g("view","ceiling_f1"):.2f}{"":14s} {g("all","weak"):.2f} / {g("sheet","weak"):.2f} / {g("view","weak"):.2f}{"":6s} {np.median([out[f"{d}|{v}"]["sym_tokens"] for d in K.TEST]):.2f}')
print('generation 1, A1: gpt %.2f claude %.2f gemini %.2f' % tuple(np.mean([CEIL[f'{d}|A1|{m}']['ceiling_f1'] for d in K.TEST]) for m in ('gpt', 'claude', 'gemini')))
