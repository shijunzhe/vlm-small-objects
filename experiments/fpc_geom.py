#!/usr/bin/env python3
"""A priori geometry for FPC-300 and FPC-sheets: tokens per symbol side S for each model's whole-image call (A1) and
its tile recipe (D), from the measured interfaces (no model calls). Writes ../data/fpc/geometry.json."""
import json, math, glob
import numpy as np
import mech as M, ceilings as K
MODELS = ['gpt54', 'sonnet46', 'gemini25', 'gemma4', 'qwen3vl', 'gpt56', 'sonnet55', 'gem38', 'gem38lo', 'gem38uh']
FIXED = {'gemini25': 258, 'gemma4': 280, 'gem38': 1090, 'gem38lo': 257, 'gem38uh': 2210}


def s_whole(v, W, H, m):
    s = min(1.0, 1600 / max(W, H)); w, h = round(W * s), round(H * s)   # A1 protocol: long side <= 1600
    if v in FIXED: return m * s / (max(w, h) / math.sqrt(FIXED[v]))
    if v == 'gpt54': return m * s * K.gpt_fit(w, h, 2500) / 32
    if v == 'gpt56': return m * s * K.gpt_fit(w, h, 8192) / 32
    if v == 'qwen3vl': return m * s * min(1.0, math.sqrt(2500 * 1024 / (w * h))) / 32
    if v == 'sonnet46': rw, rh = M.claude_resized(w, h); return m * s * rw / w / 28
    if v == 'sonnet55': rw, rh = M.claude_resized(w, h, 2576, 4784); return m * s * rw / w / 28


def s_tiles(v, a):
    mn = min(a['template_size']); m = math.sqrt(a['template_size'][0] * a['template_size'][1])
    if v in ('gemini25', 'gem38'):
        F = max(96, round(math.sqrt(FIXED[v]) * mn / 2.5)); return m / (F / math.sqrt(FIXED[v]))
    if v == 'gemma4':
        F = max(96, round(16 * mn / 1.5)); return m / (F / 16.7)
    z = 44 / mn; t = 28 if v.startswith('sonnet') else 32
    return m * z / t


if __name__ == '__main__':
    out = {}
    for bench in ('bench', 'sheets'):
        for f in sorted(glob.glob(f'../data/fpc/{bench}/annotations/*.json')):
            a = json.load(open(f)); m = math.sqrt(a['template_size'][0] * a['template_size'][1])
            out[a['id']] = {v: {'A1': s_whole(v, a['width'], a['height'], m), 'D': s_tiles(v, a)} for v in MODELS}
    json.dump(out, open('../data/fpc/geometry.json', 'w'), indent=1)
    for sub, pat in (('FPC-300', 'f'), ('sheets k1', '_k1'), ('sheets k2', '_k2'), ('sheets k3', '_k3')):
        ids = [i for i in out if (i.startswith('f') if pat == 'f' else i.endswith(pat))]
        print(sub, len(ids), ' '.join(f'{v}:{np.median([out[i][v]["A1"] for i in ids]):.2f}/{np.mean([out[i][v]["A1"] < 1 for i in ids]):.2f}' for v in MODELS))
    print('tiles S median', ' '.join(f'{v}:{np.median([out[i][v]["D"] for i in out]):.2f}' for v in MODELS))
