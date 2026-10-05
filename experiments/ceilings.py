#!/usr/bin/env python3
"""
A priori geometric ceilings, derived only from documented token geometry (no model calls).

Token-limited ideal observer: the model is assumed to know, for each instance, only which visual
token cell contains its center. Instances in the same cell merge into one detection, reported at the
cell center. Its strict F1 (Hungarian matching, tau_sym = 0.5*sqrt(template area)) is averaged over
random grid offsets. This bounds any model whose spatial information is quantized to the token grid;
it is a necessary condition, not a prediction of the observed score.

Token side in native drawing pixels, per condition (documented specs, see docs_check_and_dev_results.md):
  A1 (whole drawing, long side capped at 1600 px, as on the dev set)
    gpt    : 32 px patches; auto caps at 2500 patches (further downscale if exceeded)
    claude : 28 px patches after the documented resize (long side <= 1568, <= 1568 tokens)
    qwen3vl: 32 px per token (16 px patch, 2x2 merge); ASSUMED no provider downscale at <= 1600 px
    gemini : fixed 258 tokens = 16x16 grid over the long side (MEDIUM, default)
    gemma4 : fixed 280 tokens, ~16.7 per side over the long side (provider default)
  B  gpt original : <= 10k patches, long side <= 6000 (as sent in X4)
     gemini HIGH  : image sent at long side <= 3072, cut into 768-px crops of 16x16 tokens (48 px/token)
  C  generic tiles: drawing rescaled so the symbol short side is 44 px, 1024-px tiles
     gpt/qwen 32 px, claude 28 px (1024^2 is under both caps), gemini 1024/16, gemma 1024/16.7
  D  per-model recipe: gpt/claude/qwen as C; gemma FOV = 16*m/1.5 (symbol = 1.5 tokens);
     gemini FOV = 16*m/2.5 (symbol = 2.5 tokens)
  D-mask has D's geometry, hence D's ceiling: any D-mask gain over D is non-geometric.
"""
import json, math, sys
import numpy as np
import mech as M

DEV = ['d001', 'd004', 'd008', 'd010', 'd017', 'd019']
TEST = [f'd{i:03d}' for i in range(1, 41) if f'd{i:03d}' not in DEV]
GEMMA_SIDE = 16.7


def gpt_fit(w, h, max_patches):
    s = 1.0
    while math.ceil(w * s / 32) * math.ceil(h * s / 32) > max_patches:
        s *= 0.995
    return s


def token_side(cond, model, ann):
    """Token side in native pixels, or None if the condition does not apply to the model."""
    W, H = ann['width'], ann['height']; L = max(W, H); m = min(ann['template_size'])
    if cond == 'A1':
        s = min(1.0, 1600 / L); w, h = round(W * s), round(H * s)
        if model == 'gpt':
            return 32 / (s * gpt_fit(w, h, 2500))
        if model == 'claude':
            rw, rh = M.claude_resized(w, h); return 28 / (s * rw / w)
        if model == 'qwen3vl':
            return 32 / s
        if model == 'gemini':
            return L / 16
        if model == 'gemma4':
            return L / GEMMA_SIDE
    if cond == 'B':
        if model == 'gpt':
            s = min(1.0, 6000 / L, math.sqrt(10000 * 1024 / (W * H)) * 0.98)
            return 32 / (s * gpt_fit(round(W * s), round(H * s), 10000))
        if model == 'gemini':
            return 48 / min(1.0, 3072 / L)
        return None
    if cond in ('C', 'D', 'Dmask'):
        z = 44 / m
        if model in ('gpt', 'qwen3vl'):
            return 32 / z
        if model == 'claude':
            return 28 / z
        if cond == 'C':
            return (1024 / 16) / z if model == 'gemini' else (1024 / GEMMA_SIDE) / z
        if model == 'gemini':
            return max(96, round(16 * m / 2.5)) / 16
        if model == 'gemma4':
            return max(96, round(16 * m / 1.5)) / GEMMA_SIDE
    return None


def calls_and_tokens(cond, model, ann):
    """Image calls per drawing and image tokens per call (approximate, from the same specs)."""
    W, H = ann['width'], ann['height']; m = min(ann['template_size'])
    def ntiles(Wz, Hz, T):
        core = T - 2 * (T // 8); return math.ceil(Wz / core) * math.ceil(Hz / core)
    if cond == 'A1':
        return 1, {'gpt': 3000, 'claude': 1568, 'qwen3vl': 2500, 'gemini': 258, 'gemma4': 280}[model]
    if cond == 'B':
        return 1, {'gpt': 12000, 'gemini': 2325}[model]
    z = 44 / m
    if cond in ('C', 'D', 'Dmask') and (model in ('gpt', 'claude', 'qwen3vl') or cond == 'C'):
        n = ntiles(W * z, H * z, 1024)
        tok = {'gpt': 1229, 'claude': 1369, 'qwen3vl': 1024, 'gemini': 258, 'gemma4': 280}[model]
        return n * (2 if cond == 'Dmask' else 1), tok
    k = 2.5 if model == 'gemini' else 1.5
    F = max(96, int(round(16 * m / k)))
    return ntiles(W, H, F), (258 if model == 'gemini' else 280)


def ideal_observer(points, t, tau, n_off=64, seed=0):
    rng = np.random.default_rng(seed)
    P = np.asarray(points, float)
    f1, rec_res = [], []
    for _ in range(n_off):
        off = rng.uniform(0, t, 2)
        cells = np.floor((P + off) / t).astype(np.int64)
        uniq = np.unique(cells, axis=0)
        pred = (uniq + 0.5) * t - off
        tp, n_p, n_g = M.match([tuple(p) for p in pred], [tuple(p) for p in P], tau)
        f1.append(2 * tp / (n_p + n_g)); rec_res.append(len(uniq) / len(P))
    return float(np.mean(f1)), float(np.mean(rec_res))


MODELS = ['gpt', 'claude', 'gemini', 'gemma4', 'qwen3vl']
CONDS = ['A1', 'B', 'C', 'D']


def table(dids):
    rows = {}
    for did in dids:
        ann = json.loads((M.BENCH / 'annotations' / f'{did}.json').read_text())
        tau, _ = M.taus(ann); m = min(ann['template_size'])
        for c in CONDS:
            for mdl in MODELS:
                t = token_side(c, mdl, ann)
                if t is None:
                    continue
                f1, res = ideal_observer(ann['points'], t, tau)
                n, tok = calls_and_tokens(c, mdl, ann)
                rows[(did, c, mdl)] = {'ceiling_f1': round(f1, 3), 'resolvable': round(res, 3),
                                       'sym_tokens': round(m / t, 2), 'calls': n, 'img_tokens': n * tok,
                                       'subset': ann.get('subset')}
    return rows


def summarize(rows, dids, label):
    print(f'\n=== {label} ({len(dids)} drawings): mean ceiling F1 / resolvable / symbol tokens / calls per drawing')
    for c in CONDS:
        for mdl in MODELS:
            v = [rows[(d, c, mdl)] for d in dids if (d, c, mdl) in rows]
            if not v:
                continue
            print(f'{c:3s} {mdl:8s} ceiling {np.mean([x["ceiling_f1"] for x in v]):.2f}  '
                  f'resolvable {np.mean([x["resolvable"] for x in v]):.2f}  '
                  f'sym_tok {np.median([x["sym_tokens"] for x in v]):.2f} (median)  '
                  f'calls {np.mean([x["calls"] for x in v]):.1f}  img_tok {np.mean([x["img_tokens"] for x in v]):.0f}')


if __name__ == '__main__':
    dev = table(DEV); test = table(TEST)
    summarize(dev, DEV, 'DEV')
    summarize(test, TEST, 'TEST (all 34)')
    sheet = [d for d in TEST if test[(d, 'A1', 'gpt')]['subset'] == 'sheet']
    view = [d for d in TEST if test[(d, 'A1', 'gpt')]['subset'] == 'view']
    summarize(test, sheet, 'TEST sheets'); summarize(test, view, 'TEST views')
    out = {f'{d}|{c}|{m}': v for (d, c, m), v in {**dev, **test}.items()}
    (M.OUT.parent / 'ceilings.json').write_text(json.dumps(out, indent=1))
    print('\nwrote', M.OUT.parent / 'ceilings.json')
