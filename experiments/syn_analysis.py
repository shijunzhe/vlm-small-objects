#!/usr/bin/env python3
"""X1 analysis: recall as a function of tokens per symbol S = m / t (t = token side in canvas pixels, from each
interface's documented geometry), model-specific threshold S50 (logistic fit on the resolution sweep, bootstrap over
images), capacity slope at fixed S (cap sweep), and the area sweep re-expressed in S.
Reading: SYN_READING (default robust = lenient JSON + Claude 5.5 frame rule; 'registered' = strict)."""
import json, math, os, sys
import numpy as np
from scipy.optimize import curve_fit
import mech as M, ceilings as K
import syn as SY

FIXED = {'gemini25': 258, 'gemma4': 280, 'gem38': 1090, 'gem38lo': 257, 'gem38uh': 2210}
NAMES = {'gpt54': 'GPT-5.4', 'sonnet46': 'Sonnet 4.6', 'gemini25': 'Gemini 2.5 Flash', 'gemma4': 'Gemma 4', 'qwen3vl': 'Qwen3-VL',
         'gpt56': 'GPT-5.6', 'sonnet55': 'Sonnet 5.5', 'gem38': 'Gemini 3.8 Flash', 'gem38lo': 'Gemini 3.8 low', 'gem38uh': 'Gemini 3.8 ultra-high'}


def token_side(v, W, H):
    if v in FIXED: return math.sqrt(W * H / FIXED[v])
    if v == 'gpt54': return 32 / K.gpt_fit(W, H, 2500)
    if v == 'gpt56': return 32 / K.gpt_fit(W, H, 8192)
    if v == 'qwen3vl': return 32 / min(1.0, math.sqrt(2500 * 1024 / (W * H)))   # measured: ~2,500 image tokens at >= 2048^2
    if v == 'sonnet46': rw, rh = M.claude_resized(W, H); return 28 * W / rw
    if v == 'sonnet55': rw, rh = M.claude_resized(W, H, 2576, 4784); return 28 * W / rw


def per_image():
    items = {it['id']: it for it in json.load(open(SY.ROOT / 'items.json'))}
    rows = []
    for v in SY.CONV:
        f = SY.OUT / f'{v}_run0.jsonl'
        if not f.exists(): continue
        for l in open(f):
            r = json.loads(l)
            if r['status'] not in ('success', 'parse_error'): continue
            it = items[r['id']]; W, H = it.get('canvas', [1024, 1024])
            pts = SY.decode(v, r.get('shapes', []) if r['status'] == 'success' else [], (W, H), raw=r.get('raw') if r['status'] == 'parse_error' else None, iid=r['id'], m=it['m'])
            tp, n_p, n_g = M.match(pts, [tuple(p) for p in it['points']], it['m'] / 2)
            rows.append({'v': v, 'id': it['id'], 'exp': it['exp'], 'level': it['level'], 'n': n_g, 'm': it['m'], 'W': W, 'H': H,
                         'S': it['m'] / token_side(v, W, H), 'recall': tp / n_g, 'prec': tp / max(n_p, 1), 'f1': 2 * tp / (n_p + n_g),
                         'in_tok': (r.get('usage') or {}).get('input_tokens')})
    return rows


def logistic(logS, a, b):
    return 1 / (1 + np.exp(-b * (logS - a)))


def s50(rows, rng=None):
    x = np.log([r['S'] for r in rows]); y = np.array([r['recall'] for r in rows])
    if rng is not None:
        idx = rng.integers(0, len(rows), len(rows)); x, y = x[idx], y[idx]
    try:
        (a, b), _ = curve_fit(logistic, x, y, p0=[0.0, 3.0], bounds=([-4, 0.1], [4, 40]), maxfev=20000)
        return math.exp(a)
    except Exception:
        return float('nan')


def main():
    # the within-model budget arms (gem38lo, gem38uh) are analysed separately in t1_syn.py
    rows = [r for r in per_image() if r['v'] not in ('gem38lo', 'gem38uh')]; rng = np.random.default_rng(0); out = {}
    print('model            S50 [95% CI] (res sweep)   recall@S<0.5  recall@S>=1.5   cap slope dRecall/dlog2(n) [95% CI]  S(cap)')
    for v in SY.CONV:
        res = [r for r in rows if r['v'] == v and r['exp'] == 'res']
        if not res: continue
        cap = [r for r in rows if r['v'] == v and r['exp'] == 'cap']
        est = s50(res); bs = [s50(res, rng) for _ in range(1000)]; lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        lowS = np.mean([r['recall'] for r in res if r['S'] < 0.5]) if any(r['S'] < 0.5 for r in res) else float('nan')
        hiS = np.mean([r['recall'] for r in res if r['S'] >= 1.5]) if any(r['S'] >= 1.5 for r in res) else float('nan')
        def slope(rs):
            x = np.log2([r['n'] for r in rs]); y = np.array([r['recall'] for r in rs]); return np.polyfit(x, y, 1)[0]
        sl = slope(cap); bsl = [slope([cap[i] for i in rng.integers(0, len(cap), len(cap))]) for _ in range(2000)]
        sl_lo, sl_hi = np.percentile(bsl, [2.5, 97.5])
        out[v] = {'S50': est, 'S50_ci': [lo, hi], 'recall_lowS': lowS, 'recall_hiS': hiS, 'cap_slope': sl, 'cap_slope_ci': [sl_lo, sl_hi],
                  'S_cap': cap[0]['S'] if cap else None,
                  'cap_recall_n2': np.mean([r['recall'] for r in cap if r['level'] == 2]), 'cap_recall_n96': np.mean([r['recall'] for r in cap if r['level'] == 96]),
                  'n_items': {'res': len(res), 'cap': len(cap)}}
        print(f'{NAMES[v]:16s} {est:5.2f} [{lo:4.2f}, {hi:4.2f}]           {lowS:5.2f}         {hiS:5.2f}          {sl:+.3f} [{sl_lo:+.3f}, {sl_hi:+.3f}]           {out[v]["S_cap"]:.2f}')
    print('\narea sweep (16 targets, symbol 40 px): S and recall per canvas')
    for v in SY.CONV:
        ar = [r for r in rows if r['v'] == v and r['exp'] == 'area']
        if not ar: continue
        cells = []
        for lv in [f'{w}x{h}' for w, h in SY.AREA_CANVAS]:
            rs = [r for r in ar if r['level'] == lv]
            cells.append(f'{rs[0]["S"]:4.2f}:{np.mean([r["recall"] for r in rs]):.2f}' if rs else '   -    ')
        out.setdefault(v, {})['area'] = {lv: [np.mean([r['S'] for r in ar if r['level'] == lv]), np.mean([r['recall'] for r in ar if r['level'] == lv]), np.mean([r['f1'] for r in ar if r['level'] == lv])] for lv in sorted({r['level'] for r in ar})}
        print(f'{NAMES[v]:16s} ' + '  '.join(cells))
    print('\nX1e information-fixed upsampling x2 (same image, Lanczos): recall at native vs upsampled, paired over images')
    for v in SY.CONV:
        up = {r['id']: r for r in rows if r['v'] == v and r['exp'] == 'up'}
        if not up: continue
        base = {r['id']: r for r in rows if r['v'] == v and r['exp'] == 'res'}
        pairs = [(base[f"res_{u['level']}_{u['id'].rsplit('_', 1)[1]}"], u) for u in up.values() if f"res_{u['level']}_{u['id'].rsplit('_', 1)[1]}" in base]
        d = np.array([b['recall'] for b in [p[1] for p in pairs]]) - np.array([p[0]['recall'] for p in pairs])
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]; lo, hi = np.percentile(bs, [2.5, 97.5])
        df = np.array([p[1]['f1'] - p[0]['f1'] for p in pairs]); bsf = [df[rng.integers(0, len(df), len(df))].mean() for _ in range(4000)]
        S0 = np.mean([p[0]['S'] for p in pairs]); S1 = np.mean([p[1]['S'] for p in pairs])
        per_lv = {lv: (np.mean([p[0]['recall'] for p in pairs if p[0]['level'] == lv]), np.mean([p[1]['recall'] for p in pairs if p[0]['level'] == lv])) for lv in SY.UP_SIZES}
        print(f'{NAMES[v]:16s} S {S0:.2f}->{S1:.2f}  recall diff {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]  F1 diff {df.mean():+.2f} [{np.percentile(bsf, 2.5):+.2f}, {np.percentile(bsf, 97.5):+.2f}]  by m: ' +
              '  '.join(f'{lv}px {a:.2f}->{b:.2f}' for lv, (a, b) in per_lv.items()) + f'  (n={len(pairs)})')
        out.setdefault(v, {})['up'] = {'S': [S0, S1], 'd_recall': [d.mean(), lo, hi], 'd_f1': [df.mean(), *np.percentile(bsf, [2.5, 97.5])], 'by_m': per_lv, 'n': len(pairs)}
    print('\nX1d clutter (m=16 px, 8+8): recall by clutter density; slope per log2(1+c)')
    for v in SY.CONV:
        cl = [r for r in rows if r['v'] == v and r['exp'] == 'clut']
        if not cl: continue
        x = np.log2(1 + np.array([r['level'] for r in cl])); y = np.array([r['recall'] for r in cl])
        sl = np.polyfit(x, y, 1)[0]; bsl = []
        for _ in range(2000):
            i = rng.integers(0, len(cl), len(cl)); bsl.append(np.polyfit(x[i], y[i], 1)[0])
        lo, hi = np.percentile(bsl, [2.5, 97.5])
        cells = '  '.join(f'{c}:{np.mean([r["recall"] for r in cl if r["level"] == c]):.2f}' for c in SY.CLUTTER)
        print(f'{NAMES[v]:16s} S {cl[0]["S"]:.2f}  {cells}   slope {sl:+.3f} [{lo:+.3f}, {hi:+.3f}]')
        out.setdefault(v, {})['clut'] = {'S': cl[0]['S'], 'recall': {c: np.mean([r['recall'] for r in cl if r['level'] == c]) for c in SY.CLUTTER}, 'slope': [sl, lo, hi]}
    json.dump({'reading': SY.READING, 'models': out, 'rows': rows}, open(SY.ROOT / f'analysis_{SY.READING}.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
