#!/usr/bin/env python3
"""Per-call geometry and per-item observed recall for the S-L law (plan claude/law_analysis_plan_v5.md).
Each record: dataset, item, model (synthetic-fit key), config, split ('tune'|'test'), n (targets scored),
obs (observed recall, mean over runs), calls = [[S, n_call, A_units], ...] where S = tokens per target side in that
call, n_call = scored targets whose centre lies in the call's kept core, A_units = native area shown / m^2.
Token geometry for every dataset uses the same rules as the synthetic fit (syn_analysis.token_side, extended).
Writes results_law/law_data.json."""
import json, math, glob, pickle, ast
from collections import defaultdict
from pathlib import Path
import numpy as np
import mech as M, ceilings as K

OUT = Path('results_law'); OUT.mkdir(exist_ok=True)
FIXED = {'gemini25': 258, 'gemma4': 280, 'gem38': 1090, 'gem38lo': 257, 'gem38uh': 2210}


def tside(v, W, H):
    """Native-to-token side (in sent pixels) for an image sent at W x H."""
    if v in FIXED: return math.sqrt(W * H / FIXED[v])
    if v == 'gpt54': return 32 / K.gpt_fit(W, H, 2500)
    if v in ('gpt56', 'gpt55'): return 32 / K.gpt_fit(W, H, 8192)
    if v == 'qwen3vl': return 32 / min(1.0, math.sqrt(2500 * 1024 / (W * H)))
    if v == 'sonnet46': rw, rh = M.claude_resized(W, H); return 28 * W / rw
    if v in ('sonnet55', 'opus55'): rw, rh = M.claude_resized(W, H, 2576, 4784); return 28 * W / rw
    raise KeyError(v)


_TS = {}
def S_of(v, W, H, m_sent):
    k = (v, int(W), int(H))
    if k not in _TS: _TS[k] = tside(v, W, H)
    return m_sent / _TS[k]


def gm(a): return math.sqrt(a['template_size'][0] * a['template_size'][1])


def in_box(p, b): return b[0] <= p[0] < b[2] and b[1] <= p[1] < b[3]


RECS = []


# ---------------- synthetic (fit data) ----------------
def synthetic():
    rows = json.load(open('syn/analysis_robust.json'))['rows']
    for r in rows:
        A = r['W'] * r['H'] / r['m'] ** 2
        RECS.append({'ds': 'syn', 'exp': r['exp'], 'level': r['level'], 'item': r['id'], 'model': r['v'], 'config': 'whole', 'split': 'fit',
                     'n': r['n'], 'obs': r['recall'], 'calls': [[r['S'], r['n'], A]]})


# ---------------- REDP-X40 ----------------
def redp():
    import analyze_test as A, x_new_score as X
    GEN1 = {'gpt': ('gpt54', 'px'), 'claude': ('sonnet46', 'px_claude'), 'gemini': ('gemini25', 'n1000'), 'gemma4': ('gemma4', 'n1000'), 'qwen3vl': ('qwen3vl', 'n1000')}
    GEN2 = {'gpt56': ('gpt56', 'px'), 'sonnet55': ('sonnet55', 'px'), 'gem38': ('gem38', 'n1000')}
    items = list(K.DEV) + list(K.TEST)
    def a1_geom(rfile):
        g = {}
        for l in open(rfile):
            r = json.loads(l); g[r['did']] = (r['scale'], r['sent_size'])
        return g
    def tile_geom(rfile, key):
        g = defaultdict(dict)
        for l in open(rfile):
            r = json.loads(l); g[r['did']][(r['i'], r['j'])] = r
        return g
    for src, table, root in (('gen1', GEN1, A.R), ('gen2', GEN2, X.GEN)):
        for k, (v, dec) in table.items():
            # whole drawing
            per = A.mean_per(A.score_whole(k, dec) if src == 'gen1' else X.at(A.score_whole, k, dec))
            g = a1_geom(Path(root) / f'A1_{k}_run0.jsonl')
            for d in items:
                if d not in per or d not in g: continue
                a = A.ANN[d]; m = gm(a); s, (w, h) = g[d]; n = a['count']
                RECS.append({'ds': 'redp', 'item': d, 'model': v, 'config': 'A1', 'split': 'tune' if d in K.DEV else 'test', 'n': n,
                             'obs': float(per[d][1]), 'calls': [[S_of(v, w, h, m * s), n, a['width'] * a['height'] / m ** 2]]})
            # tiles
            f = Path(root) / f'D_{k}_run0.jsonl'
            if not f.exists(): continue
            per = A.mean_per(A.score_tiles('D', k) if src == 'gen1' else X.at(A.score_tiles, 'D', k))
            tg = tile_geom(f, None)
            for d in items:
                if d not in per or d not in tg: continue
                a = A.ANN[d]; m = gm(a); pts = a['points']; calls = []
                for r in tg[d].values():
                    zu = r['z0'] * r['up']; w, h = r['tile_size']; ox, oy = r['origin_z0']; c0, c1, c2, c3 = r['core_px']
                    box = ((ox + c0) / zu, (oy + c1) / zu, (ox + c2) / zu, (oy + c3) / zu)
                    nc = sum(in_box(p, box) for p in pts)
                    calls.append([S_of(v, w, h, m * zu), nc, (w / zu) * (h / zu) / m ** 2])
                RECS.append({'ds': 'redp', 'item': d, 'model': v, 'config': 'D', 'split': 'tune' if d in K.DEV else 'test', 'n': a['count'],
                             'obs': float(per[d][1]), 'calls': calls})


# ---------------- REDP-X40 tuning drawings (mechanism runs: tile sizes and fields of view; for the domain offset) ----------------
def redp_dev():
    import analyze_test as A
    MAP = {'gpt': 'gpt54', 'claude': 'sonnet46', 'gemini': 'gemini25', 'gemma4': 'gemma4'}
    files = sorted(glob.glob('results_mech/x2_*.jsonl') + glob.glob('results_mech/x3_*_run*.jsonl') + glob.glob('results_mech/step2_*_run*.jsonl'))
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))   # (model, cell) -> did -> run file -> (i,j) -> rec
    for f in files:
        for l in open(f):
            r = json.loads(l)
            if r.get('model') not in MAP or r['did'] not in K.DEV: continue
            groups[(r['model'], r['cell'])][r['did']][f][(r['i'], r['j'])] = r
    for (k, cell), dd in groups.items():
        v = MAP[k]
        for d, byrun in dd.items():
            a = A.ANN[d]; m = gm(a); recalls = []; geom = None
            for f, tiles in byrun.items():
                if any(t['status'] != 'success' for t in tiles.values()): continue
                pts = []
                for t in tiles.values(): pts += M.to_native(t)
                recalls.append(A.metrics(d, pts)[1]); geom = tiles
            if not recalls: continue
            calls = []
            for r in geom.values():
                zu = r['z0'] * r['up']; w, h = r['tile_size']; ox, oy = r['origin_z0']; c0, c1, c2, c3 = r['core_px']
                box = ((ox + c0) / zu, (oy + c1) / zu, (ox + c2) / zu, (oy + c3) / zu)
                calls.append([S_of(v, w, h, m * zu), sum(in_box(p, box) for p in a['points']), (w / zu) * (h / zu) / m ** 2])
            RECS.append({'ds': 'redp', 'item': d, 'model': v, 'config': 'dev_' + cell, 'split': 'tune', 'n': a['count'],
                         'obs': float(np.mean(recalls)), 'calls': calls})


# ---------------- FloorPlanCAD ----------------
def fpc():
    import fpc_score as F
    P = pickle.load(open('results_fpc/fpc_per.pkl', 'rb'))
    KEY = {'gpt': 'gpt54', 'claude': 'sonnet46', 'gemini': 'gemini25'}
    def scored_gt(a):
        tau = M.taus(a)[0]; return [p for p in a['points'] if not F.near_edge(p[0], p[1], a['ignore_rects'], tau)]
    for key, per in P.items():
        b, c, k = ast.literal_eval(key)
        if c not in ('A1', 'D') or k == 'tm': continue
        v = KEY.get(k, k)
        rf = F.RF / b / f'{c}_{k}_run0.jsonl'
        if not rf.exists(): continue
        geo = defaultdict(list)
        for l in open(rf):
            r = json.loads(l)
            if r.get('status') not in ('success', 'parse_error'): continue
            geo[r['did']].append(r)
        split = (lambda d: 'tune' if d in F.DEV else 'test') if b == 'bench' else (lambda d: 'test')
        for d, runs in per.items():
            recs = [x for x in runs if not np.isnan(x[1])]
            if not recs or d not in geo: continue
            a = F.ANN[d]; m = gm(a); gt = scored_gt(a); n = len(gt)
            if n == 0: continue
            calls = []
            if c == 'A1':
                r = geo[d][-1]; s = r['scale']; w, h = r['sent_size']
                calls = [[S_of(v, w, h, m * s), n, a['width'] * a['height'] / m ** 2]]
            else:
                seen = {}
                for r in geo[d]: seen[(r['i'], r['j'])] = r
                for r in seen.values():
                    zu = r['z0'] * r['up']; w, h = r['tile_size']; ox, oy = r['origin_z0']; c0, c1, c2, c3 = r['core_px']
                    box = ((ox + c0) / zu, (oy + c1) / zu, (ox + c2) / zu, (oy + c3) / zu)
                    calls.append([S_of(v, w, h, m * zu), sum(in_box(p, box) for p in gt), (w / zu) * (h / zu) / m ** 2])
            RECS.append({'ds': 'fpc_' + b, 'item': d, 'model': v, 'config': c, 'split': split(d), 'n': n,
                         'obs': float(np.mean([x[1] for x in recs])), 'calls': calls})


# ---------------- FSC-147 (domain-neutral prompt) ----------------
def fsc():
    ROOT = Path('../data/fsc147'); ITEMS = {it['name']: it for it in json.load(open(ROOT / 'x8_items.json'))}
    ANN = json.load(open(ROOT / 'x8_ann_subset.json'))
    val = set(json.load(open(ROOT / 'Train_Test_Val_FSC_147.json'))['val'])
    sc = json.load(open(ROOT / 'x8_scores_obj.json'))
    for f in sorted((ROOT / 'results').glob('*_?_obj.jsonl')):
        vendor, cond = f.stem[:-4].rsplit('_', 1)
        recs = defaultdict(dict)
        for l in open(f):
            r = json.loads(l)
            if r['status'] in ('success', 'parse_error'): recs[r['name']][r['v']] = r
        for name, vs in recs.items():
            key = f'{vendor}|{cond}|{name}'
            if key not in sc or name not in ITEMS: continue
            it = ITEMS[name]; m = it['m']; pts = [tuple(p) for p in ANN[name]['points']]; calls = []
            for r in vs.values():
                w, h = r['size']; core = r['core']; s = r['scale']
                # area shown: the view in native px (core plus margins), from sent size and scale
                calls.append([S_of(vendor, w, h, m * s), sum(in_box(p, core) for p in pts), (w / s) * (h / s) / m ** 2])
            RECS.append({'ds': 'fsc', 'item': name, 'model': vendor, 'config': cond, 'split': 'tune' if name in val else 'test',
                         'n': len(pts), 'obs': sc[key]['recall'], 'stratum': it['stratum'], 'calls': calls})


if __name__ == '__main__':
    synthetic(); print('syn', len(RECS), flush=True)
    redp(); redp_dev(); print('redp', len(RECS), flush=True)
    fpc(); print('fpc', len(RECS), flush=True)
    fsc(); print('fsc', len(RECS), flush=True)
    json.dump(RECS, open(OUT / 'law_data.json', 'w'))
