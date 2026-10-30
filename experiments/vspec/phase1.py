#!/usr/bin/env python3
"""Phase 1: S90 at n0 = 8. usage: phase1.py run | analyze"""
import json, sys, math, random
from collections import defaultdict
import numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import vspec as V, vrun as R

ROOT = V.ROOT; OUT = ROOT / 'results'
S_LEVELS = [0.2, 0.3, 0.45, 0.65, 0.9, 1.3, 1.9, 2.8]


def run():
    items = json.load(open(ROOT / 'items_p1.json'))
    # interleave models so that every vendor progresses in parallel
    items.sort(key=lambda it: (it['i'], S_LEVELS.index(it['S']), V.ORDER.index(it['model'])))
    R.run_items(items, OUT / 'phase1.jsonl', workers=12)


def s90_from_cells(cells):
    """cells: {S: pooled}. Smallest level such that it and all higher levels pass; log-interpolated with the
    level below on R and P. Returns (S90, status)."""
    lv = sorted(cells)
    passing = [cells[s]['pass_'] for s in lv]
    # Amendment 2 (2026-10-26, approved by the user): a failure confined to the highest tested level does not
    # right-censor S90 when the level below it passes (S90 is by definition the smallest S that reaches q, and
    # right-censoring means S90 > 2.8). The top-level failure is reported. The contiguity rule otherwise applies.
    if not passing[-1]:
        if len(lv) >= 2 and passing[-2]:
            lv = lv[:-1]; passing = passing[:-1]
        else:
            return None, 'right_censored'
    j = len(lv) - 1
    while j > 0 and passing[j - 1]:
        j -= 1
    if j == 0:
        return lv[0], 'left_censored'
    lo, hi = lv[j - 1], lv[j]
    def cross(key):
        a, b = cells[lo][key], cells[hi][key]
        if not (a == a):               # undefined at lo (no points returned) -> crossing taken at hi
            return math.log(hi)
        if a >= 0.9:                   # this metric already meets q at lo
            return math.log(lo)
        return math.log(lo) + (0.9 - a) / (b - a) * (math.log(hi) - math.log(lo))
    return math.exp(max(cross('R'), cross('P'))), 'identified'


def analyze(B=2000, seed=0):
    items = {it['id']: it for it in json.load(open(ROOT / 'items_p1.json'))}
    recs = R.load_results([OUT / 'phase1.jsonl'])
    rows = defaultdict(list)   # (model, S) -> list of (i, scored)
    for iid, it in items.items():
        if iid in recs:
            rows[(it['model'], it['S'])].append((it['i'], R.score_record(it, recs[iid])))
    out = {}
    rng = random.Random(seed)
    for m in V.ORDER:
        cells = {S: R.pooled([r for _, r in rows[(m, S)]]) for S in S_LEVELS if rows[(m, S)]}
        if len(cells) < len(S_LEVELS):
            out[m] = dict(status='incomplete', n_cells=len(cells)); continue
        s90, st = s90_from_cells(cells)
        prec_limited = all(not (c['P'] >= 0.9) for c in cells.values())
        # recall-only S90 (descriptive)
        rc = {S: dict(c, pass_=c['R'] >= 0.9, P=1.0) for S, c in cells.items()}
        s90R, stR = s90_from_cells(rc)
        # cluster bootstrap over base scenes i
        ids = sorted({i for (mm, S), v in rows.items() if mm == m for i, _ in v})
        boots = []
        for b in range(B):
            samp = [rng.choice(ids) for _ in ids]
            bc = {}
            for S in S_LEVELS:
                by_i = defaultdict(list)
                for i, r in rows[(m, S)]:
                    by_i[i].append(r)
                bc[S] = R.pooled([r for i in samp for r in by_i[i]])
            v, s = s90_from_cells(bc)
            boots.append(v if s == 'identified' or s == 'left_censored' else float('inf'))
        arr = np.array(boots)
        out[m] = dict(S90=s90, status='precision_limited' if prec_limited else st, S90_recall_only=s90R, status_recall_only=stR,
                      ci90=[float(np.quantile(arr, 0.05)), float(np.quantile(arr, 0.95))],
                      frac_unidentified=float(np.mean(~np.isfinite(arr))),
                      cells={str(S): {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for S, c in cells.items()})
    json.dump(out, open(OUT / 'phase1_analysis.json', 'w'), indent=1)
    for m, o in out.items():
        print(m, {k: v for k, v in o.items() if k != 'cells'})
        for S, c in o.get('cells', {}).items():
            print(f"   S={S:>5} R={c['R']:.3f} P={c['P']:.3f} pass={c['pass_']} trunc={c['trunc']} k={c['k']}")
    return out




def rebuild_amend1():
    """Amendment 1: re-render phase-1 items for gpt61/gpt56/opus55 without padding (phase tag 'p1r') on the same
    frozen layouts (k verified identical under the amended render set); gem38/qwen3vl items and results unchanged."""
    import vrun as RR
    old_items = json.load(open(ROOT / 'archive_padded16t/items_p1_padded16t.json'))
    log = json.load(open(ROOT / 'freeze_p1_log.json'))['layout_log']; ks = {d['i']: d['k'] for d in log}
    keep = [it for it in old_items if it['model'] in ('gem38', 'qwen3vl')]
    new = []
    for i in range(12):
        lay = V.layout('P1', i, 8, ks[i])
        for m in ('gpt61', 'gpt56', 'opus55'):
            for S in S_LEVELS:
                it, r = RR.make_item('p1r', lay, m, S, 'S')
                assert it is not None and not it['geo']['padded'], (i, m, S)
                new.append(it)
    json.dump(keep + new, open(ROOT / 'items_p1.json', 'w'), indent=1)
    keep_ids = {it['id'] for it in keep}
    with open(OUT / 'phase1.jsonl', 'w') as fo:
        for l in open(ROOT / 'archive_padded16t/phase1_padded16t.jsonl'):
            if json.loads(l)['id'] in keep_ids:
                fo.write(l)
    print(len(keep), 'kept items,', len(new), 'new unpadded items')


if __name__ == '__main__':
    {'run': run, 'analyze': analyze, 'rebuild_amend1': rebuild_amend1}[sys.argv[1]]()
