#!/usr/bin/env python3
"""Scoring and pre-registered hypothesis tests for the main experiment (34 test drawings)."""
import json, math, glob
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import mech as M, ceilings as K, hybrid_verify as HV

R = Path('results_test')
TEST = K.TEST
ANN = {d: json.loads((M.BENCH / 'annotations' / f'{d}.json').read_text()) for d in TEST + K.DEV}
SUB = {d: ANN[d]['subset'] for d in ANN}
CEIL = json.load(open('ceilings.json'))
MODELS = ['gpt', 'claude', 'gemini', 'gemma4', 'qwen3vl']
rng = np.random.default_rng(0)


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def metrics(did, pts):
    a = ANN[did]; tau, _ = M.taus(a); g = a['count']
    tp, n, _ = M.match(pts, a['points'], tau)
    return np.array([2 * tp / (n + g), tp / g, tp / max(n, 1), abs(n - g) / g, n / g])


def runs_of(pattern):
    return sorted(glob.glob(str(R / pattern)))


# ---------- per-condition point extraction: {did: [metrics per run]} ----------
def score_whole(model, decode):
    per = defaultdict(list)
    for f in runs_of(f'A1_{model}_run*.jsonl'):
        last = {}
        for l in open(f):
            r = json.loads(l)
            if r['status'] in ('success', 'parse_error'):   # parse errors count as an empty answer
                if r['status'] == 'parse_error':
                    r['shapes'] = []
                last[r['did']] = r
        for did, r in last.items():
            s = r['scale']; w, h = r['sent_size']; pts = []
            for q in r['shapes']:
                pt = M.point_of(q)
                if pt is None:
                    continue
                x, y = pt
                if decode == 'n1000':
                    x, y = x / 1000 * w, y / 1000 * h
                elif decode == 'px_claude':   # Claude reports in the frame of its internally resized image
                    rw, rh = M.claude_resized(w, h); x, y = x * w / rw, y * h / rh
                pts.append((x / s, y / s))
            per[did].append(metrics(did, pts))
    return per


def score_B(model):
    per = defaultdict(list)
    for f in runs_of(f'B_{model}_run*.jsonl'):
        for l in open(f):
            r = json.loads(l)
            if r['status'] != 'success':
                continue
            if model == 'gpt':
                s = r['scale']; pts = [(q['cx'] / s, q['cy'] / s) for q in r['shapes'] if isinstance(q, dict) and num(q.get('cx')) and num(q.get('cy'))]
            else:
                pts = [tuple(p) for p in r['pts_native']]
            per[r['did']].append(metrics(r['did'], pts))
    return per


def score_tiles(cond, model):
    per = defaultdict(list)
    for f in runs_of(f'{cond}_{model}_run*.jsonl'):
        last = {}
        for l in open(f):
            r = json.loads(l); last[(r['did'], r['i'], r['j'], r.get('half', ''))] = r
        pts = defaultdict(list); bad = defaultdict(int); n = defaultdict(int)
        for (did, *_), r in last.items():
            n[did] += 1
            if r['status'] != 'success':
                bad[did] += 1; continue
            p = M.to_native(r)
            if 'keep_x' in r:
                lo, hi = r['keep_x']; z0, up = r['z0'], r['up']; x0 = r['origin_z0'][0]
                p = [q for q in p if lo <= (q[0] * z0 - x0) * up < hi]
            pts[did] += p
        for did in n:
            per[did].append(metrics(did, pts[did]))
    return per


def score_H(model):
    per = defaultdict(list); recs = {}
    f = R / f'hyb_{model}.jsonl'
    if not f.exists():
        return per
    for l in open(f):
        r = json.loads(l); recs[(r['did'], r['k'])] = r
    for did in TEST:
        rs = [r for (d, k), r in recs.items() if d == did]
        if not rs:
            continue
        per[did].append(metrics(did, [(r['x'], r['y']) for r in rs if r.get('match') is True]))
    return per


def tm_scores():
    ev = json.load(open('../data/redp40/results/tm_eval.json'))
    cands = {d: json.load(open(f'../data/redp40/results/tm_candidates/{d}.json'))['candidates'] for d in ANN}
    grid = np.round(np.arange(0.3, 1.0, 0.01), 2)
    dev_f = [np.mean([metrics(d, [(x, y) for x, y, s, v in cands[d] if s >= t])[0] for d in K.DEV]) for t in grid]
    thr = float(grid[int(np.argmax(dev_f))])
    dev = {d: [metrics(d, [(x, y) for x, y, s, v in cands[d] if s >= thr])] for d in TEST}
    loo = {d: [np.array([ev[d]['loo']['f1_sym'], np.nan, np.nan, np.nan, np.nan])] for d in TEST}
    return dev, loo, thr


def mean_per(per):
    return {d: np.mean(v, axis=0) for d, v in per.items() if v}


def summary(per, dids):
    m = mean_per(per); v = [m[d] for d in dids if d in m]
    return (np.nanmean(v, axis=0), len(v)) if v else (None, 0)


def boot_diff(pa, pb, dids, k=0, B=4000):
    a, b = mean_per(pa), mean_per(pb)
    ds = [d for d in dids if d in a and d in b]
    x = np.array([a[d][k] - b[d][k] for d in ds])
    if len(x) == 0:
        return None
    bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(B)]
    p = 2 * min(np.mean(np.array(bs) <= 0), np.mean(np.array(bs) >= 0))
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), min(1.0, p), len(ds)


if __name__ == '__main__':
    S = {}
    for m in MODELS:
        S[('A1', m)] = score_whole(m, 'px')
        S[('A2', m)] = score_whole(m, 'n1000' if m in ('gemini', 'gemma4', 'qwen3vl') else 'px')
        S[('D', m)] = score_tiles('D', m)
        S[('C', m)] = S[('D', m)] if m in ('gpt', 'claude', 'qwen3vl') else score_tiles('C', m)
        S[('H', m)] = score_H(m)
    for m in ('gpt', 'gemini'):
        S[('B', m)] = score_B(m)
    for m in ('gpt', 'claude'):
        S[('Dmask', m)] = score_tiles('Dmask', m)
    tm_dev, tm_loo, thr = tm_scores()
    sheets = [d for d in TEST if SUB[d] == 'sheet']; views = [d for d in TEST if SUB[d] == 'view']

    print(f'TM threshold chosen on dev only: {thr}')
    print(f'\n{"cond":6s} {"model":8s} | {"all34 F1":>8s} {"recall":>6s} {"prec":>5s} {"relerr":>6s} | {"sheets":>6s} {"views":>6s} | n   runs')
    for (c, m), per in sorted(S.items(), key=lambda kv: (['A1', 'A2', 'B', 'C', 'D', 'Dmask', 'H'].index(kv[0][0]), MODELS.index(kv[0][1]))):
        v, n = summary(per, TEST)
        if v is None:
            continue
        vs, _ = summary(per, sheets); vv, _ = summary(per, views)
        nr = max((len(x) for x in per.values()), default=0)
        print(f'{c:6s} {m:8s} | {v[0]:8.2f} {v[1]:6.2f} {v[2]:5.2f} {v[3]:6.2f} | {vs[0] if vs is not None else float("nan"):6.2f} {vv[0] if vv is not None else float("nan"):6.2f} | {n:2d}  {nr}')
    for lab, per in (('TM dev-thr', tm_dev), ('TM LOO', tm_loo)):
        v, n = summary(per, TEST); vs, _ = summary(per, sheets); vv, _ = summary(per, views)
        print(f'{lab:15s} | {v[0]:8.2f} {"":6s} {"":5s} {"":6s} | {vs[0]:6.2f} {vv[0]:6.2f} | {n:2d}')
    bound = {d: (lambda R_: 2 * R_ / (1 + R_))(M.match([(x, y) for x, y, s, v in HV.candidates(d)], ANN[d]['points'], M.taus(ANN[d])[0])[0] / ANN[d]['count']) for d in TEST}
    print(f'H candidate bound (perfect verifier): {np.mean(list(bound.values())):.3f}')

    print('\n=== pre-registered hypotheses')
    def holm(ps):
        o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
        for i, k in enumerate(o):
            run = max(run, (len(ps) - i) * ps[k]); adj[k] = min(1, run)
        return adj
    h1 = {m: boot_diff(S[('D', m)], S[('A1', m)], TEST) for m in MODELS}
    adj = holm([h1[m][3] for m in MODELS])
    for m, a in zip(MODELS, adj):
        d = h1[m]; print(f'H1  D - A1  {m:8s} {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}] n={d[4]} Holm p={a:.4f}')
    rankA = sorted(MODELS, key=lambda m: -summary(S[('A1', m)], TEST)[0][0]); rankD = sorted(MODELS, key=lambda m: -summary(S[('D', m)], TEST)[0][0])
    print(f'H2  rank A1 {rankA}  rank D {rankD}  gemma A1 rank {rankA.index("gemma4")+1}, D rank {rankD.index("gemma4")+1}')
    for m in ('gemini', 'gemma4'):
        d = boot_diff(S[('D', m)], S[('C', m)], TEST); print(f'H3  D - C   {m:8s} {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]')
    for m in ('gpt', 'gemini'):
        d = boot_diff(S[('B', m)], S[('D', m)], TEST); print(f'H4  B - D   {m:8s} {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]')
    for m in ('gemini', 'gemma4', 'qwen3vl'):
        f1 = boot_diff(S[('A2', m)], S[('A1', m)], TEST, k=0); cnt = boot_diff(S[('A2', m)], S[('A1', m)], TEST, k=3)
        print(f'H5  A2 - A1 {m:8s} F1 {f1[0]:+.2f} [{f1[1]:+.2f}, {f1[2]:+.2f}]   |relerr| {cnt[0]:+.2f} (count unchanged by construction)')
    for m in MODELS:
        gs = boot_diff(S[('D', m)], S[('A1', m)], sheets); gv = boot_diff(S[('D', m)], S[('A1', m)], views)
        print(f'H6  D-A1 gain {m:8s} sheets {gs[0]:+.2f}  views {gv[0]:+.2f}  -> {"holds" if gv[0] < gs[0] else "fails"}')
    best = max(MODELS, key=lambda m: summary(S[('D', m)], TEST)[0][0])
    for lab, tm in (('dev-thr', tm_dev), ('LOO', tm_loo)):
        d = boot_diff(S[('D', best)], tm, TEST); print(f'H7  best D ({best}) - TM {lab}: {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]')
    for m in ('gpt', 'claude'):
        d = boot_diff(S[('Dmask', m)], S[('D', m)], TEST); print(f'H8  Dmask - D {m:8s} {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]')
    # H9: ceilings
    viol, tot = [], 0
    cmap = {'A1': 'A1', 'A2': 'A1', 'B': 'B', 'C': 'C', 'D': 'D', 'Dmask': 'D'}
    for (c, m), per in S.items():
        if c not in cmap:
            continue
        mp = mean_per(per)
        for lab, ds in (('sheets', sheets), ('views', views)):
            ds = [d for d in ds if d in mp and f'{d}|{cmap[c]}|{m}' in CEIL]
            if not ds:
                continue
            obs = np.mean([mp[d][0] for d in ds]); cel = np.mean([CEIL[f'{d}|{cmap[c]}|{m}']['ceiling_f1'] for d in ds]); tot += 1
            if obs > cel + 0.05:
                viol.append((c, m, lab, round(obs, 2), round(cel, 2)))
    print(f'H9  cells above ceiling+0.05: {len(viol)}/{tot}  {viol}')
    for m in ('gemini', 'gemma4'):
        vs = summary(S[('A1', m)], sheets)[0][0]; vv = summary(S[('A1', m)], views)[0][0]
        v2s = summary(S[('A2', m)], sheets)[0][0]
        print(f'H10 {m:8s} A1 sheets {vs:.2f} (<=0.10?) views {vv:.2f} | A2 sheets {v2s:.2f}')
    for m in ('gpt', 'claude'):
        mp = mean_per(S[('A1', m)]); ds = [d for d in TEST if d in mp]
        x = [CEIL[f'{d}|A1|{m}']['ceiling_f1'] for d in ds]; y = [mp[d][0] for d in ds]
        rho = spearmanr(x, y)[0]
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(ds), len(ds)); bs.append(spearmanr(np.array(x)[i], np.array(y)[i])[0])
        print(f'H11 {m:8s} Spearman(A1 obs, ceiling) {rho:+.2f} [{np.nanpercentile(bs,2.5):+.2f}, {np.nanpercentile(bs,97.5):+.2f}] n={len(ds)}')
    for m in MODELS:
        lo = [d for d in TEST if CEIL[f'{d}|A1|{m}']['ceiling_f1'] < 0.5]; hi = [d for d in TEST if CEIL[f'{d}|A1|{m}']['ceiling_f1'] >= 0.9]
        gl = boot_diff(S[('D', m)], S[('A1', m)], lo); gh = boot_diff(S[('D', m)], S[('A1', m)], hi)
        print(f'H12 {m:8s} D-A1 gain low-ceiling {gl[0]:+.2f} (n={gl[4]})  high-ceiling {gh[0]:+.2f} [{gh[1]:+.2f}, {gh[2]:+.2f}] (n={gh[4]})' if gl and gh else f'H12 {m} insufficient')
    h13 = {m: boot_diff(S[('H', m)], S[('D', m)], TEST) for m in MODELS}
    adj = holm([h13[m][3] for m in MODELS])
    for m, a in zip(MODELS, adj):
        d = h13[m]; print(f'H13 H - D   {m:8s} {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}] Holm p={a:.4f}')
    spH = [summary(S[('H', m)], TEST)[0][0] for m in MODELS]; spD = [summary(S[('D', m)], TEST)[0][0] for m in MODELS]
    print(f'H14 range across models: H {max(spH)-min(spH):.2f}  D {max(spD)-min(spD):.2f}')
    # H15 verifier accuracy
    for m in MODELS:
        recs = {}
        for l in open(R / f'hyb_{m}.jsonl'):
            r = json.loads(l); recs[(r['did'], r['k'])] = r
        y, p = [], []
        for d in TEST:
            c = HV.candidates(d); P = np.array([(x[0], x[1]) for x in c]); G = np.array(ANN[d]['points'], float)
            D = np.linalg.norm(P[:, None] - G[None], axis=2); ri, ci = linear_sum_assignment(D)
            pos = {int(i) for i, j in zip(ri, ci) if D[i, j] <= M.taus(ANN[d])[0]}
            for k in range(len(c)):
                y.append(k in pos); p.append(recs.get((d, k), {}).get('match') is True)
        y, p = np.array(y), np.array(p)
        hf = summary(S[('H', m)], TEST)[0][0]
        print(f'H15 {m:8s} verifier acc {(y==p).mean():.3f} TPR {p[y].mean():.3f} FPR {p[~y].mean():.3f} | H F1 / bound = {hf/np.mean(list(bound.values())):.2f}')
    for lab, tm in (('dev-thr', tm_dev), ('LOO', tm_loo)):
        for m in MODELS:
            d = boot_diff(S[('H', m)], tm, TEST); print(f'H16 H - TM {lab:7s} {m:8s} {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]')
