#!/usr/bin/env python3
"""
Zero-cost analyses for 'what is the limit of using a VLM directly, and how useful is the theory':
 A. FSC-147 (300 test images, class-name counting, E4 outputs already collected): does counting error and
    localization follow the symbol-size-in-tokens S_tok derived from documented geometry, outside drawings?
 B. Single-call feasibility rule: a call with N tokens across the image side resolves targets whose size
    is >= S*/N of the side. Fraction of REDP-X40 / FSC-147 images that pass, per model.
 C. Hybrid upper bound on REDP-X40: template-matching candidates at a low threshold, verified by a VLM on
    crops (where S_tok is large). Max F1 with a perfect verifier and the number of crops to verify.
"""
import json, math, ast, sys
import numpy as np
sys.path.insert(0, '.')
import mech as M, ceilings as K

E4 = 'vlm_api/results/e4/fsc147'
man = json.load(open('vlm_api/data_e4/fsc147/manifest.json'))
items = {it['item_id']: it for it in man['items']}
GRID = {'gemini': 16, 'gemma4': 16.7}
PATCH = {'gpt': 32, 'claude': 28, 'qwen3vl': 32}


def lit(v):
    return ast.literal_eval(v) if isinstance(v, str) else v


def tok_side(model, w, h):
    if model in GRID:
        return max(w, h) / GRID[model]
    if model == 'claude':
        rw, rh = M.claude_resized(w, h); return 28 * w / rw
    return PATCH[model]


print('=== A. FSC-147: error vs symbol size in tokens (both output orders pooled, run 0)')
bins = [(0, 0.5), (0.5, 1), (1, 2), (2, 99)]
reg = {}
for model in ('gpt', 'claude', 'gemini', 'gemma4', 'qwen3vl'):
    rows = []
    for l in open(f'{E4}/{model}_run0.jsonl'):
        r = json.loads(l)
        if r['status'] != 'success':
            continue
        it = items[r['item_id']]; w, h = lit(r['canvas']); s = float(r['scale'])
        side = float(it['exemplar_side']) * s; t = tok_side(model, w, h); stok = side / t
        g = int(r['gt_count']); lc = int(r['list_count']); dc = r['declared_count']
        dc = int(dc) if str(dc).lstrip('-').isdigit() else lc
        f1 = ceil = None
        if model in ('gpt', 'claude'):
            pts = [(p['x'] / s, p['y'] / s) for p in lit(r['points']) if isinstance(p, dict) and isinstance(p.get('x'), (int, float)) and isinstance(p.get('y'), (int, float))]
            gt = [(p['x'], p['y']) for p in lit(it['gt_points'])]
            tau = 0.5 * float(it['exemplar_side'])
            tp, n_p, n_g = M.match(pts, gt, tau); f1 = 2 * tp / (n_p + n_g)
            ceil = K.ideal_observer(gt, t / s, tau, n_off=16)[0]
        rows.append((stok, abs(lc - g) / g, abs(dc - g) / g, f1, ceil, g))
    R = np.array([[x if x is not None else np.nan for x in r] for r in rows])
    line = f'{model:8s} n={len(R):3d} median S_tok {np.median(R[:,0]):.2f} | list |relerr| by S_tok bin: '
    line += '  '.join(f'[{a},{b}) {np.mean(R[(R[:,0]>=a)&(R[:,0]<b),1]):.2f} (n={((R[:,0]>=a)&(R[:,0]<b)).sum()})' for a, b in bins)
    print(line)
    if model in ('gpt', 'claude'):
        print('         loc F1 by bin:   ' + '  '.join(f'[{a},{b}) {np.nanmean(R[(R[:,0]>=a)&(R[:,0]<b),3]):.2f} / ceil {np.nanmean(R[(R[:,0]>=a)&(R[:,0]<b),4]):.2f}' for a, b in bins))
        print(f'         loc F1 overall {np.nanmean(R[:,3]):.2f}  ceiling {np.nanmean(R[:,4]):.2f}  above-ceiling images {(R[:,3] > R[:,4] + 0.05).mean():.0%}')
    # regression: log list error on log S_tok controlling for log count
    X = np.c_[np.ones(len(R)), np.log(R[:, 0]), np.log(R[:, 5])]; y = np.log(R[:, 1] + 0.02)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    boot = []
    rng = np.random.default_rng(0)
    for _ in range(1000):
        i = rng.integers(0, len(R), len(R)); boot.append(np.linalg.lstsq(X[i], y[i], rcond=None)[0][1])
    print(f'         regression log(err) ~ log(S_tok) + log(count): slope S_tok {b[1]:+.2f} [{np.percentile(boot,2.5):+.2f}, {np.percentile(boot,97.5):+.2f}], slope count {b[2]:+.2f}')
    reg[model] = b.tolist()

print('\n=== B. single-call feasibility: target size / image side >= S*/N  (S* = 1.5 tokens)')
N_ACROSS = {'gpt (auto)': 50, 'gpt (original)': 100, 'claude': 39, 'gemini': 16, 'gemma4': 16.7}
redp = [json.loads((M.BENCH / 'annotations' / f'd{i:03d}.json').read_text()) for i in range(1, 41)]
r_redp = np.array([min(a['template_size']) / math.sqrt(a['width'] * a['height']) for a in redp])
r_fsc = []
for it in man['items']:
    from PIL import Image
    w, h = Image.open('vlm_api/data_e4/fsc147/' + it['file']).size
    r_fsc.append(float(it['exemplar_side']) / math.sqrt(w * h))
r_fsc = np.array(r_fsc)
print(f'relative target size (side / sqrt(image area)): REDP median {np.median(r_redp)*100:.2f}% [range {r_redp.min()*100:.2f}-{r_redp.max()*100:.2f}], FSC median {np.median(r_fsc)*100:.1f}%')
for k, N in N_ACROSS.items():
    thr = 1.5 / N
    print(f'  {k:15s} N={N:5.1f} tokens across -> needs >= {thr*100:4.1f}% : REDP {np.mean(r_redp >= thr):4.0%} of drawings, FSC {np.mean(r_fsc >= thr):4.0%} of images'
          f' | zoom needed on REDP (median) x{np.median(np.maximum(1, thr / r_redp)):.1f}')

print('\n=== C. hybrid bound: template-matching candidates + VLM verifier on crops (REDP-X40, all 40)')
for floor in (0.5, 0.6, 0.7):
    R, NC, F1tm = [], [], []
    for a in redp:
        did = a['image'].split('/')[-1][:4]
        c = json.load(open(f'../data/redp40/results/tm_candidates/{did}.json'))['candidates']
        pts = [(x, y) for x, y, sc, v in c if sc >= floor]
        tau, _ = M.taus(a)
        tp, n_p, n_g = M.match(pts, a['points'], tau)
        R.append(tp / n_g); NC.append(n_p)
    R = np.array(R)
    print(f'  TM score >= {floor}: candidate recall {R.mean():.3f} (min {R.min():.2f}), perfect-verifier F1 bound {np.mean(2*R/(1+R)):.3f}, '
          f'crops to verify per drawing median {np.median(NC):.0f} (max {max(NC)}), candidates per true instance {np.sum(NC)/sum(a["count"] for a in redp):.1f}')
