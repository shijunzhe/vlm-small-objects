#!/usr/bin/env python3
"""All numbers for paper 2, computed from stored outputs. Writes ../paper/generated/{numbers.json,macros.tex}.
Primary metric: strict F1 (one-to-one matching within tau_sym = 0.5*sqrt(template area)), mean over drawings,
runs averaged per drawing first. Paired bootstrap over the 34 test drawings (4000 resamples)."""
import contextlib, io, json, math, re
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
import mech as M, ceilings as K
import analyze_test as A
with contextlib.redirect_stdout(io.StringIO()):
    import analyze_gen as G

OUT = Path('../paper/generated'); OUT.mkdir(parents=True, exist_ok=True)
T, SH, VW = K.TEST, G.SHEETS, G.VIEWS
N = {}
S = dict(G.S)
# generation 1 extra cells and native-coordinate decodes
for k in ('gpt', 'claude', 'gemini', 'gemma4', 'qwen3vl'):
    name = {'gpt': 'gpt54', 'claude': 'sonnet46', 'gemini': 'gemini25', 'gemma4': 'gemma4', 'qwen3vl': 'qwen3vl'}[k]
    S[('A1', name)] = A.score_whole(k, 'px')
    S[('A2', name)] = A.score_whole(k, {'claude': 'px_claude', 'gpt': 'px'}.get(k, 'n1000'))
    S[('C', name)] = S[('D', name)] if k in ('gpt', 'claude', 'qwen3vl') else A.score_tiles('C', k)
S[('B', 'gpt54')] = A.score_B('gpt'); S[('B', 'gemini25')] = A.score_B('gemini')
S[('Dmask', 'gpt54')] = A.score_tiles('Dmask', 'gpt'); S[('Dmask', 'sonnet46')] = A.score_tiles('Dmask', 'claude')
for v in ('gem38', 'gem38uh'):
    S[('A2', v)] = G.at(G.GEN, A.score_whole, v, 'n1000')
for v in ('sonnet55', 'opus55', 'gpt55', 'gpt56'):
    S[('A2', v)] = S[('A1', v)]
for v in ('sonnet55', 'gpt56', 'gem38'):          # D now has run 0 and (if present) run 1
    S[('D', v)] = G.at(G.GEN, A.score_tiles, 'D', v)
tm_dev, tm_loo, thr = A.tm_scores(); S[('TM', 'dev')] = tm_dev; S[('TM', 'loo')] = tm_loo


def f1(per, ds=T, k=0):
    mp = A.mean_per(per); v = [mp[d][k] for d in ds if d in mp]
    return float(np.nanmean(v)) if v else float('nan')


def diff(a, b, ds=T, k=0):
    r = A.boot_diff(a, b, ds, k)
    return {'d': float(r[0]), 'lo': float(r[1]), 'hi': float(r[2]), 'p': float(r[3]), 'n': int(r[4])}


GEN1 = ['gpt54', 'sonnet46', 'gemini25', 'gemma4', 'qwen3vl']
GEN2 = ['gpt56', 'sonnet55', 'gem38']
cells = {}
for (c, v), per in S.items():
    cells[f'{c}|{v}'] = {'all': f1(per), 'sheets': f1(per, SH), 'views': f1(per, VW), 'recall': f1(per, T, 1),
                         'precision': f1(per, T, 2), 'relerr': f1(per, T, 3),
                         'runs': max((len(x) for x in per.values()), default=0)}
N['cells'] = cells; N['tm_threshold'] = thr

# run-to-run stability of D (all models with two runs)
stab = {}
for v in GEN1 + GEN2:
    per = S[('D', v)]
    if all(len(x) >= 2 for x in per.values()):
        r0 = np.mean([x[0][0] for x in per.values()]); r1 = np.mean([x[1][0] for x in per.values()])
        stab[v] = {'run0': float(r0), 'run1': float(r1)}
N['d_stability'] = stab

# key paired comparisons
cmp = {}
for v in GEN1:
    cmp[f'D-A2|{v}'] = diff(S[('D', v)], S[('A2', v)]); cmp[f'D-A1|{v}'] = diff(S[('D', v)], S[('A1', v)])
    cmp[f'H-D|{v}'] = diff(S[('H', v)], S[('D', v)]); cmp[f'H-TMdev|{v}'] = diff(S[('H', v)], tm_dev)
    cmp[f'D-TMdev|{v}'] = diff(S[('D', v)], tm_dev)
for v in GEN2:
    cmp[f'D-A2sheets|{v}'] = diff(S[('D', v)], S[('A2', v)], SH)
    for lab, tm in (('TMdev', tm_dev), ('TMloo', tm_loo)):
        cmp[f'D-{lab}|{v}'] = diff(S[('D', v)], tm); cmp[f'H-{lab}|{v}'] = diff(S[('H', v)], tm)
cmp['gen|claude_A2'] = diff(S[('A2', 'sonnet55')], S[('A2', 'sonnet46')]); cmp['gen|claude_A2_sheets'] = diff(S[('A2', 'sonnet55')], S[('A2', 'sonnet46')], SH)
cmp['gen|gpt56_A1'] = diff(S[('A1', 'gpt56')], S[('A1', 'gpt54')]); cmp['gen|gpt55_A1'] = diff(S[('A1', 'gpt55')], S[('A1', 'gpt54')])
cmp['gen|gemini_A2'] = diff(S[('A2', 'gem38')], S[('A2', 'gemini25')]); cmp['gen|gemini_A2_sheets'] = diff(S[('A2', 'gem38')], S[('A2', 'gemini25')], SH)
cmp['budget|gem38uh_sheets'] = diff(S[('A2', 'gem38uh')], S[('A2', 'gem38')], SH); cmp['budget|gem38uh_views'] = diff(S[('A2', 'gem38uh')], S[('A2', 'gem38')], VW)
cmp['cap|native-auto_sheets'] = diff(S[('A1nat', 'gpt56orig')], S[('A1', 'gpt56')], SH)
cmp['cap|tiles-native_sheets'] = diff(S[('D', 'gpt56')], S[('A1nat', 'gpt56orig')], SH)
cmp['B-D|gpt54'] = diff(S[('B', 'gpt54')], S[('D', 'gpt54')]); cmp['B-D|gemini25'] = diff(S[('B', 'gemini25')], S[('D', 'gemini25')])
cmp['D-C|gemini25'] = diff(S[('D', 'gemini25')], S[('C', 'gemini25')]); cmp['D-C|gemma4'] = diff(S[('D', 'gemma4')], S[('C', 'gemma4')])
cmp['Dmask-D|gpt54'] = diff(S[('Dmask', 'gpt54')], S[('D', 'gpt54')]); cmp['Dmask-D|sonnet46'] = diff(S[('Dmask', 'sonnet46')], S[('D', 'sonnet46')])
N['cmp'] = cmp

# dose-response and ceiling-group gains
CE = A.CEIL; GC = G.GC
def ceil_of(v, d):
    m1 = {'gpt54': 'gpt', 'sonnet46': 'claude', 'gemini25': 'gemini', 'gemma4': 'gemma4', 'qwen3vl': 'qwen3vl'}
    return CE[f'{d}|A1|{m1[v]}']['ceiling_f1'] if v in m1 else GC[f'{d}|{v}']['ceiling_f1']
dose = {}
for v in GEN1 + ['sonnet55', 'opus55', 'gpt55', 'gpt56', 'gem38', 'gem38uh']:
    mp = A.mean_per(S[('A2', v)]); ds = [d for d in T if d in mp]
    x = np.array([ceil_of(v, d) for d in ds]); y = np.array([mp[d][0] for d in ds])
    rng = np.random.default_rng(0)
    bs = [spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(ds), len(ds)) for _ in range(2000))]
    dose[v] = {'rho': float(spearmanr(x, y)[0]), 'lo': float(np.nanpercentile(bs, 2.5)), 'hi': float(np.nanpercentile(bs, 97.5)),
               'ceil_mean': float(x.mean()), 'obs_mean': float(y.mean()), 'points': [[float(a), float(b)] for a, b in zip(x, y)]}
N['dose'] = dose
groups = {}
for v in ('gpt54', 'sonnet46'):
    lo = [d for d in T if ceil_of(v, d) < 0.5]; hi = [d for d in T if ceil_of(v, d) >= 0.9]
    groups[v] = {'low': diff(S[('D', v)], S[('A2', v)], lo), 'high': diff(S[('D', v)], S[('A2', v)], hi)}
N['groups'] = groups

# ceiling check counts (generation 1 and 2), strong quantized observer
def ceiling_viol(pairs):
    tot = viol = 0
    for (c, v), ckey in pairs:
        mp = A.mean_per(S[(c, v)])
        for ds in (SH, VW):
            ds = [d for d in ds if d in mp]
            o = np.mean([mp[d][0] for d in ds]); cl = np.mean([ckey(d) for d in ds]); tot += 1; viol += o > cl + 0.05
    return int(viol), int(tot)
m1 = {'gpt54': 'gpt', 'sonnet46': 'claude', 'gemini25': 'gemini', 'gemma4': 'gemma4', 'qwen3vl': 'qwen3vl'}
p1 = []
for v in GEN1:
    for c, cc in (('A2', 'A1'), ('C', 'C'), ('D', 'D')):
        p1.append(((c, v), lambda d, v=v, cc=cc: CE[f'{d}|{cc}|{m1[v]}']['ceiling_f1']))
p1 += [(('B', 'gpt54'), lambda d: CE[f'{d}|B|gpt']['ceiling_f1']), (('B', 'gemini25'), lambda d: CE[f'{d}|B|gemini']['ceiling_f1'])]
p2 = [(('A2', v), lambda d, v=v: GC[f'{d}|{v}']['ceiling_f1']) for v in ('sonnet55', 'opus55', 'gpt55', 'gpt56', 'gem38', 'gem38uh')]
p2 += [(('A1nat', 'gpt56orig'), lambda d: GC[f'{d}|gpt56orig']['ceiling_f1'])] + [(('D', v), lambda d, v=v: GC[f'{d}|D_{v}']['ceiling_f1']) for v in GEN2]
N['ceiling_viol'] = {'gen1': ceiling_viol(p1), 'gen2': ceiling_viol(p2)}

# verifier accuracy per model (propose-and-verify)
ver = {}
import hybrid_verify as HV
cands = {d: HV.candidates(d) for d in T}
pos = {}
for d in T:
    P = np.array([(x[0], x[1]) for x in cands[d]]); Gt = np.array(A.ANN[d]['points'], float); tau = M.taus(A.ANN[d])[0]
    D = np.linalg.norm(P[:, None] - Gt[None], axis=2); ri, ci = linear_sum_assignment(np.where(D <= tau, D, 1e6))
    pos[d] = {int(i) for i, j in zip(ri, ci) if D[i, j] <= tau}
bound = np.mean([2 * (len(pos[d]) / A.ANN[d]['count']) / (1 + len(pos[d]) / A.ANN[d]['count']) for d in T])
for v, f in [(v, A.R / f'hyb_{k}.jsonl') for v, k in zip(GEN1, ['gpt', 'claude', 'gemini', 'gemma4', 'qwen3vl'])] + [(v, G.GEN / f'hyb_{v}.jsonl') for v in GEN2]:
    recs = {}
    for l in open(f):
        r = json.loads(l); recs[(r['did'], r['k'])] = r
    y = np.array([k in pos[d] for d in T for k in range(len(cands[d]))]); p = np.array([recs[(d, k)].get('match') is True for d in T for k in range(len(cands[d]))])
    ver[v] = {'tpr': float(p[y].mean()), 'fpr': float(p[~y].mean()), 'acc': float((p == y).mean())}
N['verifier'] = ver; N['hybrid_bound'] = float(bound); N['n_candidates'] = int(sum(len(c) for c in cands.values())); N['n_true_candidates'] = int(sum(len(p) for p in pos.values()))

# error taxonomy for D and H (pooled over drawings, run 0): misses, duplicate reports, displaced reports, spurious detections
def taxonomy(points_by_did):
    out = np.zeros(4); gt_total = 0
    for d, pts in points_by_did.items():
        a = A.ANN[d]; tau = M.taus(a)[0]; Gt = np.array(a['points'], float); gt_total += len(Gt)
        if not pts:
            out[0] += len(Gt); continue
        P = np.array(pts, float); D = np.linalg.norm(P[:, None] - Gt[None], axis=2)
        ri, ci = linear_sum_assignment(np.where(D <= tau, D, 1e6)); ok = D[ri, ci] <= tau
        matched_p = set(ri[ok]); matched_g = set(ci[ok]); out[0] += len(Gt) - len(matched_g)
        for i in range(len(P)):
            if i in matched_p: continue
            j = int(np.argmin(D[i])); near = D[i, j] <= 2 * tau
            if near and j in matched_g: out[1] += 1
            elif near: out[2] += 1
            else: out[3] += 1
    return {'miss': out[0] / gt_total, 'duplicate': out[1] / gt_total, 'displaced': out[2] / gt_total, 'spurious': out[3] / gt_total}
def pts_tiles(path):
    last = {}
    for l in open(path):
        r = json.loads(l); last[(r['did'], r['i'], r['j'], r.get('half', ''))] = r
    out = {d: [] for d in T}
    for r in last.values():
        if r['status'] == 'success': out[r['did']] += M.to_native(r)
    return out
def pts_hyb(path):
    out = {d: [] for d in T}
    for l in open(path):
        r = json.loads(l)
        if r.get('match') is True: out[r['did']].append((r['x'], r['y']))
    return out
tax = {}
for v, k in zip(GEN1, ['gpt', 'claude', 'gemini', 'gemma4', 'qwen3vl']):
    tax[f'D|{v}'] = taxonomy(pts_tiles(A.R / f'D_{k}_run0.jsonl')); tax[f'H|{v}'] = taxonomy(pts_hyb(A.R / f'hyb_{k}.jsonl'))
for v in GEN2:
    tax[f'D|{v}'] = taxonomy(pts_tiles(G.GEN / f'D_{v}_run0.jsonl')); tax[f'H|{v}'] = taxonomy(pts_hyb(G.GEN / f'hyb_{v}.jsonl'))
def pts_native():
    out = {d: [] for d in T}
    for l in open(G.GEN / 'A1nat_gpt56orig_run0.jsonl'):
        r = json.loads(l)
        if r['status'] == 'success':
            s_ = r['scale']; out[r['did']] = [(q['cx'] / s_, q['cy'] / s_) for q in r['shapes'] if isinstance(q, dict) and A.num(q.get('cx')) and A.num(q.get('cy'))]
    return out
tax['A1nat|gpt56orig'] = taxonomy(pts_native())
N['taxonomy'] = tax

# tolerance sensitivity: ink bounding box of the template, and 2x tau_sym
def ink_tau(a):
    t = np.array(Image.open(M.BENCH / a['template']).convert('L')); ys, xs = np.where(t < 200)
    return 0.5 * math.sqrt((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)), 30 / min(1, 1600 / max(a['width'], a['height']))
orig_taus = M.taus; sens = {}
for lab, fn in (('ink', ink_tau), ('double', lambda a: (2 * orig_taus(a)[0], orig_taus(a)[1]))):
    M.taus = fn
    vals = {}
    for key in (('A2', 'gpt54'), ('A2', 'sonnet46'), ('D', 'gpt54'), ('D', 'gemma4'), ('H', 'qwen3vl'), ('D', 'sonnet55'), ('D', 'gem38'), ('A2', 'sonnet55')):
        c, v = key
        if v in GEN2 or v in ('opus55',):
            per = G.at(G.GEN, A.score_tiles, 'D', v) if c == 'D' else G.at(G.GEN, A.score_whole, v, 'px')
        elif c == 'A2':
            per = A.score_whole({'gpt54': 'gpt', 'sonnet46': 'claude'}[v], {'gpt54': 'px', 'sonnet46': 'px_claude'}[v])
        elif c == 'D':
            per = A.score_tiles('D', {'gpt54': 'gpt', 'gemma4': 'gemma4'}[v])
        else:
            per = A.score_H('qwen3vl')
        vals[f'{c}|{v}'] = f1(per)
    tdev, tloo, _ = A.tm_scores(); vals['TM|dev'] = f1(tdev)
    sens[lab] = vals
M.taus = orig_taus
N['tolerance_sensitivity'] = sens

# costs per drawing (calls, input tokens)
def usage(path):
    n = t = 0
    for l in open(path):
        r = json.loads(l); u = r.get('usage') or {}; n += 1; t += (u.get('input_tokens') or r.get('input_tokens') or 0)
    return n, t
cost = {}
for lab, path in [('A1|gpt54', A.R / 'A1_gpt_run0.jsonl'), ('D|gpt54', A.R / 'D_gpt_run0.jsonl'), ('D|sonnet46', A.R / 'D_claude_run0.jsonl'),
                  ('D|gemma4', A.R / 'D_gemma4_run0.jsonl'), ('D|gemini25', A.R / 'D_gemini_run0.jsonl'), ('H|gpt54', A.R / 'hyb_gpt.jsonl'),
                  ('A1|sonnet55', G.GEN / 'A1_sonnet55_run0.jsonl'), ('A1nat|gpt56orig', G.GEN / 'A1nat_gpt56orig_run0.jsonl'),
                  ('D|sonnet55', G.GEN / 'D_sonnet55_run0.jsonl'), ('D|gpt56', G.GEN / 'D_gpt56_run0.jsonl'), ('D|gem38', G.GEN / 'D_gem38_run0.jsonl'),
                  ('H|sonnet55', G.GEN / 'hyb_sonnet55.jsonl')]:
    n, t = usage(path); cost[lab] = {'calls': n / 34, 'tokens': t / 34}
N['cost'] = cost

json.dump(N, open(OUT / 'numbers.json', 'w'), indent=1)

# ---------- LaTeX macros ----------
def nm(s):
    return ''.join(ch for ch in s.replace('5', 'Five').replace('4', 'Four').replace('3', 'Three').replace('2', 'Two')
                   .replace('6', 'Six').replace('8', 'Eight').replace('1', 'One').replace('0', 'Zero').replace('9', 'Nine').replace('7', 'Seven') if ch.isalpha())
lines = ['% auto-generated by stok/paper2_numbers.py; do not edit']
def mac(name, val):
    val = re.sub(r'(?<![$\w])-(?=\d)', '$-$', str(val))   # typeset minus signs
    lines.append(f'\\newcommand{{\\{name}}}{{{val}}}')
for key, c in cells.items():
    cc, v = key.split('|')
    for part in ('all', 'sheets', 'views'):
        if not math.isnan(c[part]):
            mac(f'F{nm(cc)}{nm(v)}{part.capitalize()}', f'{c[part]:.2f}')
for key, d in cmp.items():
    k = nm(key.replace('|', '').replace('-', 'Minus').replace('_', ''))
    mac(f'C{k}', f'{d["d"]:+.2f}'); mac(f'C{k}CI', f'[{d["lo"]:+.2f}, {d["hi"]:+.2f}]')
for v, d in dose.items():
    mac(f'Rho{nm(v)}', f'{d["rho"]:.2f}'); mac(f'Rho{nm(v)}CI', f'[{d["lo"]:.2f}, {d["hi"]:.2f}]')
for v, g in groups.items():
    mac(f'GainLow{nm(v)}', f'{g["low"]["d"]:+.2f}'); mac(f'GainHigh{nm(v)}', f'{g["high"]["d"]:+.2f}')
    mac(f'GainHigh{nm(v)}CI', f'[{g["high"]["lo"]:+.2f}, {g["high"]["hi"]:+.2f}]')
for v, x in ver.items():
    mac(f'VerTPR{nm(v)}', f'{x["tpr"]:.2f}'); mac(f'VerFPR{nm(v)}', f'{x["fpr"]:.2f}')
mac('HybBound', f'{bound:.2f}'); mac('TMThr', f'{thr:.2f}')
mac('CeilViolGenOne', f'{N["ceiling_viol"]["gen1"][0]}'); mac('CeilTotGenOne', f'{N["ceiling_viol"]["gen1"][1]}')
mac('CeilViolGenTwo', f'{N["ceiling_viol"]["gen2"][0]}'); mac('CeilTotGenTwo', f'{N["ceiling_viol"]["gen2"][1]}')
(OUT / 'macros.tex').write_text('\n'.join(lines) + '\n')
print('wrote', OUT / 'numbers.json', 'and macros.tex with', len(lines) - 1, 'macros')

# ---------- generated appendix tables ----------
LAB = {'gpt54': 'GPT-5.4', 'sonnet46': 'Claude Sonnet 4.6', 'gemini25': 'Gemini 2.5 Flash', 'gemma4': 'Gemma 4', 'qwen3vl': 'Qwen3-VL',
       'gpt56': 'GPT-5.6', 'sonnet55': 'Claude Sonnet 5.5', 'gem38': 'Gemini 3.8 Flash', 'gpt56orig': 'GPT-5.6 (original)', 'dev': 'tuning-set threshold'}
COND = {'A2': 'whole drawing', 'D': 'tiles', 'H': 'propose + verify', 'TM': 'template matching', 'A1nat': 'whole, native', 'A1': 'whole drawing'}
rows = []
for key in sens['ink']:
    c, v = key.split('|')
    base = cells.get(key, {}).get('all', float('nan')) if c != 'TM' else cells['TM|dev']['all']
    rows.append(f"{LAB.get(v, v)}, {COND.get(c, c)} & {base:.2f} & {sens['ink'][key]:.2f} & {sens['double'][key]:.2f} \\\\")
(OUT / 'tab_sens.tex').write_text('\n'.join(rows) + '\n')
rows = []
for key, t in tax.items():
    if t is None: continue
    c, v = key.split('|')
    rows.append(f"{LAB.get(v, v)} & {COND[c]} & {t['miss']:.2f} & {t['duplicate']:.2f} & {t['displaced']:.2f} & {t['spurious']:.2f} \\\\")
(OUT / 'tab_tax.tex').write_text('\n'.join(rows) + '\n')
rows = []
for key, x in cost.items():
    c, v = key.split('|')
    rows.append(f"{LAB.get(v, v)} & {COND.get(c, c)} & {x['calls']:.1f} & {x['tokens']/1000:.1f}k \\\\")
(OUT / 'tab_cost.tex').write_text('\n'.join(rows) + '\n')
extra = []
for v, st in stab.items():
    extra.append(f'\\newcommand{{\\DRunZero{nm(v)}}}{{{st["run0"]:.2f}}}'); extra.append(f'\\newcommand{{\\DRunOne{nm(v)}}}{{{st["run1"]:.2f}}}')
with open(OUT / 'macros.tex', 'a') as f:
    f.write('\n'.join(extra) + '\n')
print('appendix tables written')
