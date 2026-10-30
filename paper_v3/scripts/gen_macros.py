#!/usr/bin/env python3
"""Generate the macros of version 3 of the paper.

Writes into paper/generated/:
  macros_x.tex, macros_r1.tex  copies of the unified-reader macros (format-aware reading everywhere, POINT_XY=1),
                               produced by reproduce.sh in a copy of the release whose mech.py sets FORMAT_AWARE = True;
  macros_main.tex              numbers new in version 3: Table 2 (tab1.json), Figure 3 (fig3_cells.json),
                               the overall scale and the load sweep (recorded_results.json), Figure 1, the budget sweep
                               at 8 px, packing and stop statistics, natural images without truncated answers,
                               the newest-generation S90 values and the rule-of-thumb geometry.
Run:  cd <unified copy>/experiments && POINT_XY=1 python3 <paper>/scripts/gen_macros.py [visual-spec folder, default experiments/vspec]"""
import glob, json, math, os, shutil, sys, collections, contextlib, io
from pathlib import Path
import numpy as np

UNI = Path.cwd().parent
assert (UNI / 'experiments' / 'mech.py').exists(), 'run from the unified copy experiments/'
sys.path.insert(0, os.getcwd())
PAPER = Path(__file__).resolve().parent.parent; G = PAPER / 'generated'
V = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / 'vspec'
shutil.copy(UNI / 'paper/generated/macros_x.tex', G / 'macros_x.tex')
shutil.copy(UNI / 'paper/generated/macros.tex', G / 'macros_r1.tex')
MAC = {}
def put(k, v): assert k not in MAC, k; MAC[k] = v
def f2(x, sign=False): s = f'{x:+.2f}' if sign else f'{x:.2f}'; return s.replace('-', '$-$')
def thou(x): return f'{x:,}'.replace(',', '{,}')
def rnd(x, d): return thou(int(round(x, -d)))
def last(f, key):
    d = {}
    for l in open(f):
        r = json.loads(l)
        if r['status'] in ('success', 'parse_error'): d[key(r)] = r
    return d

# ---------------- overall scale and load sweep (recorded; see recorded_results.json) ----------------
REC = json.load(open(G / 'recorded_results.json')); sc_ = REC['scale']
put('CscaleCalls', rnd(sc_['release_calls'] + sc_['vspec_billed_calls'] + sc_['gem4_table2_calls'], 3))
put('CscaleImages', thou(sc_['images'])); put('CscaleInst', rnd(sc_['instances'], 3))
put('CscaleModels', str(sc_['models'])); put('CscaleCtrl', thou(sc_['controlled_images']))
put('CgenGptSixOneMax', str(REC['load_sweep_S15']['gpt61_max_targets_recall_ge_0.9']))

# ---------------- Figure 1 and the example sheets ----------------
sys.path.insert(0, str(UNI / 'tools')); import safe_tiles as ST
B = UNI / 'data/redp40/redp_x40'
a = json.load(open(B / 'annotations/d018.json')); w, h = a['template_size']; m = math.sqrt(w * h)
P = ST.plan(a['width'], a['height'], w, h, interface='res', token=32, budget=2500, S_star=1.5)
put('CfigSwhole', f2(P['S_whole'])); put('CfigViews', str(P['n_views'])); put('CfigTokens', rnd(P['tokens'], 3))
put('CfigCmin', rnd(P['tokens_lower_bound'], 3)); put('CfigTargets', str(len(a['points'])))
put('CfigPropB', rnd(a['width'] * a['height'] / m ** 2 * 1.5 ** 2, 3))
put('CfigW', thou(a['width'])); put('CfigH', thou(a['height'])); put('CfigM', f'{m:.0f}')
put('CfigMsent', f'{P["S_whole"] * 32:.0f}'); put('CfigPctToken', f'{100 * P["S_whole"]:.0f}')
a2 = json.load(open(B / 'annotations/d025.json')); w2, h2 = a2['template_size']; m2 = math.sqrt(w2 * h2)
put('CsmallW', thou(a2['width'])); put('CsmallH', thou(a2['height'])); put('CsmallM', f'{m2:.0f}')
put('CsmallSw', f2(m2 * (1600 / max(a2['width'], a2['height'])) / 32)); put('CsmallSt', f2(44 * m2 / (min(w2, h2) * 32)))
with contextlib.redirect_stdout(io.StringIO()):
    import mech as M, analyze_test as A
ra = last('results_test/A1_gpt_run0.jsonl', lambda r: r['did'])['d025']; s = ra['scale']
recA = A.metrics('d025', [(p[0] / s, p[1] / s) for p in (M.point_of(q) for q in ra.get('shapes') or []) if p])[1]
D = [r for k, r in last('results_test/D_gpt_run0.jsonl', lambda r: (r['did'], r['cell'], r['i'], r['j'], r.get('half', ''))).items() if k[0] == 'd025']
recD = A.metrics('d025', sum((M.to_native(r) for r in D if r['status'] == 'success'), []))[1]
put('CsmallRecWhole', f2(recA)); put('CsmallRecTile', f2(recD))

# ---------------- budget intervention at 8 px (Gemini 3.8, three budgets, same pixels) ----------------
with contextlib.redirect_stdout(io.StringIO()):
    import syn_analysis as SA
rows = [r for r in SA.per_image() if r['exp'] == 'res' and r['v'] in ('gem38lo', 'gem38', 'gem38uh')]
for v, nmv in (('gem38lo', 'Lo'), ('gem38', 'Def'), ('gem38uh', 'Uh')):
    put('CbudEight' + nmv, f2(np.mean([r['recall'] for r in rows if r['v'] == v and r['m'] == 8])))

# ---------------- packing: targets per packed GPT-5.6 call; stop reasons of the content sweep ----------------
per = []
for r in last('results_x/mosaic/gpt56_k8_run0.jsonl', lambda r: (r['did'], r['g'])).values():
    pts = json.load(open(B / f'annotations/{r["did"]}.json'))['points']; n = 0
    for c in r['cells']:
        _, _, (ax, ay, cx, cy), (ox, oy), z, _ = c
        n += sum(1 for x, y in pts if ox + ax <= x * z < ox + cx and oy + ay <= y * z < oy + cy)
    per.append(n)
put('CpackMedian', f'{np.median(per):.0f}'); put('CpackMean', f'{np.mean(per):.1f}'); put('CpackMax', str(max(per)))
items = {it['id']: it for it in json.load(open('syn/items.json'))}; stops = collections.Counter(); outmax = 0
for f in glob.glob('syn/results/*.jsonl'):
    for l in open(f):
        r = json.loads(l); it = items.get(r.get('id'))
        if not it or it['exp'] != 'cap': continue
        u = r.get('usage') or {}; sr = str(u.get('stop_reason', '')).lower()
        stops['max' in sr or 'length' in sr] += 1; outmax = max(outmax, u.get('output_tokens') or 0)
put('CcapCalls', str(sum(stops.values()))); put('CcapNormal', str(stops[False])); put('CcapMaxOut', thou(outmax))

# ---------------- newest generation (visual-spec experiment, vendor-default reasoning) ----------------
put('CgenSninetyGemThreeEight', f2(json.load(open(V / 'frozen_s90.json'))['gem38']['S_ref']))
put('CgenSninetyGemFour', f2(json.load(open(V / 'frozen_s90_gem4.json'))['S90']))

# ---------------- natural images without truncated answers ----------------
FS = UNI / 'data/fsc147'; sc = json.load(open(FS / 'x8_scores_obj.json'))
for v, nmv in (('gpt56', 'GptFiveSix'), ('sonnet55', 'SonnetFiveFive')):
    recs = {}
    for c in 'UT':
        for l in open(FS / f'results/{v}_{c}_obj.jsonl'):
            r = json.loads(l)
            if r['status'] in ('success', 'parse_error'): recs[(c, r['name'], r['v'])] = r
    names = [k.split('|')[2] for k in sc if k.startswith(v + '|W|') and sc[k]['stratum'] == 'small']
    keep = [n for n in names if f'{v}|U|{n}' in sc and f'{v}|T|{n}' in sc and not any(
        (recs.get((c, n, k)) or {}).get('status') == 'parse_error' or ((recs.get((c, n, k)) or {}).get('usage') or {}).get('stop_reason') in ('length', 'max_tokens')
        for c, nv in (('U', 1), ('T', 4)) for k in range(nv))]
    put('CfscNoTrunc' + nmv, f2(np.mean([sc[f'{v}|T|{n}']['f1'] - sc[f'{v}|U|{n}']['f1'] for n in keep])))
    put('CfscNoTruncN' + nmv, str(len(keep)))

# ---------------- Table 2 ----------------
T = json.load(open(G / 'tab1.json'))
NM = {'gpt61': 'GptSixOne', 'opus55': 'OpusFiveFive', 'gem38': 'GemThreeEight', 'gem4': 'GemFour', 'qwen3vl': 'QwenThreevl'}
DSN = {'redp': 'Ls', 'fpc': 'Fp'}
sig = lambda d: d is not None and (d[1] > 0 or d[2] < 0)
WORD = {4: 'four', 5: 'five', 6: 'six', 10: 'ten'}
for k, c in T.items():
    b, v = k.split('|'); p = 'Ct' + DSN[b] + NM[v]
    put(p + 'RecW', f2(c['recW'])); put(p + 'RecR', f2(c['recR'])); put(p + 'Rel', f'{100 * (c["recR"] - c["recW"]) / c["recW"]:.0f}')
    put(p + 'dRec', f2(c['dRec'][0], True))
    if c['dRecD']: put(p + 'dRecD', f2(c['dRecD'][0], True))
put('CtMinChange', f2(min(c['dRec'][0] for c in T.values()), True))
put('CtSigRec', WORD[sum(sig(c['dRec']) for c in T.values())])
put('CtSigPrec', WORD[sum(c['dPrec'][1] > 0 for c in T.values())])
put('CtSigFOne', WORD.get(sum(c['dF1'][1] > 0 for c in T.values()), str(sum(c['dF1'][1] > 0 for c in T.values()))))
assert not any(c['dF1'][2] < 0 for c in T.values())
assert not any(c['dRec'][2] < 0 for c in T.values()) and not any(c['dPrec'][2] < 0 for c in T.values())
put('CtTokMin', f'{min(c["tokW"] for c in T.values()):.1f}'); put('CtTokMax', f'{max(c["tokW"] for c in T.values()):.1f}')
tD = [c['tokD'] for c in T.values() if c['tokD']]; put('CtTokDMin', f2(min(tD))); put('CtTokDMax', f2(max(tD)))
put('CtCminMin', f'{min(c["cmin"] for c in T.values()):.1f}'); put('CtCminMax', f'{max(c["cmin"] for c in T.values()):.1f}')
put('CtCells', str(len(T)))
S = json.load(open('results_cs/cs_summary.json'))
put('CtOneViewGpt', str(round(S['fpc|gpt61']['summary']['frac_one_view'] * S['fpc|gpt61']['summary']['n'])))

# ---------------- Gemini 4 whole image under the pixel prompt (Supp. B), rerun 2026-10-28 ----------------
GA = json.load(open(G / 'gem4_A1_readings.json'))
put('CgemFourOver', str(GA['redp']['with_coord_over_1000'] + GA['fpc']['with_coord_over_1000']))
put('CgemFourAns', str(GA['redp']['answers'] + GA['fpc']['answers']))
for b, nb in (('redp', 'Ls'), ('fpc', 'Fp')):
    for k, nk in (('px', 'Px'), ('n1000', 'Norm'), ('mixed', 'Mix')): put('CgemFour' + nb + nk, f2(GA[b][k]))

# ---------------- Figure 3 ----------------
F3 = json.load(open(G / 'fig3_cells.json')); a_, b_ = F3['(a) Splits that never zoom out'], F3['(b) Splits that zoom out']
A_ = np.array([[np.nan if x is None else x for x in row] for row in a_['mean_gain']]); B_ = np.array([[np.nan if x is None else x for x in row] for row in b_['mean_gain']])
put('CmapWorst', f2(np.nanmin(A_), True)); put('CmapLowSMin', f2(np.nanmin(A_[:, 0]), True)); put('CmapLowSMax', f2(np.nanmax(A_[:, 0]), True))
put('CmapMidSMin', f2(np.nanmin(A_[:, 3]), True)); put('CmapMidSMax', f2(np.nanmax(A_[:, 3]), True))
put('CmapBelowMin', f2(np.nanmin(A_[:, :3]))); put('CmapBelowMax', f2(np.nanmax(A_[:, :3])))
put('CmapZoomMin', f2(-np.nanmax(B_))); put('CmapZoomMax', f2(-np.nanmin(B_)))
put('CmapZoomCells', WORD.get(int(np.sum(~np.isnan(B_))), str(int(np.sum(~np.isnan(B_))))) if int(np.sum(~np.isnan(B_))) != 6 else 'six')
put('CmapZoomSig', WORD[int(np.sum(np.array(b_['sig']) < 0))])
assert not np.any(np.array(a_['sig']) < 0) and np.nanmax(B_) < 0
R = json.load(open('results_law/law_data.json')); pair = collections.defaultdict(dict)
for r in R:
    if r['ds'] in ('redp', 'fpc_bench', 'fpc_sheets') and r['config'] in ('A1', 'D'): pair[(r['ds'], r['item'], r['model'])][r['config']] = r
    if r['ds'] == 'fsc' and r['config'] in ('U', 'T'): pair[(r['ds'], r['item'], r['model'])]['A1' if r['config'] == 'U' else 'D'] = r
fam = collections.defaultdict(list)
for k, v in pair.items():
    if 'A1' in v and 'D' in v:
        Sw, n, _ = v['A1']['calls'][0]
        if min(c[0] for c in v['D']['calls']) >= Sw * 0.999 and 1.5 <= Sw < 3: fam[k[2]].append(v['D']['obs'] - v['A1']['obs'])
band = [np.mean(fam[mm]) for mm in ('gem38', 'gemma4', 'qwen3vl', 'sonnet46', 'sonnet55')]
put('CmapBandMin', f2(min(band), True)); put('CmapBandMax', f2(max(band), True))
put('CmapGptBand', f2(max(abs(np.mean(fam[mm])) for mm in ('gpt54', 'gpt56'))))

# ---------------- unsigned copies of signed release macros, for "by X" phrasing ----------------
import re as _re
X = {**{m.group(1): m.group(2) for m in _re.finditer(r'\\newcommand\{\\(\w+)\}\{(.*)\}\s*$', open(G / 'macros_x.tex').read(), _re.M)},
     **{m.group(1): m.group(2) for m in _re.finditer(r'\\newcommand\{\\(\w+)\}\{(.*)\}\s*$', open(G / 'macros_r1.tex').read(), _re.M)}}
for src, dst in (('XcsMax', 'UcsMax'), ('XfTOnesheetssmallDL', 'UfTOnesheetssmallDL'), ('XfTOnesheetslargeDL', 'UfTOnesheetslargeDL'),
                 ('XtOneHThree', 'UtOneHThree'), ('XfscODgptFiveSixsmallTU', 'UfscODgptFiveSixsmallTU'),
                 ('XfscODsonnetFiveFivesmallTU', 'UfscODsonnetFiveFivesmallTU'), ('XnHThreegptFiveSixOVZOVC', 'UnHThreegptFiveSixOVZOVC')):
    put(dst, X[src].lstrip('+'))

# ---------------- rule-of-thumb geometry ----------------
for nmv, N in (('Gpt', 8192), ('Claude', 4784), ('Gemini', 1090), ('Qwen', 2500)):
    put('CthumbSide' + nmv, str(round(math.sqrt(N) / 1.5)))

with open(G / 'macros_main.tex', 'w') as f:
    f.write('% generated by scripts/gen_macros.py; do not edit\n')
    for k in sorted(MAC): f.write(f'\\newcommand{{\\{k}}}{{{MAC[k]}}}\n')
json.dump(MAC, open(G / 'macros_main.json', 'w'), indent=1, sort_keys=True)
print(len(MAC), 'macros')
for k in sorted(MAC): print(k, '=', MAC[k])
