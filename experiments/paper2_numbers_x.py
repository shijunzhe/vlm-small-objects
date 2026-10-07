#!/usr/bin/env python3
"""Macros for the expansion experiments (X1-X8). Reads the stored analysis outputs and writes
../paper/generated/macros_x.tex. Run after syn_analysis.py, x_new_score.py, redp10_score.py, fsc_run.py score."""
import json, re, math
from pathlib import Path
import numpy as np

OUT = Path('../paper/generated/macros_x.tex')
W = {'0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four', '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Nine'}
MAC = {}


def nm(*parts):
    s = ''.join(str(p) for p in parts)
    s = re.sub(r'\d', lambda m: W[m.group()], s)
    return re.sub(r'[^A-Za-z]', '', s)


def f2(x, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return 'n/a'
    if abs(x) < 0.0005: return '0.00'
    if sign and x != 0 and abs(x) < 0.005:
        s = f'{x:+.3f}'   # a bound that rounds to zero keeps its sign visible
        return s.replace('-', '$-$')
    if sign and x == 0: return '0.00'
    if round(x, 2) == 0: x = 0.0
    s = f'{x:+.2f}' if sign else f'{x:.2f}'
    return s.replace('-', '$-$') if sign or x < 0 else s


def put(name, val): MAC[name] = val


def ci(lo, hi): return f'[{f2(lo, True)}, {f2(hi, True)}]'


# ---------------- X1 synthetic ----------------
Z = json.load(open('syn/analysis_robust.json'))['models']
for v, d in Z.items():
    if 'S50' in d:
        put(nm('XSfifty', v), f2(d['S50'])); put(nm('XSfiftyCI', v), f'[{d["S50_ci"][0]:.2f}, {d["S50_ci"][1]:.2f}]')
        put(nm('XlowS', v), f2(d['recall_lowS'])); put(nm('XcapSlope', v), f'{d["cap_slope"]:+.3f}'.replace('-', '$-$'))
        put(nm('XcapSlopeCI', v), f'[{d["cap_slope_ci"][0]:+.3f}, {d["cap_slope_ci"][1]:+.3f}]'.replace('-', '$-$'))
        put(nm('XcapNtwo', v), f2(d['cap_recall_n2'])); put(nm('XcapNlast', v), f2(d['cap_recall_n96'])); put(nm('XcapS', v), f2(d['S_cap']))
    if 'area' in d:
        lv = sorted(d['area'], key=lambda s: int(s.split('x')[0]) * int(s.split('x')[1]))
        put(nm('XareaFirst', v), f2(d['area'][lv[0]][1])); put(nm('XareaLast', v), f2(d['area'][lv[-1]][1]))
        put(nm('XareaTwoK', v), f2(d['area']['2048x2048'][1])); put(nm('XareaSTwoK', v), f2(d['area']['2048x2048'][0]))
        put(nm('XareaSLast', v), f2(d['area'][lv[-1]][0]))
    if 'up' in d:
        u = d['up']; put(nm('XupD', v), f2(u['d_recall'][0], True)); put(nm('XupDCI', v), ci(u['d_recall'][1], u['d_recall'][2]))
        put(nm('XupS', v), f'{u["S"][0]:.2f} to {u["S"][1]:.2f}')
        put(nm('XupEight', v), f'{u["by_m"]["8"][0]:.2f} to {u["by_m"]["8"][1]:.2f}' if '8' in u['by_m'] else 'n/a')
    if 'clut' in d:
        c = d['clut']; put(nm('XclutSlope', v), f'{c["slope"][0]:+.3f}'.replace('-', '$-$')); put(nm('XclutSlopeCI', v), f'[{c["slope"][1]:+.3f}, {c["slope"][2]:+.3f}]'.replace('-', '$-$'))
neg = sum(1 for d in Z.values() if 'cap_slope_ci' in d and d['cap_slope_ci'][1] < 0); put('XcapNneg', str(neg)); put('XcapNmodels', str(sum(1 for d in Z.values() if 'cap_slope_ci' in d)))

# ---------------- X2-X4 ----------------
X = json.load(open('results_x/x_new_scores_robust.json'))
XR = json.load(open('results_x/x_new_scores_registered.json')) if Path('results_x/x_new_scores_registered.json').exists() else {}
for key, d in X.items():
    p = key.split('|')
    if p[0] == 'X2':
        v, k = p[1], p[2]
        put(nm('XmosF', v, k), f2(d['f1'])); put(nm('XmosR', v, k), f2(d['recall'])); put(nm('XmosP', v, k), f2(d['precision']))
        if 'dF1' in d:
            put(nm('XmosdF', v, k), f2(d['dF1'][0], True)); put(nm('XmosdFCI', v, k), ci(d['dF1'][1], d['dF1'][2]))
            put(nm('XmosdR', v, k), f2(d['dR'][0], True)); put(nm('XmosdRCI', v, k), ci(d['dR'][1], d['dR'][2]))
    if p[0] == 'X3':
        v = p[1]
        for i, sub in enumerate(('All', 'Sheets', 'Views')):
            put(nm('XrsnF', v, sub), f2(d['f1'][i])); put(nm('XplainF', v, sub), f2(d['plain'][i])); put(nm('XrsnDF', v, sub), f2(d['D'][i]))
        for i, sub in enumerate(('All', 'Sheets')):
            a = d['rsn_minus_plain'][i]; b = d['rsn_minus_D'][i]
            put(nm('XrsnMinusPlain', v, sub), f2(a[0], True)); put(nm('XrsnMinusPlainCI', v, sub), ci(a[1], a[2]))
            put(nm('XrsnMinusD', v, sub), f2(b[0], True)); put(nm('XrsnMinusDCI', v, sub), ci(b[1], b[2]))
        put(nm('XrsnTok', v), f'{d["out_tokens"][0]:,.0f}'.replace(',', '{,}')); put(nm('XplainTok', v), f'{d["out_tokens"][1]:,.0f}'.replace(',', '{,}')); put(nm('XrsnN', v), str(d['n']))
    if p[0] == 'X4':
        v = p[1]
        for i, sub in enumerate(('All', 'Sheets', 'Views')): put(nm('XagF', v, sub), f2(d['f1'][i]))
        put(nm('XagMinusAOne', v), f2(d['agent_minus_A1'][0], True)); put(nm('XagMinusAOneCI', v), ci(d['agent_minus_A1'][1], d['agent_minus_A1'][2]))
        put(nm('XagMinusD', v), f2(d['agent_minus_D'][0][0], True)); put(nm('XagMinusDCI', v), ci(d['agent_minus_D'][0][1], d['agent_minus_D'][0][2]))
        put(nm('XagMinusDSheets', v), f2(d['agent_minus_D'][1][0], True)); put(nm('XagMinusDSheetsCI', v), ci(d['agent_minus_D'][1][1], d['agent_minus_D'][1][2]))
        put(nm('XagZooms', v), f'{d["zooms"]:.1f}'); put(nm('XagTokIn', v), f'{d["tok_in"] / 1000:.1f}k'); put(nm('XagTokOut', v), f'{d["tok_out"] / 1000:.1f}k')
        put(nm('XagCov', v), f'{d["coverage"]:.2f}'); put(nm('XagCovSheets', v), f'{d["coverage_sheets"]:.2f}')
        put(nm('XagRho', v), f2(d['rho_cov_recall'][0], True)); put(nm('XagRhoP', v), f'{d["rho_cov_recall"][1]:.3f}')
        put(nm('XagRhoF', v), f2(d['rho_cov_f1'][0], True)); put(nm('XagRhoFP', v), f'{d["rho_cov_f1"][1]:.3f}')
    if p[0] == 'X4b':
        v = p[1]
        put(nm('XovF', v), f2(d['ov'])); put(nm('XovNat', v), f2(d['native'])); put(nm('XovN', v), str(d['n']))
        for i, sub in enumerate(('All', 'Sheets', 'Views')):
            put(nm('XovD', v, sub), f2(d['diff'][i][0], True)); put(nm('XovDCI', v, sub), ci(d['diff'][i][1], d['diff'][i][2]))
    if p[0] == 'X4bsheets':
        put(nm('XovFSheets', p[1]), f2(d['ov'])); put(nm('XnatFSheets', p[1]), f2(d['native']))
    if p[0] == 'X4c':
        v = p[1]; put(nm('XrunOne', v), f2(d['run1'])); put(nm('XrunZero', v), f2(d['run0'])); put(nm('XrunMAD', v), f'{d["mad"]:.3f}'); put(nm('XrunN', v), str(d['n']))
for key, d in XR.items():
    if key.startswith('X2|sonnet55'):
        put(nm('XmosRegF', 'sonnet55', key.split('|')[2]), f2(d['f1']))
if 'agent_count' in X:
    for v, c in X['agent_count'].items(): put(nm('XagExactCount', v), f'{100 * c:.0f}')

# ---------------- X6 REDP-10 ----------------
R = json.load(open('results_redp10/x6_scores.json'))
for key, d in R.items():
    if key.startswith('S_A1|'):
        put(nm('XrdS', key.split('|')[1]), f'{d["median"]:.2f}'); continue
    if 'f1' in d:
        k = key.replace("('", '').replace("')", '').replace("', '", '')
        put(nm('XrdF', k), f2(d['f1']))
    elif 'item' in d:
        k = re.sub(r"[\(\)' ,]", '', key)
        put(nm('XrdD', k), f2(d['item'][0], True)); put(nm('XrdDCI', k), ci(d['item'][1], d['item'][2])); put(nm('XrdDCl', k), ci(d['cluster'][1], d['cluster'][2]))

# ---------------- X8 FSC-147 ----------------
import collections as _c
for _v in ('sonnet55', 'gpt56', 'gem38'):
    _n = _p = 0
    for _cnd in 'WUT':
        _f = Path(f'../data/fsc147/results/{_v}_{_cnd}_obj.jsonl')
        if _f.exists():
            for _l in open(_f):
                _st = json.loads(_l)['status']; _n += 1; _p += _st == 'parse_error'
    if _n: put(nm('XfscParse', _v), f'{100 * _p / _n:.0f}')
_FS = Path('../data/fsc147/x8_summary_obj_strict.json')
if _FS.exists():
    for key, d in json.load(open(_FS)).items():
        v, st = key.split('|')
        put(nm('XfscStrictD', v, st, 'UW'), f2(d['UW'][0], True)); put(nm('XfscStrictDCI', v, st, 'UW'), ci(d['UW'][1], d['UW'][2]))
        for c in 'WU': put(nm('XfscStrictF', v, st, c), f2(d['f1'][c]))
for sfx, pre in (('', 'Xfsc'), ('_obj', 'XfscO')):
    F = Path(f'../data/fsc147/x8_summary{sfx}.json')
    if not F.exists(): continue
    for key, d in json.load(open(F)).items():
        v, st = key.split('|')
        for c in ('W', 'U', 'T'): put(nm(pre + 'F', v, st, c), f2(d['f1'][c])); put(nm(pre + 'E', v, st, c), f2(d['rel'][c]))
        for c in ('UW', 'TU', 'TW'): put(nm(pre + 'D', v, st, c), f2(d[c][0], True)); put(nm(pre + 'DCI', v, st, c), ci(d[c][1], d[c][2]))
        put(nm(pre + 'S', v, st), f2(d['S'])); put(nm(pre + 'N', v, st), str(d['n'])); put(nm(pre + 'Count', v, st), str(d['count']))

OUT.write_text('% generated by stok/paper2_numbers_x.py; do not edit\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(MAC.items())))
print(len(MAC), 'macros ->', OUT)

# ---------------- table: controlled stimuli ----------------
NAMES = [('gemini25', 'Gemini 2.5 Flash'), ('gemma4', 'Gemma 4'), ('qwen3vl', 'Qwen3-VL'), ('sonnet46', 'Claude Sonnet 4.6'), ('gpt54', 'GPT-5.4'),
         ('gem38', 'Gemini 3.8 Flash'), ('sonnet55', 'Claude Sonnet 5.5'), ('gpt56', 'GPT-5.6')]
rows = []
for v, lab in NAMES:
    d = Z[v]; s50 = d['S50']
    s50s = f'{s50:.2f}' if s50 >= 0.25 else '$<$0.25'
    a = d['area']; lv = sorted(a, key=lambda s: int(s.split('x')[0]) * int(s.split('x')[1]))
    rows.append(f'{lab} & {s50s} & {d["recall_lowS"]:.2f} & {d["S_cap"]:.2f} & {d["cap_recall_n2"]:.2f} $\\to$ {d["cap_recall_n96"]:.2f} & '
                f'{d["cap_slope"]:+.3f} & {a[lv[0]][1]:.2f} $\\to$ {a[lv[-1]][1]:.2f} ({a[lv[-1]][0]:.2f}) \\\\'.replace('-0.', '$-$0.').replace('+0.', '+0.'))
    if v == 'gpt54': rows.append('\\midrule')
Path('../paper/generated/tab_syn.tex').write_text('\n'.join(rows) + '\n')
print('tab_syn written')

# overview S for the Claude agent on full sheets (t = 28 px, no resize at <= 1600 px long side)
import math as _m
import mech as _M
_S = []
for l in open('results_x/agent/sonnet55_run0.jsonl'):
    r = json.loads(l); a = json.loads((_M.BENCH / 'annotations' / f"{r['did']}.json").read_text())
    if a['subset'] == 'sheet':
        _S.append(_m.sqrt(a['template_size'][0] * a['template_size'][1]) * r['scale'] / 28)
MAC['XovSmedSheets'] = f'{np.median(_S):.2f}'
OUT.write_text('% generated by stok/paper2_numbers_x.py; do not edit\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(MAC.items())))
print('XovSmedSheets', MAC['XovSmedSheets'], 'range', round(min(_S), 2), round(max(_S), 2))

# ---------------- table: FSC-147 (both prompts) ----------------
_rows = []
_names = {'gpt56': 'GPT-5.6', 'sonnet55': 'Claude Sonnet 5.5', 'gem38': 'Gemini 3.8'}
for sfx, lab in (('_obj', 'neutral prompt (X8b)'), ('', 'drawing prompt (X8)')):
    F = Path(f'../data/fsc147/x8_summary{sfx}.json')
    if not F.exists(): continue
    Z8 = json.load(open(F))
    _rows.append(f'\\multicolumn{{8}}{{l}}{{\\emph{{{lab}}}}} \\\\')
    for v in ('gpt56', 'sonnet55', 'gem38'):
        for st in ('small', 'medium', 'large'):
            d = Z8.get(f'{v}|{st}')
            if not d: continue
            _rows.append(f'{_names[v] if st == "small" else ""} & {st} & {d["S"]:.2f} & {d["f1"]["W"]:.2f} & {d["f1"]["U"]:.2f} & {d["f1"]["T"]:.2f} & '
                         f'{f2(d["UW"][0], True)} {ci(d["UW"][1], d["UW"][2])} & {f2(d["TU"][0], True)} {ci(d["TU"][1], d["TU"][2])} \\\\')
Path('../paper/generated/tab_fsc.tex').write_text('\n'.join(_rows) + '\n')

# tile pipeline token cost per drawing (second-round D, all runs averaged per drawing)
import glob as _g
for _v in ('gpt56', 'sonnet55', 'gem38'):
    _per = {}
    for _f in _g.glob(f'results_gen/D_{_v}_run*.jsonl'):
        _run = _f.split('_run')[1].split('.')[0]
        for _l in open(_f):
            r = json.loads(_l); u = r.get('usage') or {}
            if r['status'] not in ('success', 'parse_error'): continue
            k = (r['did'], _run); a = _per.setdefault(k, [0, 0]); a[0] += u.get('input_tokens') or 0; a[1] += (u.get('output_tokens') or 0)
    _bydid = {}
    for (d, rn), (i, o) in _per.items(): _bydid.setdefault(d, []).append((i, o))
    _mi = np.mean([np.mean([x[0] for x in v]) for v in _bydid.values()]); _mo = np.mean([np.mean([x[1] for x in v]) for v in _bydid.values()])
    MAC[nm('XtileTokIn', _v)] = f'{_mi / 1000:.1f}k'; MAC[nm('XtileTokOut', _v)] = f'{_mo / 1000:.1f}k'
    print('tile tokens', _v, round(_mi), round(_mo), 'n', len(_bydid))
OUT.write_text('% generated by stok/paper2_numbers_x.py; do not edit\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(MAC.items())))

# ---------------- X5 learned exemplar detectors ----------------
_LB = Path('results_x/learned/lb_scores.json')
if _LB.exists():
    for _m, _d in json.load(open(_LB)).items():
        MAC[nm('XlbF', _m)] = f2(_d['f1'][0]); MAC[nm('XlbFSheets', _m)] = f2(_d['f1'][1]); MAC[nm('XlbFViews', _m)] = f2(_d['f1'][2])
        MAC[nm('XlbOracle', _m)] = f2(_d['oracle']); MAC[nm('XlbRd', _m)] = f2(_d['redp10_f1']); MAC[nm('XlbRdOracle', _m)] = f2(_d['redp10_oracle'])
OUT.write_text('% generated by stok/paper2_numbers_x.py; do not edit\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(MAC.items())))

# ---------------- X9 confound controls ----------------
_X9 = Path('results_x/x9_scores.json')
if _X9.exists():
    Z9 = json.load(open(_X9))
    for k, v in Z9.items():
        p = k.split('|')
        if len(p) == 2 and isinstance(v, dict) and 'sheets' in v:
            MAC[nm('XnF', p[0], p[1], 'Sheets')] = f2(v['sheets']); MAC[nm('XnF', p[0], p[1], 'All')] = f2(v['all'])
        elif isinstance(v, list) and len(v) >= 4:
            key = nm('Xn', *p)
            MAC[key] = f2(v[0], True); MAC[key + 'CI'] = ci(v[1], v[2]); MAC[key + 'P'] = f'{v[3]:.3f}'
        elif k.endswith('|holm'):
            MAC[nm('Xn', k.replace('|holm', ''), 'Holm')] = f'{v:.3f}'
# ---------------- T1 / T2 ----------------
_T1 = Path('results_x/t1_syn.json')
if _T1.exists():
    t1 = json.load(open(_T1)); MAC['XtOneSfifty'] = f2(t1['S50_pooled']); MAC['XtOneDAIC'] = f'{t1["aic"][0] - t1["aic"][1]:.1f}'
    MAC['XtOneHThree'] = f2(t1['h3'][0], True); MAC['XtOneHThreeCI'] = ci(t1['h3'][1], t1['h3'][2])
_T2 = Path('results_x/t2_nested_redp.json')
if _T2.exists():
    for v, d in json.load(open(_T2)).items():
        MAC[nm('XtTwoRS', v)] = f2(d['r2_logS']); MAC[nm('XtTwoRC', v)] = f2(d['r2_ceil']); MAC[nm('XtTwoDC', v)] = f2(d['ceil_given_S'][0], True); MAC[nm('XtTwoDCP', v)] = f'{d["ceil_given_S"][2]:.3f}'
# ---------------- learned v2 ----------------
_L2 = Path('results_x/learned2/lb2_scores.json')
if _L2.exists():
    for k, d in json.load(open(_L2)).items():
        MAC[nm('XlbTwo', *k.split('|'))] = f2(d['f1'][0] if isinstance(d['f1'], list) else d['f1'])
OUT.write_text('% generated by stok/paper2_numbers_x.py; do not edit\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(MAC.items())))

# ---------------- FPC (FloorPlanCAD) ----------------
for _sfx, _pre in (('', 'Xf'), ('_block0', 'XfB')):
    _F = Path(f'results_fpc/fpc_scores{_sfx}.json')
    if not _F.exists(): continue
    ZF = json.load(open(_F))
    if 'tm_threshold' in ZF and ZF['tm_threshold'] is not None: MAC[_pre + 'TMthr'] = f2(ZF['tm_threshold'])
    for k, v in ZF.items():
        p = k.split('|')
        if p[0] == 'bench' and isinstance(v, list):
            for c, x in zip(('A', 'D', 'AG'), v):
                if x is not None: MAC[nm(_pre, 'Bench', c, p[1])] = f2(x)
        elif p[0] == 'H12':
            MAC[nm(_pre, 'DmA', p[1])] = f2(v[0], True); MAC[nm(_pre, 'DmACI', p[1])] = ci(v[1], v[2])
        elif p[0] == 'sheets':
            for kk, x in zip((1, 2, 3), v): MAC[nm(_pre, 'Sh', p[1], p[2], 'k', kk)] = f2(x)
        elif p[0] == 'T1':
            for nmv, x in zip(('Lo', 'Def', 'Uh'), v['f1']): MAC[nm(_pre, 'TOne', p[1], p[2], nmv)] = f2(x)
            MAC[nm(_pre, 'TOne', p[1], p[2], 'DL')] = f2(v['def_lo'][0], True); MAC[nm(_pre, 'TOne', p[1], p[2], 'DLCI')] = ci(v['def_lo'][1], v['def_lo'][2])
            MAC[nm(_pre, 'TOne', p[1], p[2], 'UD')] = f2(v['uh_def'][0], True); MAC[nm(_pre, 'TOne', p[1], p[2], 'UDCI')] = ci(v['uh_def'][1], v['uh_def'][2])
            MAC[nm(_pre, 'TOne', p[1], p[2], 'N')] = str(v['n'])
        elif p[0] == 'H4':
            MAC[nm(_pre, 'HFour')] = f2(v[0], True); MAC[nm(_pre, 'HFourCI')] = ci(v[1], v[2])
        elif p[0] == 'H7':
            MAC[nm(_pre, 'HSeven', p[1])] = f2(v[0], True); MAC[nm(_pre, 'HSevenCI', p[1])] = ci(v[1], v[2])
        elif p[0] in ('H3', 'H8'):
            MAC[nm(_pre, p[0], p[1])] = f2(v[0], True); MAC[nm(_pre, p[0], p[1], 'P')] = f'{v[1]:.3f}'
        elif p[0] == 'k3k1':
            (f, r) = v; MAC[nm(_pre, 'KD', p[1], p[2])] = f2(f[0], True); MAC[nm(_pre, 'KDCI', p[1], p[2])] = ci(f[1], f[2])
            MAC[nm(_pre, 'KDR', p[1], p[2])] = f2(r[0], True); MAC[nm(_pre, 'KDRCI', p[1], p[2])] = ci(r[1], r[2])

# ---- open-model zoom agent (Qwen3-VL), REDP-X40 ----
_Q = Path('results_x/x4q_scores.json')
if _Q.exists():
    q = json.load(open(_Q))['n1000']
    put('XqAgF', f2(q['f1'][0])); put('XqAgFSheets', f2(q['f1'][1])); put('XqD', f2(q['D'][0])); put('XqATwo', f2(q['A2'][0]))
    put('XqAgMinusD', f2(q['ag_minus_D'][0][0], True)); put('XqAgMinusDCI', ci(q['ag_minus_D'][0][1], q['ag_minus_D'][0][2]))
    put('XqExactAg', f'{100 * q["exact_count"][0]:.0f}'); put('XqExactD', f'{100 * q["exact_count"][1]:.0f}')
    put('XqCountErrAg', f2(q['count_err'][0])); put('XqCountErrD', f2(q['count_err'][1]))
    put('XqZooms', f'{q["zooms"]:.1f}'); put('XqTokIn', f'{q["tok_in"] / 1000:.1f}k')

# ---- learned counters validated on FSC-147 (leave-one-image-out threshold) ----
_FC = Path('results_x/learned2/fsc_counters.json')
if _FC.exists():
    fc = json.load(open(_FC))
    for m in ('countgd', 'geco2'):
        put(nm('XctrFsc', m), f2(fc[m]['f1']))
        for st in ('small', 'medium', 'large'): put(nm('XctrFsc', m, st), f2(fc[m][st]['f1']))
    for v in ('gpt56', 'sonnet55', 'gem38'):
        if f'vlm_W|{v}' in fc:
            put(nm('XctrFscVlm', v), f2(fc[f'vlm_W|{v}']['f1']))
            for st in ('small', 'medium', 'large'): put(nm('XctrFscVlm', v, st), f2(fc[f'vlm_W|{v}'][st]))

# ---- defaults for FPC-300 table cells whose runs are not finished (shown as n/a until scored) ----
for _m in ('gpt', 'claude', 'gemini', 'gemma4', 'qwen3vl', 'gpt56', 'sonnet55', 'gem38'):
    for _c in ('A', 'D', 'AG'): MAC.setdefault(nm('XfBench', _c, _m), 'n/a')

# ---- FPC-sheets appendix tables (all blocks and first block) ----
_NAMES = [('gpt', 'GPT-5.4'), ('claude', 'Claude Sonnet 4.6'), ('gemini', 'Gemini 2.5 Flash'), ('gemma4', 'Gemma 4 31B'), ('qwen3vl', 'Qwen3-VL'),
          ('gpt56', 'GPT-5.6'), ('sonnet55', 'Claude Sonnet 5.5'), ('gem38', 'Gemini 3.8 (default)'), ('gem38lo', 'Gemini 3.8 (low)'), ('gem38uh', 'Gemini 3.8 (ultra-high)')]
for _f, _out in (('fpc_scores.json', 'tab_fpcsheets_all.tex'), ('fpc_scores_block0.json', 'tab_fpcsheets_b0.tex')):
    _p = Path('results_fpc') / _f
    if not _p.exists(): continue
    _d = json.load(open(_p)); rows = []
    for _m, _lab in _NAMES:
        cells = []
        for _c in ('A1', 'D', 'AG'):
            v = _d.get(f'sheets|{_c}|{_m}')
            cells += [f2(x) if x is not None else '' for x in v] if v else ['', '', '']
        rows.append(_lab + ' & ' + ' & '.join(cells) + ' \\\\\n')
        if _m == 'qwen3vl': rows.append('\\midrule\n')
    tm = _d.get('sheets|TM|tm')
    rows.append('\\midrule\nTemplate matching & \\multicolumn{9}{c}{' + ' / '.join(f2(x) for x in tm) + '} \\\\\n')
    (OUT.parent / _out).write_text(''.join(rows))

# ---- T2 on FloorPlanCAD: how often the ceiling adds variance beyond log S ----
_T2F = Path('results_x/t2_nested_fpc.json')
if _T2F.exists():
    t2f = json.load(open(_T2F))
    for _b, _lab in (('bench', 'Bench'), ('sheets', 'Sheets')):
        rows_ = [v for k, v in t2f.items() if k.startswith(_b + '|')]
        if not rows_: continue
        sig = [r for r in rows_ if r['ceil_given_S'][2] < 0.05]; sigS = [r for r in rows_ if r['S_given_ceil'][2] < 0.05]
        put(f'XtTwoF{_lab}N', str(len(rows_))); put(f'XtTwoF{_lab}NsigC', str(len(sig))); put(f'XtTwoF{_lab}NsigS', str(len(sigS)))
        put(f'XtTwoF{_lab}MaxDC', f2(max(r['ceil_given_S'][0] for r in rows_))); put(f'XtTwoF{_lab}MaxDS', f2(max(r['S_given_ceil'][0] for r in rows_)))
        put(f'XtTwoF{_lab}MaxRB', f2(max(r['r2_both'] for r in rows_)))

# ---- post hoc: FPC-300 tile effect against the change in S ----
_SR = Path('results_fpc/fpc_sratio.json')
if _SR.exists():
    for _v, _d in json.load(open(_SR)).items():
        put(nm('XfSrRho', _v), f2(_d['rho'][0], True)); put(nm('XfSrRhoP', _v), f'{_d["rho"][1]:.3f}')
        for _k in ('up', 'down'):
            if _d[_k][0] is not None: put(nm('XfSr', _k, _v), f2(_d[_k][0], True))
            if _d[_k][1] is not None: put(nm('XfSr', _k, 'CI', _v), ci(_d[_k][1], _d[_k][2]))
        put(nm('XfSrFracDown', _v), f'{100 * _d["frac_down"]:.0f}')

# ---- theory check: every ordering implied by Assumption 1 ----
_TC = Path('results_law/theory_check_final.json')
if _TC.exists():
    tc = json.load(open(_TC)); tested = [o for o in tc if o['verdict'] in ('supported', 'consistent', 'violated')]; outs = [o for o in tc if o['family'] == 'outside']
    th = [o for o in tested if o['family'] == 'Thm1']
    put('XtcN', str(len(tested))); put('XtcSup', str(sum(o['verdict'] == 'supported' for o in tested)))
    put('XtcCons', str(sum(o['verdict'] == 'consistent' for o in tested))); put('XtcViol', str(sum(o['verdict'] == 'violated' for o in tested)))
    put('XtcThmN', str(len(th))); put('XtcThmSup', str(sum(o['verdict'] == 'supported' for o in th)))
    put('XtcThmMin', f2(min(o['diff'] for o in th), True)); put('XtcThmMax', f2(max(o['diff'] for o in th), True))
    put('XtcOutN', str(len(outs))); put('XtcOutNeg', str(sum(o['hi'] < 0 for o in outs))); put('XtcOutNegPoint', str(sum(o['diff'] < 0 for o in outs)))
    put('XtcExpFalse', f'{0.025 * len(tested):.1f}')
    put('XtcInteriorPct', f"{100 * sum(o['targets'] for o in th) / sum(o['targets_all'] for o in th):.0f}")
    v = [o for o in tested if o['verdict'] == 'violated'][0]
    put('XtcViolD', f2(v['diff'], True)); put('XtcViolCI', ci(v['lo'], v['hi']))
    put('XtcViolDpos', f2(-v['diff']) + ' (95\\% interval ' + f2(-v['hi']) + ' to ' + f2(-v['lo']) + ')')
    _vv = {(o['label'], 'reasoning' in o['rule']): o for o in tc if o['label'].startswith('REDP-X40') and ('original mode' in o['rule'] or 'ultra-high' in o['rule'])}
    for _k, _key in ((('REDP-X40 gpt54', False), 'GptFiveFour'), (('REDP-X40 gpt56', False), 'GptFiveSix'), (('REDP-X40 gpt56', True), 'GptFiveSixR'), (('REDP-X40 gem38', False), 'GemThreeEight')):
        if _k in _vv:
            put('XtcVend' + _key, f2(_vv[_k]['diff'], True)); put('XtcVendCI' + _key, ci(_vv[_k]['lo'], _vv[_k]['hi'])); put('XtcVendN' + _key, str(_vv[_k]['n']))
    put('XtcViolVendor', str(sum(o['verdict'] == 'violated' for o in _vv.values())))
    for o in tc:
        if o['label'] in ('FPC-300 gpt56', 'FPC-300 sonnet55', 'FPC-300 gemma4') and o['family'] in ('Thm1', 'outside'):
            key = ('In' if o['family'] == 'Thm1' else 'Out') + o['label'].split()[1]
            put(nm('Xtc', key), f2(o['diff'], True)); put(nm('XtcCI', key), ci(o['lo'], o['hi']))

# ---- direct synthetic test of the assumptions (prereg v7) ----
_AS = Path('syn_assume/assume_scores.json')
if _AS.exists():
    asd = json.load(open(_AS))
    put('XasN', str(len(asd))); put('XasSup', str(sum(v['verdict'] == 'supported' for v in asd.values())))
    put('XasCons', str(sum(v['verdict'] == 'consistent' for v in asd.values()))); put('XasViol', str(sum(v['verdict'] == 'violated' for v in asd.values())))
    put('XasMin', f2(min(v['diff'] for v in asd.values()), True))
    for _v, _n in (('gpt56', 'GptFiveSix'), ('gpt61', 'GptSixOne'), ('opus55', 'OpusFiveFive')):
        _k = f'{_v}|M: crop - whole (same scale)'
        if _k in asd: put('XasM' + _n, f2(asd[_k]['diff'], True))
    cols = ['M: crop - whole (same scale)', 'R: 2x crop - crop', 'I: native - degraded (same scale)', 'Prop crop: 2x crop - whole', 'Thm1: safe tiles - whole']
    NAMES = [('gpt54', 'GPT-5.4'), ('gpt56', 'GPT-5.6'), ('sonnet55', 'Claude Sonnet 5.5'), ('qwen3vl', 'Qwen3-VL'), ('gemma4', 'Gemma 4'), ('gem38', 'Gemini 3.8'),
             ('opus55', 'Claude Opus 5.5$^\\dagger$'), ('gpt61', 'GPT-6.1 Sol$^\\dagger$')]
    rows = []
    for v, lab in NAMES:
        if v == 'opus55': rows.append('\\midrule\n')
        cells = []
        for c in cols:
            d = asd.get(f'{v}|{c}')
            if not d: cells.append('n/a'); continue
            txt = f2(d['diff'], True) + '{\\scriptsize\\,' + ci(d['lo'], d['hi']) + '}'
            cells.append('\\textbf{' + f2(d['diff'], True) + '}{\\scriptsize\\,' + ci(d['lo'], d['hi']) + '}' if d['verdict'] == 'supported' else txt)
        rows.append(lab + ' & ' + ' & '.join(cells) + ' \\\\\n')
    (OUT.parent / 'tab_assume.tex').write_text(''.join(rows))
    thm = [v for k, v in asd.items() if k.endswith('Thm1: safe tiles - whole')]
    put('XasThmMin', f2(min(v['diff'] for v in thm), True)); put('XasThmMax', f2(max(v['diff'] for v in thm), True))
    put('XasThmSup', str(sum(v['verdict'] == 'supported' for v in thm))); put('XasModels', str(len(thm)))
    _new = {k: v for k, v in asd.items() if k.split('|')[0] in ('opus55', 'gpt61')}
    put('XasNewN', str(len(_new))); put('XasNewSup', str(sum(v['verdict'] == 'supported' for v in _new.values()))); put('XasNewViol', str(sum(v['verdict'] == 'violated' for v in _new.values())))
    try:
        _fu2 = json.load(open('theory_v8/f1_union.json'))
        for _v, _n in (('gpt61', 'GptSixOne'), ('opus55', 'OpusFiveFive'), ('gpt56', 'GptFiveSix')):
            put('XasWholeRec' + _n, f2(_fu2[_v]['agg'][0] / 576))
    except Exception as e:
        print('whole rec', e)
    put('XasTargetsCrop', str(max(v['n_targets'] for k, v in asd.items() if 'crop' in k))); put('XasTargetsTile', str(max(v['n_targets'] for k, v in asd.items() if 'Thm1' in k)))

# ---- tokens one whole-sheet call would need for S = 1.5 (Prop. geometry a), REDP-X40 full sheets ----
try:
    import analyze_test as _A, ceilings as _K
    _sh = [d for d in _K.TEST if _A.SUB[d] == 'sheet']
    _need = [(1.5 * math.sqrt(_A.ANN[d]['width'] * _A.ANN[d]['height']) / math.sqrt(_A.ANN[d]['template_size'][0] * _A.ANN[d]['template_size'][1])) ** 2 for d in _sh]
    put('XneedTokMed', f'{int(round(float(np.median(_need)), -2)):,}'.replace(',', '{,}')); put('XneedTokMax', f'{int(round(max(_need), -3)):,}'.replace(',', '{,}'))
except Exception as e:
    print('need tokens', e)

# ---- F1 of the safe tiling on the assumption battery (Prop. f1), and false points under crops; existing calls only ----
try:
    _fu = json.load(open('theory_v8/f1_union.json')); _of = json.load(open('theory_v8/output_form.json'))
    _d = [v['dF1'] for v in _fu.values()]
    put('XfoFOneMin', f2(min(x[0] for x in _d), True)); put('XfoFOneMax', f2(max(x[0] for x in _d), True))
    put('XfoFOneSup', str(sum(x[1] > 0 for x in _d))); put('XfoFOneN', str(len(_d)))
    put('XfoFewer', str(sum(v['rate'] <= 1 for v in _fu.values())))
    _g = _fu['gemma4']; put('XfoGemmaRate', f'{_g["rate"]:.1f}'); put('XfoGemmaRho', f'{_g["rho_w"]:.0f}')
    put('XfoGemmaRecall', f'{100 * _g["agg"][0] / 576:.0f}')
    put('XfoCropFalse', str(sum(v['M:F'][2] < 0 for v in _of.values()))); put('XfoModels', str(len(_of)))
except Exception as e:
    print('f1 union', e)

# ---- bits about target positions carried by whole-image answers (Thm capacity), assumption battery ----
try:
    _ic = json.load(open('theory_v10/info_carried.json'))
    _best = max(((v, m, d) for v, r in _ic.items() for m, d in r.items()), key=lambda z: z[2]['I'])
    put('XcapBestBits', f"{_best[2]['I']:.0f}"); put('XcapBestFull', f"{_best[2]['Lfull']:.0f}"); put('XcapBestM', _best[1])
    put('XcapBestRecall', f2(_best[2]['recall']))
except Exception as e:
    print('cap bits', e)

# ---- defects implied by the lower 95% limits of the (M) and (R) pairs (Thm robust illustration) ----
try:
    _asd = json.load(open('syn_assume/assume_scores.json'))
    _NM = {'gpt54': 'GPT-5.4', 'gpt56': 'GPT-5.6', 'sonnet55': 'Claude Sonnet 5.5', 'qwen3vl': 'Qwen3-VL', 'gemma4': 'Gemma 4', 'gem38': 'Gemini 3.8', 'opus55': 'Claude Opus 5.5', 'gpt61': 'GPT-6.1 Sol'}
    _def = {v: max(0, -_asd[f'{v}|M: crop - whole (same scale)']['lo']) + max(0, -_asd[f'{v}|R: 2x crop - crop']['lo']) for v in _NM if f'{v}|M: crop - whole (same scale)' in _asd}
    _vm = max(_def, key=_def.get)
    put('XrobMax', f2(_def[_vm])); put('XrobMaxModel', _NM[_vm]); put('XrobZero', str(sum(x < 0.005 for x in _def.values())))
except Exception as e:
    print('robust', e)

# ---- prereg v8: the safe rule (Cor. cor:safe) on REDP-X40 and FPC-sheets k3 ----
try:
    _cs = json.load(open('results_cs/cs_summary.json'))
    _NMc = {'gpt61': 'GPT-6.1', 'opus55': 'Claude Opus 5.5', 'gem38': 'Gemini 3.8', 'qwen3vl': 'Qwen3-VL'}
    rows = []; diffs = []; nsig = 0; nviol = 0; tokr = []
    for bench, lab in (('redp', 'REDP-X40'), ('fpc', 'FPC-sheets')):
        for v in ('gpt61', 'opus55', 'gem38', 'qwen3vl'):
            k = f'{bench}|{v}'
            if k not in _cs: continue
            S_ = _cs[k]['summary']; d = S_['CS-W']['recall']; diffs.append(d[0]); nsig += d[1] > 0; nviol += d[2] < 0
            dtxt = (('\\textbf{' + f2(d[0], True) + '}') if d[1] > 0 else f2(d[0], True)) + '{\\scriptsize\\,' + ci(d[1], d[2]) + '}'
            if 'CS-D' in S_:
                e = S_['CS-D']['recall']; etxt = f2(e[0], True) + '{\\scriptsize\\,' + ci(e[1], e[2]) + '}'; ttxt = f"{S_['tok_CS_over_D']:.2f}"; tokr.append(S_['tok_CS_over_D'])
            else: etxt = ttxt = 'n/a'
            _rc = float(np.median([x['ratio_cmin'] for x in _cs[k]['rows']]))
            rows.append(f"{lab} & {_NMc[v]} & {f2(S_['W']['recall'])} & {f2(S_['CS']['recall'])} & {dtxt} & {etxt} & {ttxt} & {_rc:.1f} \\\\\n")
    (OUT.parent / 'tab_csrule.tex').write_text(''.join(rows))
    put('XcsCells', str(len(diffs))); put('XcsSig', str(nsig)); put('XcsViol', str(nviol))
    put('XcsMax', f2(max(diffs), True)); put('XcsMin', f2(min(diffs), True))
    put('XcsTokMin', f'{100 * min(tokr):.0f}'); put('XcsTokMax', f'{100 * max(tokr):.0f}')
    _q = _cs['fpc|qwen3vl']['summary']
    put('XcsPthreeQ', f2(_q['P3']['recall'][0], True) + ' ' + ci(_q['P3']['recall'][1], _q['P3']['recall'][2])); put('XcsPthreeQn', str(_q['P3_n']))
    put('XcsOneViewGpt', f"{100 * _cs['fpc|gpt61']['summary']['frac_one_view']:.0f}")
    put('XcsGemRedp', f2(_cs['redp|gem38']['summary']['CS-W']['recall'][0], True))
    put('XcsGainGptSixOneRedp', f2(_cs['redp|gpt61']['summary']['CS-W']['recall'][0], True))
    put('XcsWholeGptSixOneRedp', f2(_cs['redp|gpt61']['summary']['W']['recall'])); put('XcsWholeGptSixOneFpc', f2(_cs['fpc|gpt61']['summary']['W']['recall']))
    _ra = [x['ratio_cmin'] for k in _cs for x in _cs[k]['rows']]
    _cellmed = [float(np.median([x['ratio_cmin'] for x in _cs[k]['rows']])) for k in _cs]
    put('XcsRatioMinCell', f'{min(_cellmed):.1f}'); put('XcsRatioMaxCell', f'{max(_cellmed):.1f}')
    put('XcsRatioMed', f'{np.median(_ra):.1f}'); put('XcsRatioNinety', f'{np.percentile(_ra, 90):.1f}')
except Exception as e:
    print('csrule', e)
# ---- overall scale: model calls, tokens, images, annotated instances ----
try:
    import hashlib as _h, glob as _gg
    _seen = set(); _tin = 0; _nreq = [0]
    _files = [f for f in _gg.glob('**/*.jsonl', recursive=True) + _gg.glob('../data/fsc147/results/**/*.jsonl', recursive=True)
              if 'pilot' not in f and 'logs' not in f and 'k5_scp' not in f]
    for _f in sorted(_files):
        for _l in open(_f):
            try: _r = json.loads(_l)
            except Exception: continue
            _k = _h.md5(json.dumps(_r, sort_keys=True).encode()).hexdigest()
            if _k in _seen: continue          # merged shard files repeat records
            _seen.add(_k); _u = _r.get('usage') or {}
            if '/learned' in _f: _seen.discard(_k); continue   # trained counters: no VLM request
            if isinstance(_r.get('log'), dict) and 'turns' in _r['log']:   # a zoom-agent episode: one request per turn
                _nreq[0] += max(int(_r['log']['turns']), 1) - 1; _tin += _r['log'].get('in') or 0; continue
            if isinstance(_u, dict): _tin += _u.get('input_tokens') or _u.get('prompt_tokens') or 0
    def _ann(d, pat='*.json'):
        fs = _gg.glob(d + '/' + pat); return len(fs), sum(json.load(open(f))['count'] for f in fs)
    _redp = _ann('../data/redp40/redp_x40/annotations'); _fpcb = _ann('../data/fpc/bench/annotations')
    _fpcs = _ann('../data/fpc/sheets/annotations', '*_k3.json')       # k1 and k2 sheets are nested in k3
    _fsc = json.load(open('../data/fsc147/x8_items.json'))
    _syn = json.load(open('syn/items.json')); _sa = json.load(open('syn_assume/items.json'))
    _R10_IMG, _R10_INST = 10, 225                                            # REDP-10 annotations are not released
    _img = _redp[0] + _R10_IMG + _fpcb[0] + _fpcs[0] + len(_fsc) + len(_syn) + len(_sa)
    _inst = _redp[1] + _R10_INST + _fpcb[1] + _fpcs[1] + sum(i['n'] for i in _fsc) + sum(len(i['points']) for i in _syn) + sum(len(i['points']) for i in _sa)
    put('XscaleCalls', f"{round(len(_seen) + _nreq[0], -3):,}".replace(',', '{,}')); put('XscaleTokM', f"{_tin / 1e6:.0f}")
    put('XscaleImages', f"{_img:,}".replace(',', '{,}')); put('XscaleInst', f"{round(_inst, -2):,}".replace(',', '{,}'))
    put('XscaleRealImg', str(_redp[0] + _R10_IMG + _fpcb[0] + _fpcs[0])); put('XscaleFscInst', f"{sum(i['n'] for i in _fsc):,}".replace(',', '{,}'))
    print('scale', len(_seen) + _nreq[0], _tin, _img, _inst)
except Exception as e:
    print('scale', e)
OUT.write_text('% generated by stok/paper2_numbers_x.py; do not edit\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(MAC.items())))

# ---- FPC-sheets first block: the drop with more context, restricted to sheets where the whole 3x3 sheet still gives S >= 1 ----
try:
    import pickle as _pk
    _G = json.load(open('../data/fpc/geometry.json')); _P0 = _pk.load(open('results_fpc/fpc_per_block0.pkl', 'rb'))
    _rg = np.random.default_rng(11)
    for _v, _n in (('gpt56', 'GptFiveSix'), ('sonnet55', 'SonnetFiveFive'), ('gem38', 'GemThreeEight')):
        _per = _P0.get(str(('sheets', 'A1', _v)), {})
        _r = {d: float(np.nanmean([x[0] for x in rs])) for d, rs in _per.items() if rs and not np.all(np.isnan([x[0] for x in rs]))}
        _pairs = [(_r[d[:-1] + '3'] - _r[d], _G[d[:-1] + '3'][_v]['A1']) for d in _r if d.endswith('_k1') and d[:-1] + '3' in _r]
        _x = np.array([p[0] for p in _pairs]); _s = np.array([p[1] for p in _pairs], float); _xs = _x[_s >= 1]
        _bs = [_xs[_rg.integers(0, len(_xs), len(_xs))].mean() for _ in range(4000)]
        put('XfBKSone' + _n, f2(float(_xs.mean()), True)); put('XfBKSoneCI' + _n, ci(float(np.percentile(_bs, 2.5)), float(np.percentile(_bs, 97.5))))
        put('XfBKSoneN' + _n, str(len(_xs))); put('XfBKSmed' + _n, f'{np.median(_s):.1f}'); put('XfBKSltOne' + _n, f'{100 * np.mean(_s < 1):.0f}')
except Exception as e:
    print('fpc block0 S>=1', e)
OUT.write_text('% generated by stok/paper2_numbers_x.py; do not edit\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(MAC.items())))
