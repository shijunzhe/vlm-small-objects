from theory_check2_base import *
import theory_check2_base as TB
RES = TB.RES
# ---- REDP-X40 ----
import analyze_test as A
GEN1 = {'gpt': ('gpt54', 'px'), 'claude': ('sonnet46', 'px_claude'), 'gemini': ('gemini25', 'n1000'), 'gemma4': ('gemma4', 'n1000'), 'qwen3vl': ('qwen3vl', 'n1000')}
GEN2 = {'gpt56': ('gpt56', 'px'), 'sonnet55': ('sonnet55', 'px'), 'gem38': ('gem38', 'n1000')}
for root, table in ((Path('results_test'), GEN1), (Path('results_gen'), GEN2)):
    for k, (v, dec) in table.items():
        Ap, Am = a1_points(sorted(glob.glob(str(root / f'A1_{k}_run*.jsonl'))), dec)
        Dp, Dg = d_points(sorted(glob.glob(str(root / f'D_{k}_run*.jsonl'))))
        Ax = {d: (Ap[d], Am[d]) for d in Ap}
        report('REDP-X40', v, contrast(Ax, Dp, Dg, lambda d: A.ANN[d], v, K.TEST), lambda d: d)

# REDP-X40, further tile conditions against the whole drawing with the same model and settings:
# fixed-grid 1024 px tiles (C), tiles and whole drawing both with reasoning, and the third-round models.
# (label, A1 file key, tile file pattern, geometry key, decode)
EXTRA = [(Path('results_test'), 'gemini25 fixed tiles', 'gemini', 'C_gemini', 'gemini25', 'n1000'),
         (Path('results_test'), 'gemma4 fixed tiles', 'gemma4', 'C_gemma4', 'gemma4', 'n1000'),
         (Path('results_gen'), 'gpt56 +reasoning', 'gpt56m', 'D_gpt56m', 'gpt56', 'px'),
         (Path('results_gen'), 'sonnet55 +reasoning', 'sonnet55t', 'D_sonnet55t', 'sonnet55', 'px'),
         (Path('results_gen'), 'gem38 +reasoning', 'gem38t', 'D_gem38t', 'gem38', 'n1000'),
         (Path('results_gen'), 'gpt61', 'gpt61', 'D_gpt61', 'gpt56', 'px'),
         (Path('results_gen'), 'opus55', 'opus55', 'D_opus55', 'opus55', 'px')]
for root, lab, ak, dk, geo, dec in EXTRA:
    Ap, Am = a1_points(sorted(glob.glob(str(root / f'A1_{ak}_run*.jsonl'))), dec)
    Dp, Dg = d_points(sorted(glob.glob(str(root / f'{dk}_run*.jsonl'))))
    Ax = {d: (Ap[d], Am[d]) for d in Ap}
    report('REDP-X40', lab, contrast(Ax, Dp, Dg, lambda d: A.ANN[d], geo, K.TEST), lambda d: d)

# ---- REDP-10 (held-out drawings; one item per drawing and legend class, clustered by drawing) ----
B10 = Path('../data/redp10/redp10'); R10 = Path('results_redp10')
if (B10 / 'annotations').exists():
    ANN10 = {p.stem: json.loads(p.read_text()) for p in (B10 / 'annotations').glob('r*.json')}
    for v, dec in (('gpt56', 'px'), ('sonnet55', 'px'), ('gem38', 'n1000')):
        Ap, Am = a1_points([R10 / f'A1_{v}_run0.jsonl'], dec)
        Dp, Dg = d_points([R10 / f'D_{v}_run0.jsonl'])
        Ax = {d: (Ap[d], Am[d]) for d in Ap}
        report('REDP-10', v, contrast(Ax, Dp, Dg, lambda d: ANN10[d], v, sorted(ANN10)), lambda d: ANN10[d]['drawing'])
else:   # REDP-10 annotations are not released: keep the shipped REDP-10 rows
    _old = json.load(open('results_law/theory_check_thm1.json'))
    TB.RES.extend(o for o in _old if o['label'].startswith('REDP-10'))
    print('REDP-10 annotations not available: shipped rows kept')

# ---- FloorPlanCAD ----
import fpc_score as F
KEY = {'gpt': 'gpt54', 'claude': 'sonnet46', 'gemini': 'gemini25'}
for b in ('bench', 'sheets'):
    items = F.TEST if b == 'bench' else F.SHEETS
    clus = (lambda d: d.split('-')[0]) if b == 'bench' else (lambda d: d.split('_')[0])
    for k in ('gpt', 'claude', 'gemini', 'gemma4', 'qwen3vl', 'gpt56', 'sonnet55', 'gem38'):
        v = KEY.get(k, k)
        Ap, Am = a1_points([F.RF / b / f'A1_{k}_run0.jsonl'], F.DEC[k])
        Dp, Dg = d_points([F.RF / b / f'D_{k}_run0.jsonl'])
        Ax = {d: (Ap[d], Am[d]) for d in Ap}
        def excl(d):
            a = F.ANN[d]; tau = M.taus(a)[0]
            return lambda p: F.near_edge(p[0], p[1], a['ignore_rects'], tau)
        report('FPC-300' if b == 'bench' else 'FPC-sheets', v, contrast(Ax, Dp, Dg, lambda d: F.ANN[d], v, items, excl), clus)

json.dump(TB.RES, open('results_law/theory_check_thm1.json', 'w'), indent=1)
