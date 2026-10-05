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
