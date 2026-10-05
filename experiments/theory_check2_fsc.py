import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import theory_check2_base as TB
import fsc_run as FR
ROOT = Path('../data/fsc147'); ITEMS = {it['name']: it for it in json.load(open(ROOT / 'x8_items.json'))}
val = set(json.load(open(ROOT / 'Train_Test_Val_FSC_147.json'))['val'])
for v in ('gpt56', 'sonnet55', 'gem38'):
    recs = {}
    for cond in ('W', 'T'):
        g = defaultdict(dict)
        for l in open(ROOT / 'results' / f'{v}_{cond}_obj.jsonl'):
            r = json.loads(l)
            if r['status'] in ('success', 'parse_error'): g[r['name']][r['v']] = r
        recs[cond] = g
    rows = []
    for name in ITEMS:
        if name in val or name not in recs['W'] or name not in recs['T'] or len(recs['T'][name]) < 4: continue
        it = ITEMS[name]; m = it['m']; a = FR.ANN[name]; W, H = recs['W'][name][0]['size'][0] / recs['W'][name][0]['scale'], recs['W'][name][0]['size'][1] / recs['W'][name][0]['scale']
        tiles = []
        for r in recs['T'][name].values():
            ox, oy = r['origin']; sw, sh = r['size']; s = r['scale']
            tiles.append({'core': tuple(r['core']), 'view': (ox, oy, ox + sw / s, oy + sh / s), 'sent': (sw, sh)})
        ann = {'template_size': [m, m], 'points': [tuple(p) for p in a['points']], 'width': W, 'height': H}
        tg, tau = TB.interior_targets(ann, tiles, W, H)
        if not tg: continue
        rw = recs['W'][name][0]
        s_w = TB.S_view(v, W, H, rw['size'][0], rw['size'][1], m)
        s_t = min(TB.S_view(v, t['view'][2] - t['view'][0], t['view'][3] - t['view'][1], t['sent'][0], t['sent'][1], m) for t in tiles)
        pw = FR.native_points(rw); pt = sum((FR.native_points(r) for r in recs['T'][name].values()), [])
        rows.append({'item': name, 'cond': s_t >= s_w - 1e-9, 'diff': TB.hit(pt, tg, tau) - TB.hit(pw, tg, tau), 'n_targets': len(tg), 'n_all': len(a['points']), 'S_w': s_w, 'S_t': s_t})
    TB.report('FSC-147', v, rows, lambda d: d)
old = json.load(open('results_law/theory_check_thm1.json'))
json.dump([o for o in old if not o['label'].startswith('FSC')] + TB.RES, open('results_law/theory_check_thm1.json', 'w'), indent=1)
