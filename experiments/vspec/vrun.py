#!/usr/bin/env python3
"""Item building, resumable runner and scoring for the visual-spec experiment."""
import json, math, re, sys, time, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import vspec as V, vcall as VC

ROOT = V.ROOT
COORD_TEXT = {
    'px': ("The image is {w}x{h} pixels. All coordinates you output must be in this\n"
           "image's pixel space, where (0,0) is the top-left corner.", '"cx": <pixel_x>, "cy": <pixel_y>'),
    'n1000': ("Report coordinates normalized to the image: x and y each range from 0 to 1000,\n"
              "where (0,0) is the top-left corner and (1000,1000) is the bottom-right corner of the image.",
              '"cx": <x_0_to_1000>, "cy": <y_0_to_1000>'),
}
SYS = """You are a precise engineering drawing analysis assistant.

You will be shown a reference image of ONE target symbol, followed by a drawing.
Find every instance of that target symbol in the drawing. Other symbols and linework are distractors.

{coord}

Respond with ONLY a JSON object (no markdown, no commentary). Format:
{{"target_type": "target_symbol", "count": <integer>, "shapes": [{{"type": "target_symbol", {fmt}}}, ...]}}
The "count" must equal the length of "shapes". Count only symbols matching the reference."""
USER = "The first image is the reference symbol; the second is the drawing. Find every instance of the reference symbol. JSON only."


def system_prompt(model, W, H):
    t, f = COORD_TEXT[V.MODELS[model]['coord']]
    return SYS.format(coord=t.format(w=W, h=H), fmt=f)


# ---------------- items ----------------
def make_item(phase, lay, model, S, tag, pad=True):
    r = V.render(lay, model, S, pad=pad)
    if 'img' not in r:
        return None, r
    iid = f"{phase}_{tag}_{model}_{lay['set']}_i{lay['i']}_n{lay['n']}_S{S:.4f}" + ('' if pad else '_nopad')
    d = ROOT / 'images' / phase; d.mkdir(parents=True, exist_ok=True)
    r['img'].save(d / f'{iid}.png')
    W, H = r['geo']['canvas']
    it = dict(id=iid, phase=phase, tag=tag, model=model, set=lay['set'], i=lay['i'], n=lay['n'], k=lay['k'], S=S, pad=pad,
              target=lay['target'], distractors=lay['distractors'], points=r['points'], geo=r['geo'],
              exp_img_tokens=V.expected_tokens(model, W, H), image=str(d / f'{iid}.png'))
    return it, r


# ---------------- runner ----------------
def run_items(items, out_path, workers=6, budget_usd=None, spent_fn=None):
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for l in out_path.open():
            r = json.loads(l)
            if r['status'] in ('success', 'parse_error'):
                done.add(r['id'])
    todo = [it for it in items if it['id'] not in done]
    print(f'{out_path.name}: {len(done)} done, {len(todo)} to go', flush=True)
    lock = threading.Lock(); stop = threading.Event()

    def one(it):
        if stop.is_set():
            return None
        img = Image.open(it['image']).convert('RGB'); ref = V.tpl(it['target'])
        W, H = img.size
        rec = {'id': it['id'], 'model': it['model'], 'phase': it['phase']}
        for a in range(4):
            try:
                t0 = time.time()
                raw, u = VC.call(it['model'], [ref, img], system_prompt(it['model'], W, H), USER)
                rec.update(status='success', raw=raw, usage=u, latency=time.time() - t0, ts=time.time(),
                           truncated=VC.truncated(it['model'], u), cost=VC.cost(it['model'], u))
                if it['model'] == 'qwen3vl' and u.get('provider') not in (None, VC.QWEN_PROVIDER):
                    rec.update(status='provider_mismatch')
                break
            except Exception as e:
                rec.update(status='api_error', error=f'{type(e).__name__}: {e}'[:400])
                time.sleep(10 * 2 ** a)
        with lock:
            with out_path.open('a') as f:
                f.write(json.dumps(rec) + '\n')
        return rec

    n = 0
    with ThreadPoolExecutor(workers) as ex:
        for r in ex.map(one, todo):
            if r is None:
                continue
            n += 1
            if r['status'] != 'success' or n % 25 == 0:
                print(f"  [{n}/{len(todo)}] {r['id'][:70]} {r['status']} {r.get('error', '')[:120]}", flush=True)


# ---------------- reading and scoring (frozen before the first scored call) ----------------
NUM = r'-?\d+(?:\.\d+)?'


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def point_of(q):
    if not isinstance(q, dict):
        return None
    for kx, ky in (('cx', 'cy'), ('x', 'y'), ('center_x', 'center_y')):
        x, y = _num(q.get(kx)), _num(q.get(ky))
        if x is not None and y is not None:
            return x, y
    for key in ('cx', 'point', 'point_2d', 'center'):
        v = q.get(key)
        if isinstance(v, list) and len(v) == 2 and all(_num(a) is not None for a in v):
            return _num(v[0]), _num(v[1])
    v = q.get('bbox_2d') or q.get('bbox')
    if isinstance(v, list) and len(v) == 4 and all(_num(a) is not None for a in v):
        return (_num(v[0]) + _num(v[2])) / 2, (_num(v[1]) + _num(v[3])) / 2
    x = q.get('cx', q.get('x')); y = q.get('cy', q.get('y'))
    if isinstance(x, list) and len(x) == 1: x = x[0]
    if isinstance(y, list) and len(y) >= 1: y = y[0]
    x, y = _num(x), _num(y)
    return (x, y) if x is not None and y is not None else None


_LEN = re.compile(
    r'"c?x"\s*:\s*(' + NUM + r')\s*,\s*"c?y"\s*:\s*(' + NUM + r')'                      # "cx": a, "cy": b
    r'|"cx"\s*:\s*\[\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\]'                       # "cx": [a, b]
    r'|"(?:point_2d|point)"\s*:\s*\[\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\]'         # "point_2d": [a, b]
    r'|"bbox_2d"\s*:\s*\[\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\]')


def lenient(s):
    """Point centres in order of appearance from malformed, truncated or duplicate-key JSON."""
    pts = []
    for mt in _LEN.finditer(s):
        g = mt.groups()
        if g[0] is not None: pts.append((float(g[0]), float(g[1])))
        elif g[2] is not None: pts.append((float(g[2]), float(g[3])))
        elif g[4] is not None: pts.append((float(g[4]), float(g[5])))
        else: pts.append(((float(g[6]) + float(g[8])) / 2, (float(g[7]) + float(g[9])) / 2))
    return pts


class _Dup(Exception):
    pass


def _nodup(pairs):
    keys = [k for k, _ in pairs]
    if len(keys) != len(set(keys)):
        raise _Dup()
    return dict(pairs)


def parse(raw):
    """(points, declared_count, parse_ok). Strict JSON first; duplicate keys or malformed JSON fall back to the
    lenient scan, which reads every complete point in order of appearance."""
    s = (raw or '').strip()
    s = re.sub(r'^```(?:json)?\s*', '', s); s = re.sub(r'\s*```\s*$', '', s)
    obj, dup = None, False
    for cand in (s, s[s.find('{'):s.rfind('}') + 1] if s.find('{') >= 0 else ''):
        if not cand:
            continue
        try:
            obj = json.loads(cand, object_pairs_hook=_nodup); break
        except _Dup:
            dup = True; break
        except Exception:
            obj = None
    if isinstance(obj, dict) and not dup:
        sh = obj.get('shapes', [])
        pts = [p for p in (point_of(q) for q in (sh if isinstance(sh, list) else [])) if p is not None]
        return pts, obj.get('count'), True
    if isinstance(obj, list) and not dup:
        return [p for p in (point_of(q) for q in obj) if p is not None], None, True
    # Amendment 4 (2026-10-26, user-approved reader fix): some answers are written as a Python-like expression over
    # lists, e.g. [list A][:0] or [list B]; the frozen reader concatenated A and B although the model discarded A.
    # Such answers are now evaluated with a restricted literal evaluator. READER=frozen restores the frozen reader.
    if not dup and READER != 'frozen' and re.search(r'\]\s*(\[\s*-?\d*\s*:|if\b|or\b|and\b|\+)', s):
        try:
            a, b = s.find('{'), s.rfind('}')
            o2 = _safe_eval(s[a:b + 1])
            if isinstance(o2, dict) and isinstance(o2.get('shapes', []), list):
                return [p for p in (point_of(q) for q in o2.get('shapes', [])) if p is not None], o2.get('count'), True
        except Exception:
            pass
    m = re.search(r'"count"\s*:\s*(\d+)', s)
    return lenient(s), (int(m.group(1)) if m else None), False


READER = __import__('os').environ.get('READER', 'v2')


def _safe_eval(src):
    import ast
    src = re.sub(r'\btrue\b', 'True', re.sub(r'\bfalse\b', 'False', re.sub(r'\bnull\b', 'None', src)))
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.List): return [ev(e) for e in n.elts]
        if isinstance(n, ast.Dict): return {ev(k): ev(v) for k, v in zip(n.keys, n.values)}
        if isinstance(n, ast.Subscript):
            base = ev(n.value); sl = n.slice
            if isinstance(sl, ast.Slice):
                return base[slice(ev(sl.lower) if sl.lower else None, ev(sl.upper) if sl.upper else None)]
            return base[ev(sl)]
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]; out = vals[0]
            for v in vals[1:]:
                out = (out or v) if isinstance(n.op, ast.Or) else (out and v)
            return out
        if isinstance(n, ast.IfExp): return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add): return ev(n.left) + ev(n.right)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -ev(n.operand)
        raise ValueError(type(n).__name__)
    return ev(ast.parse(src, mode='eval'))


def to_px(model, pts, W, H):
    if V.MODELS[model]['coord'] == 'n1000':
        return [(x * W / 1000, y * H / 1000) for x, y in pts]
    return list(pts)


def match(pred, gt, tau):
    if not pred or not gt:
        return 0
    P, G = np.asarray(pred, float), np.asarray(gt, float)
    D = np.linalg.norm(P[:, None] - G[None], axis=2)
    r, c = linear_sum_assignment(np.where(D <= tau, D, 1e6))
    return int((D[r, c] <= tau).sum())


def score_record(it, rec):
    W, H = it['geo']['canvas']
    pts, declared, ok = parse(rec.get('raw'))
    P = to_px(it['model'], pts, W, H)
    tau = it['geo']['m_act'] / 2
    tp = match(P, it['points'], tau)
    return dict(tp=tp, n_pred=len(P), n_gt=len(it['points']), parse_ok=ok, declared=declared,
                truncated=bool(rec.get('truncated')))


def load_results(paths):
    recs = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for l in p.open():
            r = json.loads(l)
            if r['status'] == 'success':
                recs[r['id']] = r
    return recs


def pooled(rows):
    TP = sum(r['tp'] for r in rows); N = sum(r['n_gt'] for r in rows); NP = sum(r['n_pred'] for r in rows)
    R = TP / N if N else float('nan'); P = TP / NP if NP else float('nan')
    ok = (NP > 0) and R >= 0.9 and P >= 0.9
    F1 = 2 * TP / (N + NP) if (N + NP) else float('nan')
    return dict(R=R, P=P, F1=F1, pass_=bool(ok), k=len(rows), TP=TP, N=N, NP=NP,
                trunc=sum(r['truncated'] for r in rows))
