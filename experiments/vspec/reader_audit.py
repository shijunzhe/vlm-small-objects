#!/usr/bin/env python3
"""Reader audit (2026-10-26, after coauthor review): classify every record read by the lenient path and,
for Opus pseudo-code answers, evaluate the expression semantically (sensitivity only; the frozen reader stays primary)."""
import ast, json, re, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import vrun as R

def safe_eval(src):
    src = re.sub(r'\btrue\b', 'True', re.sub(r'\bfalse\b', 'False', re.sub(r'\bnull\b', 'None', src)))
    tree = ast.parse(src, mode='eval')
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant): return n.value
        if isinstance(n, ast.List): return [ev(e) for e in n.elts]
        if isinstance(n, ast.Dict): return {ev(k): ev(v) for k, v in zip(n.keys, n.values)}
        if isinstance(n, ast.Subscript):
            base = ev(n.value); s = n.slice
            if isinstance(s, ast.Slice):
                return base[slice(ev(s.lower) if s.lower else None, ev(s.upper) if s.upper else None)]
            return base[ev(s)]
        if isinstance(n, ast.BoolOp):
            vals = [ev(v) for v in n.values]
            out = vals[0]
            for v in vals[1:]:
                out = (out or v) if isinstance(n.op, ast.Or) else (out and v)
            return out
        if isinstance(n, ast.IfExp): return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add): return ev(n.left) + ev(n.right)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub): return -ev(n.operand)
        raise ValueError(type(n).__name__)
    return ev(tree)

def semantic_points(raw):
    s = raw.strip(); a, b = s.find('{'), s.rfind('}')
    obj = safe_eval(s[a:b + 1])
    sh = obj.get('shapes', []) if isinstance(obj, dict) else []
    return [p for p in (R.point_of(q) for q in sh) if p is not None]

if __name__ == '__main__':
    files = [('items_p1.json', 'results/phase1.jsonl'), ('items_p2.json', 'results/phase2.jsonl'), ('items_val.json', 'results/validation.jsonl'),
             ('items_p1_gem4.json', 'results/phase1_gem4.jsonl'), ('items_p2_gem4.json', 'results/phase2_gem4.jsonl'), ('items_val_gem4.json', 'results/validation_gem4.jsonl')]
    out = []
    for itf, rf in files:
        its = {t['id']: t for t in json.load(open(__import__('os').path.dirname(__import__('os').path.abspath(__file__)) + '/' + itf))}
        for iid, r in R.load_results([__import__('os').path.dirname(__import__('os').path.abspath(__file__)) + '/' + rf]).items():
            pts, dec, ok = R.parse(r['raw'])
            if ok: continue
            it = its[iid]; s0 = R.score_record(it, r)
            kind = 'truncated' if r.get('truncated') else ('pseudo_code' if re.search(r'\]\s*(\[:|if |or |\+)', r['raw']) else 'malformed')
            rec = dict(id=iid, model=r['model'], kind=kind, declared=dec, n_read=len(pts), tp=s0['tp'], n_gt=s0['n_gt'])
            if kind == 'pseudo_code':
                try:
                    sp = semantic_points(r['raw']); W, H = it['geo']['canvas']
                    P = R.to_px(it['model'], sp, W, H); rec.update(n_semantic=len(sp), tp_semantic=R.match(P, it['points'], it['geo']['m_act'] / 2))
                except Exception as e:
                    rec.update(semantic_error=str(e)[:100])
            out.append(rec)
    json.dump(out, open(__import__('os').path.dirname(__import__('os').path.abspath(__file__)) + '/results/reader_audit.json', 'w'), indent=1)
    from collections import Counter
    print(Counter((o['model'], o['kind']) for o in out))
    for o in out:
        if o['kind'] == 'pseudo_code' or (o['kind'] == 'malformed' and o['declared'] is not None and o['n_read'] > o['declared']):
            print(o)
