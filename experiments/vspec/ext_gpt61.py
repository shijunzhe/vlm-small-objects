#!/usr/bin/env python3
"""Amendment 3 (2026-10-26, user request): GPT-6.1 capacity beyond the v3 grid at x = 2 (n = 384, 512; 16 images,
P2 scenes). Bisection per v3 5.3 if a bracket with b/a >= 1.5 appears. Results go to results/phase2.jsonl."""
import json, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import vrun as R, phase2 as P2
OUT = P2.OUT
m, x = 'gpt61', 2.0
S = P2.scale(m, x)
todo = []
for n in (384, 512):
    todo += P2.items_for(m, x, n, extra_renders=[(m, S)])
P2.save_items(todo); print(len(todo), 'items', sorted({it['n'] for it in todo}), [it['geo']['canvas'] for it in todo[:1]], [it['exp_img_tokens'] for it in todo[:1]])
R.run_items(todo, OUT / 'phase2.jsonl', workers=8)
st = P2.state()[(m, x)]
pooled = {n: R.pooled([r for _, r in v]) for n, v in st.items()}
a, b = P2.bracket(pooled); print('bracket', a, b)
if a and b and b / a >= 1.5 and b > 256:
    import math
    n = int(round(math.sqrt(a * b) / 2) * 2)
    its = P2.items_for(m, x, n, extra_renders=[(m, S)]); P2.save_items(its)
    R.run_items(its, OUT / 'phase2.jsonl', workers=8)
