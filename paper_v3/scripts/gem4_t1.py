#!/usr/bin/env python3
"""2026-10-27: Gemini 4 Argon added to Table 1 (user approval). Same three conditions and code paths as the
Table 1 models: A1 whole image, D (fixed-grid FOV, symbol 2.5 tokens), CS (the rule, S* = 1.5).
usage: python3 gem4_t1.py <redp|fpc> <A1|D|CS|smoke>
fpc runs only the 60 k3 mosaics used in Table 1."""
import sys, glob, json
from pathlib import Path
bench, cond = sys.argv[1], sys.argv[2]
if bench == 'fpc':
    sys.argv = ['fpc_vlm.py', 'sheets']
    import fpc_vlm  # noqa: F401  sets M.BENCH, T.OUT (results_fpc/sheets)
    import main_test as T
    T.TEST = sorted(Path(p).stem for p in glob.glob(str(Path.cwd().parent / 'data/fpc/sheets/annotations/*_k3.json')))
else:
    sys.argv = ['gen_test.py']
    import gen_test  # noqa: F401  sets T.OUT = results_gen, dispatch to gen_common
    import main_test as T
import mech as M
M.C.load_env(str(Path.cwd().parent / 'config.env'))
if cond == 'smoke':
    T.TEST = T.TEST[:1]; T.OUT = Path('/tmp/claude-0/-home-claude/a281cdb4-95de-596d-8c7d-0f83ea38ca9a/scratchpad/gem4_smoke'); T.OUT.mkdir(exist_ok=True)
    T.whole('gem4', 0)
elif cond == 'A1':
    T.whole('gem4', 0)
elif cond == 'D':
    T.tiles('D', 'gem4', 0)
elif cond == 'CS':
    sys.argv = ['cs_rule.py', bench]
    import importlib, cs_rule as CR
    if bench == 'fpc':
        CR.BENCH_ARG = 'fpc'; M.BENCH = Path.cwd().parent / 'data/fpc/sheets'; CR.OUT = Path.cwd() / 'results_cs' / 'fpc'
    CR.run('gem4')
