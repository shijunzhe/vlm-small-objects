#!/usr/bin/env python3
"""VLM runs on FPC-300 (bench) and FPC-sheets (sheets), same protocols as REDP-X40.
usage: python3 fpc_vlm.py <bench|sheets> <A1|D|AG> <vendor> [run]"""
import json, sys, math
from pathlib import Path
import mech as M
which = sys.argv[1]
M.BENCH = Path(f'../data/fpc/{which}')
import gen_test  # noqa: E402  dispatch for second-round vendors
import main_test as T  # noqa: E402
import x_small as XS  # noqa: E402
XS.ZCAP = 12000
ITEMS = sorted(p.stem for p in (M.BENCH / 'annotations').glob('*.json'))
T.TEST = ITEMS
T.OUT = Path(f'results_fpc/{which}'); T.OUT.mkdir(parents=True, exist_ok=True)
gen_test.OUT = T.OUT
T.WORKERS.update({'gpt': 8, 'claude': 10, 'qwen3vl': 12, 'gemini': 16, 'gemma4': 12, 'sonnet55': 10, 'gpt56': 8, 'gem38': 16, 'gem38lo': 16, 'gem38uh': 16})  # FPC_WORKERS: small 1000 px images, more parallel calls
if __name__ == '__main__':
    M.C.load_env('../config.env')
    cond, v = sys.argv[2], sys.argv[3]; run = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    if cond == 'A1': T.whole(v, run)
    elif cond == 'D': T.tiles('D', v, run)
    elif cond == 'AG':
        import agent as AG
        AG.OUT = T.OUT / 'agent'; AG.OUT.mkdir(exist_ok=True)
        AG.run(v, run, ITEMS)
