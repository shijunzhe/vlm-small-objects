#!/usr/bin/env python3
"""X6: out-of-sample replication on REDP-10 (23 drawing-class items, 225 instances; configurations computed in advance,
nothing tuned on these drawings). usage: python3 redp10_run.py <cond A1|D|H> <vendor> [run]"""
import json, sys
from pathlib import Path
import mech as M
M.BENCH = Path('../data/redp10/redp10')
import gen_test  # noqa: E402  (installs new-vendor dispatch)
import main_test as T  # noqa: E402
import hybrid_verify as HV  # noqa: E402

ITEMS = sorted(p.stem for p in (M.BENCH / 'annotations').glob('r*.json'))
T.TEST = ITEMS
T.OUT = Path('results_redp10'); T.OUT.mkdir(exist_ok=True)
gen_test.OUT = T.OUT


def candidates(did):
    c = json.load(open(f'../data/redp10/results/tm_candidates/{did}.json'))['candidates']
    return sorted([x for x in c if x[2] >= HV.FLOOR], key=lambda x: -x[2])[:HV.CAP]


HV.candidates = candidates

if __name__ == '__main__':
    M.C.load_env('../config.env')
    cond, vendor = sys.argv[1], sys.argv[2]; run = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if cond == 'A1':
        T.whole(vendor, run)
    elif cond == 'D':
        T.tiles('D', vendor, run)
    elif cond == 'H':
        HV.M.OUT = T.OUT
        HV.run([vendor], ITEMS)
