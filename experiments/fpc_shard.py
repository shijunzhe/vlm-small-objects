#!/usr/bin/env python3
"""Run fpc_vlm-style D condition for one vendor on a shard of the not-yet-done FPC items, writing to a shard dir.
usage: python3 fpc_shard.py <bench|sheets> D <vendor> <i> <k>; merge with cat afterwards."""
import sys, json
from pathlib import Path
sys.argv = [sys.argv[0]] + sys.argv[1:]
which, cond, v, i, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
sys.argv = [sys.argv[0], which, cond, v]
import fpc_vlm as F
main = Path(f'results_fpc/{which}/{cond}_{v}_run0.jsonl')
done = {json.loads(l)['did'] for l in open(main) if json.loads(l)['status'] in ('success', 'parse_error')} if main.exists() else set()
rem = [d for d in F.ITEMS if d not in done]
F.T.TEST = rem[i::k]
F.T.OUT = Path(f'results_fpc/{which}_shard{i}'); F.T.OUT.mkdir(exist_ok=True); F.gen_test.OUT = F.T.OUT
F.M.C.load_env('../config.env')
F.T.tiles(cond, v, 0)
