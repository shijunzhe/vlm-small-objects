#!/usr/bin/env python3
"""X6b: the zoom agent (X4 protocol, unchanged) on REDP-10. usage: python3 redp10_agent.py <vendor> [run]"""
import sys
from pathlib import Path
import mech as M
M.BENCH = Path('../data/redp10/redp10')
import agent as AG  # noqa: E402
AG.OUT = Path('results_redp10/agent'); AG.OUT.mkdir(parents=True, exist_ok=True)
ITEMS = sorted(p.stem for p in (M.BENCH / 'annotations').glob('r*.json'))
if __name__ == '__main__':
    M.C.load_env('../config.env')
    AG.run(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0, ITEMS)
