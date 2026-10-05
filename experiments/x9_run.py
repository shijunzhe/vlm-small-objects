#!/usr/bin/env python3
"""X9 confound controls (prereg v3.3). usage: python3 x9_run.py <D|A1|A1nat> <vendor> [run]"""
import sys, math
import gen_test as GT   # installs dispatch for new vendors
import main_test as T
T.TILE_IFACE.update({'gpt56m': ('px', 'px'), 'sonnet55t': ('px', 'px'), 'gem38t': ('n1000', 'n1000')})
T.XTILE.update({'gpt56m', 'sonnet55t'})
T.FOV_K.update({'gem38t': 2.5 * 16 / math.sqrt(1090)})
T.WORKERS.update({'gpt56m': 6, 'sonnet55t': 6, 'gem38t': 8})
if __name__ == '__main__':
    import mech as M
    M.C.load_env('../config.env')
    cond, v = sys.argv[1], sys.argv[2]; run = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    if cond == 'D': T.tiles('D', v, run)
    elif cond == 'A1': T.whole(v, run)
    elif cond == 'A1nat': GT.whole_native(v, run)
