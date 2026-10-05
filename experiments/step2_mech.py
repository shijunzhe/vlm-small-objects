#!/usr/bin/env python3
"""Step 2: FOV x upscale mechanism test (pre-registered in stok_experiment_design.md).
Cells: L1, L2, S1, S2 (upsampling the same pixels) for all models, plus L2n
(native resampling: more tokens and more information) for GPT and Claude.
Processes one drawing and one cell at a time to bound memory."""
import gc, sys
import mech as M

DIDS = ['d001', 'd004', 'd008', 'd010', 'd017', 'd019']
IFACE = {'gpt': ('px', 'px'), 'claude': ('px', 'px_claude'),
         'gemini': ('n1000', 'n1000'), 'gemma4': ('n1000', 'n1000')}
CELLS = {'L1': ('L', 1, False), 'L2': ('L', 2, False), 'S1': ('S', 1, False),
         'S2': ('S', 2, False), 'L2n': ('L', 2, True)}


def main(model, run=0, workers=6):
    M.C.load_env('../config.env')
    coord, decode = IFACE[model]
    cells = [c for c in CELLS if not (c == 'L2n' and model in ('gemini', 'gemma4'))]
    out = M.OUT / f'step2_{model}_run{run}.jsonl'
    for did in DIDS:
        ann, img, tpl = M.load(did)
        for cell in cells:
            fov, up, nat = CELLS[cell]
            ts = M.tiles_for(ann, img, fov, up, native=nat)
            tplz = M.scaled_template(tpl, ts[0]['z0'], up)
            jobs = [{'model': model, 'coord': coord, 'tpl': tplz, 'tile': t,
                     'meta': {'did': did, 'model': model, 'cell': cell, 'i': t['i'], 'j': t['j'],
                              'run': run, 'decode': decode}} for t in ts]
            M.run_jobs(jobs, out, key=lambda r: (r['did'], r['cell'], r['i'], r['j']), workers=workers)
            del jobs, ts
            gc.collect()
        del img
        gc.collect()


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0)
