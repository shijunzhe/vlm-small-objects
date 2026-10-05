import json, sys, time
from pathlib import Path
sys.path.insert(0, '../data/redp40')
import baseline_template_matching as TM
which = sys.argv[1]
TM.BENCH = Path(f'../data/fpc/{which}'); OUT = Path(f'../data/fpc/tm_{which}'); OUT.mkdir(exist_ok=True)
for p in sorted((TM.BENCH / 'annotations').glob('*.json')):
    o = OUT / f'{p.stem}.json'
    if o.exists(): continue
    t0 = time.time(); r = TM.detect(p.stem); o.write_text(json.dumps(r)); print(p.stem, len(r['candidates']), f'{time.time()-t0:.0f}s', flush=True)
