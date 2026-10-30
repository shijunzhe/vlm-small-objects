import sys, math
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import vspec as V
V.MODELS['gpt61'] = dict(V.MODELS['gpt61'], maxside=2896)
LOADS = {'gpt61': [8,16,32,64,128,256,324], 'gpt56': [8,16,32,64,128,256,324], 'opus55': [8,16,32,64,128,196],
         'gem4': [8,16,32,42], 'gem38': [8,16,32,42], 'qwen3vl': [8,16,32,64]}
PRICE = {'gpt61': (2.0, 10.0), 'gpt56': (4.0, 20.0), 'opus55': (4.0, 20.0), 'gem38': (0.75, 3.75), 'gem4': (2.0, 10.0), 'qwen3vl': (0.5, 2.0)}
tot = 0; calls = 0
for m, ns in LOADS.items():
    t = V.MODELS[m]['t']; c_m = 0
    for n in ns:
        Wu, Hu = V.extent(n); cv = V.canvas_for(m, Wu, Hu, 1.5 * t)
        assert cv is not None, (m, n)
        tin = V.expected_tokens(m, *cv) + 300 + 100
        tout = 25 * n + 200 + (3000 if m in ('gpt61', 'gem4') else 0)
        pi, po = PRICE[m]; c = 12 * (tin * pi + tout * po) / 1e6; c_m += c; calls += 12
        print(m, n, cv, tin, tout, round(c, 2))
    # feasibility of next load
    print(m, 'total $%.2f' % c_m); tot += c_m
print('calls', calls, 'total $%.2f' % tot)
