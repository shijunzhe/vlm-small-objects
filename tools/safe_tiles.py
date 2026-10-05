#!/usr/bin/env python3
"""Safe tiling for vision-language models (the rule of Corollary 1 in the paper).

Given an image, the size of the target objects, and the documented image interface of a model,
plan views such that

  * every view gives a target at least as many tokens per side as the whole image would (S_view >= S_whole),
    and at least S_star tokens per side when that is more;
  * consecutive views overlap by one target margin e = max(target side, 2 * tau), so every target lies
    entirely inside some view.

Under the paper's monotonicity assumptions such a decomposition cannot lower expected recall relative
to sending the whole image, for any model that satisfies them, and its token cost approaches the
minimum over all protected decompositions as the image grows.

Only the standard library is needed for planning; Pillow is needed to cut views.

Example
-------
    # GPT-style interface: 32 px tokens, at most 8192 tokens per image
    python safe_tiles.py plan --image-size 9000 6000 --target 30 30 --interface res --token 32 --budget 8192
    # Gemini-style fixed grid: every image becomes about 1090 tokens
    python safe_tiles.py plan --image-size 9000 6000 --target 30 30 --interface grid --budget 1090
    # cut the views of a real image
    python safe_tiles.py cut drawing.png --target 30 30 --interface res --token 32 --budget 8192 --out views/

After the model answers each view, map its points back with `merge` (see merge_points below):
points from all views are translated to image coordinates and points closer than tau are merged.
"""
import argparse, json, math, os


def whole_image_S(W, H, m, interface, token=None, budget=None, cap=None):
    """Tokens per target side when the whole W x H image is sent through the interface."""
    if interface == 'grid':                       # the image is mapped to about budget tokens, longest side fits sqrt(budget)
        return m * math.sqrt(budget) / max(W, H)
    s = min(1.0, math.sqrt(budget * token * token / (W * H)))   # resolution-scaled: resize to fit the token budget
    if cap:
        s = min(s, cap / max(W, H))
    return m * s / token


def plan(W, H, w, h, interface='res', token=32, budget=2500, cap=None, S_star=1.5, tau=None):
    """Return the view geometry of the safe rule. Coordinates are native image pixels."""
    m = math.sqrt(w * h)
    tau = m / 2 if tau is None else tau
    ex, ey = max(w, 2 * tau), max(h, 2 * tau)                 # overlap margins
    Sw = whole_image_S(W, H, m, interface, token, budget, cap)
    S = max(S_star, Sw)                                        # never fewer tokens per target than the whole image
    lam = (S / m) ** 2                                         # tokens per native pixel
    sigma = math.sqrt(lam * ex * ey / budget)                  # fraction of a view's side spent on one margin
    if sigma >= 1:
        raise ValueError(f'target needs more than one call per target at S={S:.2f}; raise the budget or lower S_star')
    if interface == 'grid':
        a = m * math.sqrt(budget) / S                          # square view; the vendor maps it to the full grid
        ax = ay = a; scale = 1.0; tokens_per_view = budget; canvas = None
    else:
        jx, jy = math.floor(ex * S / (m * sigma)), math.floor(ey * S / (m * sigma))
        if cap:
            jx, jy = min(jx, cap // token), min(jy, cap // token)
        while jx * jy > budget:
            jx, jy = (jx - 1, jy) if jx >= jy else (jx, jy - 1)
        scale = S * token / m                                  # resample factor applied to each view
        ax, ay = jx * m / S, jy * m / S
        tokens_per_view = jx * jy; canvas = (jx * token, jy * token)
    px, py = ax - ex, ay - ey                                  # pitch: views overlap by one margin
    nx = max(1, math.ceil(max(W - ex, 1e-9) / px)); ny = max(1, math.ceil(max(H - ey, 1e-9) / py))
    if Sw >= S_star and W <= ax and H <= ay:
        nx = ny = 1
    views = []
    for j in range(ny):
        for i in range(nx):
            x0, y0 = i * px, j * py
            views.append({'i': i, 'j': j, 'box': [round(x0), round(y0), round(min(W, x0 + ax)), round(min(H, y0 + ay))]})
    cmin = max(W - ex, 0) * max(H - ey, 0) * lam / (1 - sigma) ** 2
    return {'S_whole': Sw, 'S': S, 'sigma': sigma, 'tau': tau, 'scale': scale, 'view_size': [ax, ay], 'canvas': canvas,
            'pitch': [px, py], 'n_views': len(views), 'tokens': len(views) * tokens_per_view,
            'tokens_lower_bound': cmin, 'views': views}


def cut(path, p, out):
    """Write one PNG per view (padded to the full canvas so that the vendor does not resize it differently)."""
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(path).convert('RGB'); os.makedirs(out, exist_ok=True)
    for v in p['views']:
        crop = img.crop(tuple(v['box']))
        if p['scale'] != 1.0:
            crop = crop.resize((max(1, round(crop.width * p['scale'])), max(1, round(crop.height * p['scale']))), Image.LANCZOS)
        if p['canvas']:
            can = Image.new('RGB', p['canvas'], 'white'); can.paste(crop, (0, 0)); crop = can
        v['file'] = os.path.join(out, f"view_{v['j']:02d}_{v['i']:02d}.png"); crop.save(v['file'])
    json.dump(p, open(os.path.join(out, 'plan.json'), 'w'), indent=1)
    return p


def merge_points(p, answers):
    """answers: {(i, j): [(x, y), ...]} in the pixel frame of each sent view. Returns image-frame points,
    with points closer than tau merged."""
    pts = []
    for v in p['views']:
        x0, y0 = v['box'][:2]
        for x, y in answers.get((v['i'], v['j']), []):
            pts.append((x0 + x / p['scale'], y0 + y / p['scale']))
    kept = []
    for q in pts:
        if all((q[0] - k[0]) ** 2 + (q[1] - k[1]) ** 2 > p['tau'] ** 2 for k in kept):
            kept.append(q)
    return kept


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('plan', 'cut'):
        a = sub.add_parser(name)
        if name == 'cut':
            a.add_argument('image'); a.add_argument('--out', default='views')
        else:
            a.add_argument('--image-size', type=int, nargs=2, required=True, metavar=('W', 'H'))
        a.add_argument('--target', type=float, nargs=2, required=True, metavar=('w', 'h'), help='target box in native pixels')
        a.add_argument('--interface', choices=['res', 'grid'], default='res')
        a.add_argument('--token', type=int, default=32, help='token side in pixels (resolution-scaled interfaces)')
        a.add_argument('--budget', type=int, default=2500, help='maximum tokens per image (grid: tokens per image)')
        a.add_argument('--cap', type=int, default=None, help='maximum image side in pixels, if the vendor has one')
        a.add_argument('--s-star', type=float, default=1.5, help='tokens per target side to aim for')
    a = ap.parse_args()
    if a.cmd == 'cut':
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        W, H = Image.open(a.image).size
    else:
        W, H = a.image_size
    p = plan(W, H, *a.target, interface=a.interface, token=a.token, budget=a.budget, cap=a.cap, S_star=a.s_star)
    if a.cmd == 'cut':
        cut(a.image, p, a.out)
    summary = {k: v for k, v in p.items() if k != 'views'}
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
