# REDP-X40: finding every instance of a legend symbol in real engineering drawings

40 real engineering-drawing images (screenshots without project information), each
with ONE target symbol defined by a crop from the drawing's own legend, and a point
annotation at the center of every instance of that symbol. Other symbols in the
drawing are not annotated.

| Subset | Drawings | Instances | Count per drawing | Long side (px) |
|---|---|---|---|---|
| sheet (d001-d020) | 20 | 747 | 6-116 | 2947-11484 |
| view (d021-d040) | 20 | 281 | 3-43 | 483-3520 |
| all | 40 | 1028 | 3-116 | 483-11484 |

## Files
- `redp_x40/images/dNNN.png` drawing (RGB; alpha composited on white)
- `redp_x40/templates/dNNN.png` legend crop of the target symbol
- `redp_x40/annotations/dNNN.json` `{width, height, template_size, count, points: [[x, y], ...]}` in image pixels
- `redp_x40/manifest.json` per-drawing summary and SHA-256 of every image
- `baseline_template_matching.py` classical baseline (below)

## Task and metrics
Input: the drawing and the template. Output: a list of instance centers.
- Count error: |predicted count - true count|, reported as MAE and mean |relative error|.
- Localization: F1 after one-to-one Hungarian matching of predicted and true points,
  with a match accepted within tau. Two tolerances:
  - tau_sym = 0.5 * sqrt(template_w * template_h) in native pixels;
  - tau_1600 = 30 px in the 1600-px space used for API models
    (= 30 / min(1, 1600 / long_side) native pixels).

## Classical baseline
Multi-scale (0.8-1.25) and 4-rotation normalized cross-correlation template matching
(OpenCV `TM_CCOEFF_NORMED`) with non-maximum suppression. The score threshold is
chosen leave-one-drawing-out (maximize mean F1 on the other 39 drawings); a
per-drawing oracle threshold is reported as an upper bound only.

## Splits used in the paper
Six drawings (d001, d004, d008, d010, d017, d019) were used for tuning; the other 34 drawings (660 instances) form the test set.

## License
CC BY-NC 4.0 (see `LICENSE-DATA` in the repository root). The images are partial screenshots without project-identifying information.
