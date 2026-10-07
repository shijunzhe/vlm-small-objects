# Why VLMs Miss Small Objects, and When Zooming In Is Safe

Code, data, and model outputs for the paper by Junzhe Shi, Yuan Gan, and Shida Jiang (arXiv link to be added). It contains:

* a stand-alone tool that plans the paper's safe decomposition for any image and model interface (`tools/safe_tiles.py`);
* every script that sent images to a model, scored the answers, or computed a number, table, or figure in the paper (`experiments/`);
* every raw model answer the paper uses, with the request metadata and token usage (`experiments/results*/`, `experiments/syn*/`);
* the REDP-X40 benchmark (`data/redp40/`), the FloorPlanCAD-derived items (`data/fpc/`), and the FSC-147 item list and annotations used (`data/fsc147/`).

All numbers in the paper can be recomputed from the shipped answers without any API key (see "Reproduce the paper's numbers"). Rerunning the model calls needs vendor keys and costs money.

## The idea in one paragraph

A vision-language model resizes an image to a token budget and cuts it into visual tokens of a fixed size. Two quantities of this interface describe every call: **S**, the number of visual tokens across one side of a target object, and **L**, the content the call must cover and enumerate. Tiling raises S and lowers L, a high-resolution mode raises S but keeps L, and a zoom tool sets both region by region. Under weak monotonicity assumptions about the model (more tokens per target does not hurt, less content does not hurt), a decomposition cannot lower expected recall when each view gives every target at least the whole image's S and the views overlap by one target margin. `tools/safe_tiles.py` implements the paper's rule for such a decomposition, whose cost approaches the lower bound as the image grows when the ideal views fit the vendor's limits.

## Quick start: safe tiling for your own images

```bash
pip install pillow
# plan only: a 9000 x 6000 px drawing, 30 px symbols, an interface with 32 px tokens and 8,192 tokens per image
python tools/safe_tiles.py plan --image-size 9000 6000 --target 30 30 --interface res --token 32 --budget 8192
# a fixed-grid interface that maps every image to about 1,090 tokens
python tools/safe_tiles.py plan --image-size 9000 6000 --target 30 30 --interface grid --budget 1090
# cut the views of a real image into views/ (one PNG per view plus plan.json)
python tools/safe_tiles.py cut drawing.png --target 30 30 --interface res --token 32 --budget 8192 --out views/
```

Send each view to the model, then map the answers back with `safe_tiles.merge_points(plan, answers)`. It translates the points to image coordinates and merges points closer than the matching tolerance. The planner gives the same views as the experiment code (`experiments/cs_rule.py`) on all REDP-X40 test drawings for all four models in the paper.

Measure your model's interface first. Read the billed input tokens for blank images of several sizes to find the token side and the budget, and use a one-dot image to find its coordinate frame (`experiments/calib_tokens.py`, `experiments/gen_calib.py`). Then test monotonicity on nested image pairs, as in the paper's controlled battery (`experiments/syn_assume.py`).

## Repository layout

```
tools/safe_tiles.py        stand-alone planner for the safe decomposition rule
experiments/               all experiment, scoring, and analysis code (run from this directory)
  vlm_api/                 model-call layer shared by the experiments (vendor SDK wrappers, JSON parsing)
  results*/, syn*/         raw model answers (.jsonl) and scores (.json)
data/redp40/               REDP-X40: 40 drawings, legend templates, point annotations
data/fpc/                  FloorPlanCAD-derived items (FPC-300, FPC-sheets): images, references, annotations, build scripts
data/fsc147/               FSC-147 item list, annotations of those items, model answers, scores (no images)
paper/generated/           macros and tables used by the paper (the reproduction target)
paper/figures/             figures used by the paper
reproduce.sh               recompute every paper number from the shipped answers
config.env.example         names of the API keys needed to rerun model calls
```

## Reproduce the paper's numbers (no API calls)

```bash
pip install -r requirements.txt
bash reproduce.sh
```

`reproduce.sh` reruns every scoring and analysis script on the shipped answers, regenerates `paper/generated/` and `paper/figures/`, and compares the regenerated macros and tables with the shipped ones. With the pinned packages they match byte for byte. Scripts that need data that is not distributed (REDP-10 annotations) are skipped, and their shipped scores are used.

## Where each result comes from

All scripts live in `experiments/` and read and write paths relative to it.

| Paper result | Scripts | Answers and scores |
|---|---|---|
| Measured interfaces (token side, budget, coordinate frame) | `calib_tokens.py`, `gen_calib.py`, `step1_coord_calib.py`, `step1b_gemini.py`, `ceilings.py`, `gen_ceilings.py` | `calib_tokens.json`, `gen_calib*.json`, `ceilings.json`, `gen_ceilings.json` |
| Direct tests of the assumptions on controlled images (Table 2) | `syn_assume.py build / run <model> / score` | `syn_assume/results/`, `syn_assume/assume_scores.json` |
| False outputs and the F1 exchange rate | `theory_v8/output_form.py`, `theory_v8/f1_union.py` | `theory_v8/*.json` |
| Bits carried per answer (capacity theorem) | `theory_v10/info_carried.py` | `theory_v10/info_carried.json` |
| Synthetic sweeps of S, content, area, clutter, upsampling (Figure 4) | `syn.py build / run <model>`, `syn_analysis.py`, `t1_syn.py` | `syn/results/`, `syn/analysis_robust.json`, `results_x/t1_syn.json` |
| REDP-X40, first-generation models (whole drawing, tiles, vendor modes, masks, propose and verify) | `main_test.py`, `x_small.py`, `x3_fixedgrid.py`, `x4_vendor_highres.py`, `x4b_gemini_high.py`, `hybrid_verify.py`, `analyze_test.py` | `results_test/` |
| REDP-X40, second-generation models | `gen_test.py`, `gen_common.py`, `analyze_gen.py` | `results_gen/` |
| Mechanism checks on the tuning set | `phase0.py`, `mech.py`, `step2_mech.py`, `step2_score.py`, `tile_runner.py`, `score_tiles.py` | `results_mech/`, `results/` |
| FloorPlanCAD (FPC-300, FPC-sheets) | `fpc_vlm.py`, `fpc_shard.py`, `fpc_score.py`, `fpc_geom.py`, `fpc_sratio.py`, `t2_fpc.py` | `results_fpc/`, `results_x/t2_nested_fpc.json` |
| Nested content, packed tiles, vendor high-resolution modes | `mosaic.py`, `x_new_score.py`, `x9_run.py`, `x9_score.py`, `x4q_score.py`, `t2_nested.py` | `results_x/` |
| Zoom agents | `agent.py`, `redp10_agent.py` | `results_x/agent/`, `results_redp10/agent/` |
| Classical and learned references | `../data/redp40/baseline_template_matching.py`, `learned_baselines.py`, `lb_modal.py`, `lb_score2.py`, `fsc_counters.py` | `../data/redp40/results/`, `results_x/learned*/` |
| FSC-147 | `fsc_run.py` | `../data/fsc147/` |
| Held-out drawings with library references (REDP-10; drawings not released) | `redp10_run.py`, `redp10_score.py` | `results_redp10/` (scores and answers only) |
| Every implied ordering in the other experiments (Figure 2) | `law_data.py`, `theory_check.py`, `theory_check2_main.py`, `theory_check2_fsc.py`, `merge_tc.py`, `fig_theory.py` | `results_law/theory_check_final.json` |
| The safe rule on construction drawings (Table 3) | `cs_rule.py <redp\|fpc> <model> [plan]`, `cs_analyze.py` | `results_cs/` |
| Coupling, locality, annotation audit, limits | `locality_check.py`, `annot_audit.py`, `limits_analysis.py`, `ceiling_check_dev.py` | `results_x/` |
| All macros, tables, and figures in the paper | `paper2_numbers.py`, `paper2_numbers_x.py`, `paper2_figures.py`, `paper2_figures_x.py`, `fig_intro.py` (Figure 1) | `../paper/generated/`, `../paper/figures/` |

`law_fit.py`, `law_frontier.py`, `law_m3.py`, and `feas_law.py` are the exploratory curve fits of one analysis plan. The paper does not report them as laws; Appendix F lists them with the other deviations.

## Raw answers

Each line of a `.jsonl` file is one model call. Common fields are:

* `did`, `id`: the drawing or item;
* `model`, `cond`, `run`: the model, the condition, and the run index;
* `i`, `j`: the view index;
* `status`;
* `raw`: the model's text answer;
* `shapes`: the parsed points;
* `usage`: billed tokens, as returned by the vendor.

Tile records also carry the view geometry: `origin_z0`, `z0`, `up`, `core_px`, and, for the safe rule, `S`, `Sw`, `sigma`, `chi`. Scoring matches points one to one by a maximum matching within the tolerance tau, half the geometric mean of the template sides.

## Rerunning model calls

Copy `config.env.example` to `config.env` in the repository root and fill in the keys you need. Keys are read from that file or the environment and are never printed or logged. Each experiment script is resumable, so rerunning a command skips calls already in its output file. Vendors change models without notice; the shipped answers were collected in September and October 2026.

## Data and licenses

* **Code**: MIT License (`LICENSE`).
* **REDP-X40**: drawings, templates, and annotations, released for research under `LICENSE-DATA`. The drawings are partial screenshots without project-identifying information. See `data/redp40/README.md`.
* **FloorPlanCAD**-derived images and annotations are CC BY-NC 4.0, as FloorPlanCAD is (`data/fpc/README.md`).
* **FSC-147** images are not redistributed. We include the item list and the original annotations of the 150 images used, for scoring.
* **REDP-10**: the ten held-out drawings are used for evaluation only and are not released. Their model answers and scores are included.

## Citation

```bibtex
@misc{shi2026zoomin,
  title  = {Why {VLMs} Miss Small Objects, and When Zooming In Is Safe},
  author = {Shi, Junzhe and Gan, Yuan and Jiang, Shida},
  year   = {2026},
  note   = {arXiv preprint},
  url    = {https://github.com/shijunzhe/vlm-small-objects}
}
```
