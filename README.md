# Why VLMs Miss Small Objects, and When Zooming In Is Safe

[![arXiv](https://img.shields.io/badge/arXiv-2610.09313-b31b1b.svg)](https://arxiv.org/abs/2610.09313)

Code, data, and model outputs for the paper by Junzhe Shi, Yuan Gan, and Shida Jiang ([arXiv:2610.09313](https://arxiv.org/abs/2610.09313); this release matches version 3). It contains:

* a stand-alone tool that plans the paper's safe split for any image and model interface (`tools/safe_tiles.py`);
* every script that sent images to a model, scored the answers, or computed a number, table, or figure in the paper (`experiments/`, `paper_v3/scripts/`);
* every raw model answer the paper uses, with the request metadata and token usage (`experiments/results*/`, `experiments/syn*/`, `experiments/vspec/results/`);
* the LegendSpot benchmark (`data/redp40/`), the FloorPlanCAD-derived floor-plan blocks and mosaics (`data/fpc/`), and the FSC-147 item list and annotations used (`data/fsc147/`);
* the LaTeX source of version 3 (`paper_v3/`).

All numbers in the paper can be recomputed from the shipped answers without any API key (see "Reproduce the paper's numbers"). Rerunning the model calls needs vendor keys and costs money.

## The idea in one paragraph

A vision-language model resizes an image to a token budget and cuts it into visual tokens of a fixed size. VLMs miss small objects for two reasons that a caller can estimate before any call: an object gets too few visual tokens (**S**, the tokens across one side of a target), and a call must search and report too much (**L**, the content of one call). Raising S by a factor a raises the minimum tokens needed to see an image a^2-fold; splitting at the same scale lowers L at a cost at most linear in the number of views. Under two monotonicity assumptions about the model (more tokens per target do not hurt, less content does not hurt), a split cannot lower expected recall when no view zooms out relative to the whole image and views overlap by one object, and neither part can be dropped. Every such split pays a minimum token cost, and `tools/safe_tiles.py` implements the paper's rule, which pays it up to an explicit factor.

## Quick start: safe splitting for your own images

```bash
pip install pillow
# plan only: a 9000 x 6000 px image, 30 px targets, an interface with 32 px tokens and 8,192 tokens per image
python tools/safe_tiles.py plan --image-size 9000 6000 --target 30 30 --interface res --token 32 --budget 8192
# a fixed-grid interface that maps every image to about 1,090 tokens
python tools/safe_tiles.py plan --image-size 9000 6000 --target 30 30 --interface grid --budget 1090
# cut the views of an image into views/ (one PNG per view plus plan.json)
python tools/safe_tiles.py cut drawing.png --target 30 30 --interface res --token 32 --budget 8192 --out views/
```

Send each view to the model, then map the answers back with `safe_tiles.merge_points(plan, answers)`. It translates the points to image coordinates and keeps all of them, as the paper does; duplicates in the overlaps count against precision (`merge=True` merges points closer than the matching tolerance). The planner gives the same views as the experiment code (`experiments/cs_rule.py`) on all LegendSpot test drawings for GPT-6.1, Claude Opus 5.5, Gemini 3.8 and Qwen3-VL.

Measure your model's interface first. Read the billed input tokens for blank images of several sizes to find the token side and the budget, and use a one-dot image to find its coordinate frame (`experiments/calib_tokens.py`, `experiments/gen_calib.py`). Then test monotonicity with a few pairs of calls on the same region at different token counts, or on nested image pairs as in the paper's controlled test (`experiments/syn_assume.py`).

## Repository layout

```
tools/safe_tiles.py        stand-alone planner for the safe splitting rule
experiments/               all experiment, scoring, and analysis code (run from this directory)
  vlm_api/                 model-call layer shared by the experiments (vendor SDK wrappers, JSON parsing)
  vspec/                   newest-generation measurements of the S that recognition needs (S90)
  loadsweep/               load at S = 1.5 (rule-of-thumb table); images rebuild with `ls_run.py build`
  results*/, syn*/         raw model answers (.jsonl) and scores (.json)
data/redp40/               LegendSpot: 40 drawings, legend templates, point annotations
data/fpc/                  floor-plan blocks and mosaics from FloorPlanCAD: images, references, annotations, build scripts
data/fsc147/               FSC-147 item list, annotations of those items, model answers, scores (no images)
paper/                     macros, tables, and figures of the scoring pipeline (input to paper_v3)
paper_v3/                  LaTeX source, generated macros and tables, figures, and scripts of version 3
reproduce.sh               recompute every score from the shipped answers
reproduce_v3.sh            recompute every macro and table of version 3 from those scores
config.env.example         names of the API keys needed to rerun model calls
```

## Names

The code keeps the internal names used during the project and in versions 1 and 2 of the paper:

| Paper (version 3) | Code and versions 1 and 2 |
|---|---|
| LegendSpot | REDP-X40 (`redp40`, `redp`) |
| Library-10 | REDP-10 (`redp10`) |
| floor-plan blocks | FPC-300 (`fpc`, `bench`) |
| floor-plan mosaics | FPC-sheets (`sheets`) |
| controlled images | synthetic images (`syn`, `syn_assume`) |

## Reproduce the paper's numbers (no API calls)

```bash
pip install -r requirements.txt
POINT_XY=1 bash reproduce.sh
bash reproduce_v3.sh
```

`reproduce.sh` reruns every scoring and analysis script on the shipped answers, regenerates `paper/generated/` and `paper/figures/`, and compares them with the shipped files. `POINT_XY=1` selects the reader that version 3 uses throughout: it keeps points written with `x`/`y` keys and points written as lists, as well as the requested `cx`/`cy` keys. `reproduce_v3.sh` then regenerates every macro and table of version 3 in `paper_v3/generated/` and compares them with the shipped ones. With the pinned packages both match byte for byte. Scripts that need data that is not distributed (the Library-10 drawings) are skipped, and their shipped scores are used. Build the paper with `cd paper_v3 && bash build.sh`.

Without `POINT_XY=1`, `reproduce.sh` gives the reading of versions 1 and 2, which keeps only `cx`/`cy` keys; `xy_reading.sh` compares the two readings.

## Where each result comes from

All scripts live in `experiments/` and read and write paths relative to it.

| Paper result | Scripts | Answers and scores |
|---|---|---|
| Measured interfaces (token side, budget, coordinate frame) | `calib_tokens.py`, `gen_calib.py`, `step1_coord_calib.py`, `step1b_gemini.py`, `ceilings.py`, `gen_ceilings.py` | `calib_tokens.json`, `gen_calib*.json`, `ceilings.json`, `gen_ceilings.json` |
| Controlled test of the assumptions | `syn_assume.py build / run <model> / score` | `syn_assume/results/`, `syn_assume/assume_scores.json` |
| False outputs and the F1 exchange rate | `theory_v8/output_form.py`, `theory_v8/f1_union.py` | `theory_v8/*.json` |
| Bits carried per answer (capacity theorem) | `theory_v10/info_carried.py` | `theory_v10/info_carried.json` |
| Controlled sweeps of S, content, area, clutter, upsampling | `syn.py build / run <model>`, `syn_analysis.py`, `t1_syn.py` | `syn/results/`, `syn/analysis_robust.json`, `results_x/t1_syn.json` |
| Newest generation: the S at which recall and precision reach 0.9 | `vspec/phase0.py`, `vspec/phase1.py`, `vspec/phase2.py`, `vspec/phase_val.py`, `vspec/vspec.py` | `vspec/results/`, `vspec/frozen_s90*.json` |
| Load at S = 1.5 (rule-of-thumb table) | `loadsweep/ls_run.py build / run`, `loadsweep/ls_analyze.py`, `../paper_v3/scripts/tabL_load.py` | `results_ls/` |
| LegendSpot, older-generation models (whole image, tiles, vendor modes, propose and verify) | `main_test.py`, `x_small.py`, `x3_fixedgrid.py`, `x4_vendor_highres.py`, `x4b_gemini_high.py`, `hybrid_verify.py`, `analyze_test.py` | `results_test/` |
| LegendSpot, newer-generation models, Gemini 4 | `gen_test.py`, `gen_common.py`, `analyze_gen.py`, `gem4_t1.py`, `gem4_a1n.py` | `results_gen/` |
| Mechanism checks on the prompt-writing sheets | `phase0.py`, `mech.py`, `step2_mech.py`, `step2_score.py`, `tile_runner.py`, `score_tiles.py` | `results_mech/`, `results/` |
| Floor-plan blocks and mosaics | `fpc_vlm.py`, `fpc_shard.py`, `fpc_score.py`, `fpc_geom.py`, `fpc_sratio.py`, `t2_fpc.py` | `results_fpc/`, `results_x/t2_nested_fpc.json` |
| Nested content, packed tiles, vendor high-resolution modes | `mosaic.py`, `x_new_score.py`, `x9_run.py`, `x9_score.py`, `x4q_score.py`, `t2_nested.py` | `results_x/` |
| Zoom agents, and the check of their counts and positions | `agent.py`, `redp10_agent.py`, `../paper_v3/scripts/agent_err_check.py` | `results_x/agent/`, `results_redp10/agent/`, `results_x/agent_err_check.json` |
| Classical and learned references | `../data/redp40/baseline_template_matching.py`, `learned_baselines.py`, `lb_modal.py`, `lb_score2.py`, `fsc_counters.py` | `../data/redp40/results/`, `results_x/learned*/` |
| FSC-147 | `fsc_run.py` | `../data/fsc147/` |
| Library-10 (drawings not released) | `redp10_run.py`, `redp10_score.py` | `results_redp10/` (scores and answers only) |
| Every ordering implied in the other experiments; where splitting helps and hurts | `law_data.py`, `theory_check.py`, `theory_check2_main.py`, `theory_check2_fsc.py`, `merge_tc.py`, `fig_theory.py`, `../paper_v3/scripts/fig3.py` | `results_law/` |
| The rule against the whole image and the 44 px recipe | `cs_rule.py <redp\|fpc> <model> [plan]`, `cs_analyze.py`, `../paper_v3/scripts/tab1.py` | `results_cs/` |
| Common splitting recipes checked from geometry alone | `../paper_v3/scripts/geom_recipes.py` | `../paper_v3/generated/geom_recipes.json` |
| Coupling, locality, annotation audit, limits | `locality_check.py`, `annot_audit.py`, `limits_analysis.py`, `ceiling_check_dev.py` | `results_x/` |
| Macros, tables, and figures | `paper2_numbers.py`, `paper2_numbers_x.py`, `paper2_figures.py`, `paper2_figures_x.py`, `../paper_v3/scripts/` | `../paper/generated/`, `../paper_v3/generated/`, `../paper_v3/figures*/` |

`law_fit.py`, `law_frontier.py`, `law_m3.py`, and `feas_law.py` are exploratory curve fits that the paper does not use.

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

Copy `config.env.example` to `config.env` in the repository root and fill in the keys you need. Keys are read from that file or the environment and are never printed or logged. Each experiment script is resumable, so rerunning a command skips calls already in its output file. The shipped answers were collected in September and October 2026.
## Data and licenses

* **Code**: MIT License (`LICENSE`).
* **LegendSpot**: drawings, templates, and annotations, released for research under `LICENSE-DATA`. The drawings are partial screenshots without project-identifying information. See `data/redp40/README.md`.
* **FloorPlanCAD**-derived images and annotations are CC BY-NC 4.0, as FloorPlanCAD is (`data/fpc/README.md`).
* **FSC-147** images are not redistributed. We include the item list and the original annotations of the 150 images used, for scoring.
* **Library-10**: the ten drawings with library references are used for evaluation only and are not released. Their model answers and scores are included.
* The load-sweep images (`experiments/loadsweep/images/`) are not shipped; `ls_run.py build` rebuilds them byte for byte.

## Citation

```bibtex
@misc{shi2026zoomin,
  title         = {Why {VLMs} Miss Small Objects, and When Zooming In Is Safe},
  author        = {Shi, Junzhe and Gan, Yuan and Jiang, Shida},
  year          = {2026},
  eprint        = {2610.09313},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2610.09313},
  url           = {https://arxiv.org/abs/2610.09313}
}
```
