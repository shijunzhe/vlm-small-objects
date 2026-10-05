# FSC-147 subset

FSC-147 (Ranjan et al., CVPR 2021) images are not redistributed. Download them from the authors' release.

* `x8_items.json` lists the 150 test images used in the paper, with stratum, object scale m (pixels), and count n; `x8_ann_subset.json` holds the original FSC-147 annotations of these 150 images and `Train_Test_Val_FSC_147.json` the original split file (Ranjan et al., 2021), for scoring.
* `results/` holds the raw model answers.
* `x8_scores*.json` and `x8_summary*.json` hold the scores. The `_obj` files score the listed objects; `_strict` is the strict-matching variant reported in the paper.

To rerun, place the FSC-147 images in `images/` and the annotation file `annotation_FSC147_384.json` in this directory, then run `python fsc_run.py` from `experiments/`.
