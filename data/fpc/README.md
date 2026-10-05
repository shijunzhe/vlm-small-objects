# FloorPlanCAD-derived items (FPC-300, FPC-sheets)

Both sets are built from the test split of FloorPlanCAD (Fan et al., ICCV 2021), as distributed in the Voxel51/FloorPlanCAD dataset on Hugging Face (1000 x 1000 px rasters). FloorPlanCAD is licensed CC BY-NC 4.0. The derived images, annotations, and references here carry the same license and are for non-commercial use.

* **FPC-300** (`bench/`): one item per (image, symbol class), with 3 to 80 instances, at most 20 items per class, seed 2026. There are 369 items; 339 are the test set. The reference is the instance with the median box area, cropped from the same image, as a legend crop would be. The ground truth is the box centres of all instances of the class.
* **FPC-sheets** (`sheets/`): for source drawings with at least nine test blocks, the first nine blocks are tiled into a 3 x 3 sheet (3000 x 3000 px). The top-left 2 x 2 and the first block are nested subsets (k = 2 and k = 1), so the three sheet areas share symbols, style, and reference. There are 60 sheets at each of the three areas, 180 items in all. Scoring ignores predictions and labels within m/2 of a block edge, where FloorPlanCAD's cropping cuts instances.

Annotation files use the same format as REDP-X40. `geometry.json` and `geometry_v4.json` hold per-item tokens per symbol side for every model, from the measured interfaces.

## Rebuilding from the original dataset
Download `samples.json` (the dataset's sample metadata) from the Voxel51/FloorPlanCAD dataset into this directory. Then run:

```bash
python build_fpc.py         # FPC-300; downloads each source raster from Hugging Face
python build_fpc_sheets.py  # FPC-sheets
```

The selection is deterministic (seed 2026). `tm_fpc.py` is the template-matching baseline.
