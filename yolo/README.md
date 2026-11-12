# YOLO11-seg comparison

A single-stage instance-segmentation comparison on the same observation-grouped folds and medoid annotators as the submitted pipeline (`src.data.prepare_train`). It starts from Ultralytics' generic COCO `yolo11s-seg.pt`, trains for a fixed 30 epochs on four folds of the competition training set, and predicts the held-out fold with high-resolution masks.

On held-out fold 1 it scores PQ 0.357 with 110 one-to-many and 52 many-to-one relations, against 0.412 and 23 / 43 for the detector and refiner on the same fold, so the two-stage pipeline was submitted.

| File | Role |
|---|---|
| `fold.py` | Train one fold and predict its held-out observations |
| `common.py` | Polygon conversion and mask handling |
| `tune.py`, `tune_single.py`, `tune_fallback.py` | Threshold grids on a held-out fold |
| `make_kernel.py` | Packaging of the Kaggle kernel |
