# Detector and refiner

Five-fold training and inference of the submitted pipeline: a torchvision Mask R-CNN (ResNet-50 FPN v2) detector and a ResNet-18 U-Net that refines each detection on a native-resolution crop.

| File | Role |
|---|---|
| `run_fold.py` | Train detector and refiner on four folds, predict and score the held-out fold, write provenance |
| `detector.py` | Mask R-CNN training (two GPUs, mixed precision) |
| `refiner.py` | Crop cache and U-Net training |
| `handoff.py` | Per-image inference: detection, grouping, crops, guarded flip averaging, seeded split, overlap merge |
| `predict.py` | Test inference with weight verification, area filter and empty-image retry |
| `pool_oof.py` | Pool folds from summed counts after checking hashes and split membership |
| `grid_pooled.py`, `grid_report.py` | Five-fold configuration grid and bootstrap interval |
| `tune_handoff.py`, `tune_area.py`, `fallback_oof.py` | Selection of detector score, minimum area and retry thresholds on a held-out fold |
| `ablate_refiner.py` | Held-out refiner ablation: detector masks only, single refiner pass, guarded flip averaging |
| `split_weights.py` | Lossless split of the detector state into two files under 100 MB, with a tensor-level check |
| `make_*.py` | Packaging of the corresponding Kaggle kernels |

The training and crop code derives from [Nomannic19's Apache-2.0 notebook](https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner); `handoff.py` is the inference code of [phuongncn's notebook](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff), used here with our own weights. The detector starts from torchvision's COCO weights and the refiner encoder from ImageNet weights; the only solar labels used are the competition training annotations.

For each fold, `src.data.prepare_train` supplies the observation groups, the seed and the medoid annotators. The detector manifest and the refiner crop cache contain only the four training folds; held-out medoid masks are used for scoring only. Epoch counts (6 for the detector, 8 for the refiner) are fixed in advance. Each `provenance_foldN.json` binds the annotation bytes, each image's bytes, the medoid and fold assignment, the source code, the settings, the environment, the train and held-out file lists and the output hashes. `pool_oof.py` requires five distinct complete folds with matching dataset, source, settings and environment hashes, verifies the output hashes and the split, and then sums IoU and TP/FP/FN across folds before computing PQ.

```sh
python baseline/make_kernel.py --fold 0 --slug filament-baseline-fold0 --output ../tmp/baseline-fold0
kaggle kernels push -p ../tmp/baseline-fold0
```
