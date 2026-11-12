# Detect, then refine: instance-first solar filament segmentation

Solution of team **OAO noir** to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026).

**Technical report:** [`report/filament-segmentation-2026-report.pdf`](report/filament-segmentation-2026-report.pdf)

A Mask R-CNN detector proposes one mask per filament on the full-disk image, and a U-Net refiner redraws every detection on a native-resolution crop. The output is a set of pairwise disjoint instance masks. Both networks are trained only on the competition training annotations, starting from generic COCO and ImageNet weights.

| | |
|---|---|
| Public leaderboard PQ (submission 58176807) | **0.37** |
| Held-out PQ, pooled over 707 observations in five folds | **0.412** (95% bootstrap interval 0.402–0.422) |
| Held-out PQ per fold | 0.449, 0.412, 0.381, 0.407, 0.408 (mean 0.411, sample SD 0.024) |
| Matched instances of fold 0 (673) | IoU median 0.689, Dice median 0.816 |
| Refiner ablation on fold 0 | PQ 0.380 with detector masks only, 0.444 with a single refiner pass, **0.449** with guarded flip averaging |
| One-to-many / many-to-one relations, five folds | 102 / 211 for 5,231 reference instances |
| Test inference | 180 images, 1,194 disjoint instances, 249 s on one Kaggle T4 |
| Reproduction | byte-identical CSV on Kaggle T4, SHA-256 `e1d77f9dd6d1b37df1dc629e186c3f63abcbeba7eab8e1f72d0ca12ffb6ea7d7` |

## Repository layout

| Path | Content |
|---|---|
| `notebooks/pipeline.ipynb` | End-to-end inference notebook: test images in, `submission.csv` out |
| `weights/baseline-fold0/` | Final weights: detector (two parts, each under 100 MB) and refiner |
| `requirements.txt` | Pinned package versions |
| `baseline/predict.py` | Inference entry point used by the notebook |
| `baseline/handoff.py` | Detector, refiner and post-processing for one image |
| `baseline/detector.py`, `baseline/refiner.py`, `baseline/run_fold.py` | Training of one fold and its held-out prediction |
| `src/data.py` | Observation-grouped folds and medoid-annotator selection |
| `src/metrics.py`, `score_oof.py`, `baseline/pool_oof.py` | PQ, one-to-many / many-to-one counts, pooling over folds |
| `baseline/grid_pooled.py`, `baseline/grid_report.py` | Five-fold configuration grid and bootstrap interval |
| `baseline/ablate_refiner.py` | Refiner ablation on a held-out fold (detector masks only, single pass, guarded flip averaging) |
| `check_submission.py` | Coverage, emptiness and overlap checks of a submission file |
| `tests/` | Unit tests of scoring, data preparation and pipeline components |
| `report/` | Technical report (PDF) and the refiner ablation output |

## Reproduce the submission

```sh
pip install -r requirements.txt
export FILAMENT_TEST_IMAGES=/path/to/MAGFiLO_1.0_Kaggle_2026/test/test_images
jupyter nbconvert --to notebook --execute notebooks/pipeline.ipynb
```

The notebook reads the 180 test images and the three weight files in `weights/baseline-fold0`, and writes `submission.csv`. It does not train and reads no annotation. Before inference it rebuilds the detector state from its two parts in memory and verifies the SHA-256 of all tensors (`4409a5d50d18139097693c26fd2f06efa19a6463606d9c16eceda77cfe5ee2b2`) and of the refiner file (`95ff4976ae06e7022c1f207290a355fdf5a8758a1d47dd660b9bb1f4e3ecfdc4`). After inference it checks that every test image is covered, that no image is empty and that no two masks of an image overlap.

On Kaggle (T4 GPU, competition data attached) the notebook reproduces the submitted file byte for byte: 1,194 rows, SHA-256 `e1d77f9dd6d1b37df1dc629e186c3f63abcbeba7eab8e1f72d0ca12ffb6ea7d7`. A CPU run passes the same checks and differs from the T4 file in two masks, by 1 and 18 pixels.

Submitted settings: detector score 0.8, detector mask 0.5, refiner probability 0.7, minimum instance area 256 pixels, no morphological closing. An image left without any instance is retried once with detector proposal floor 0.001 and score 0.01.

## Pipeline

1. Resize the 2048 x 2048 image to 1024 x 1024 by area interpolation and run Mask R-CNN (ResNet-50 FPN v2, anchors 16–256 pixels).
2. Group detections that share pixels. For each group cut a square window with 1.8x context from the original 2048 x 2048 image and resize it to 256 x 256.
3. Refine each crop with a ResNet-18 U-Net. A four-flip average replaces the single pass only if area ratio, component count and Dice stay within fixed guards.
4. Threshold, paste back, split merged groups with the detections as seeds when the seeds are clearly distinct, merge remaining overlaps, drop instances under 256 pixels, and assert that the masks are disjoint.

## Training and evaluation

Training runs on Kaggle with two T4 GPUs; one fold takes about 20 minutes including held-out scoring.

```sh
export FILAMENT_DATA_ROOT=/path/to/MAGFiLO_1.0_Kaggle_2026
python train.py --baseline-fold 0          # train on folds 1-4, predict and score fold 0
python score_oof.py --train-json $FILAMENT_DATA_ROOT/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json \
    --predictions oof.csv --fold 0          # re-score a held-out CSV
python -m baseline.pool_oof out-fold0 out-fold1 out-fold2 out-fold3 out-fold4   # pool five folds from summed counts
python -m unittest discover -s tests -v    # 12 unit tests
```

Evaluation protocol:

- **Grouped folds.** The 707 training observations (1,154 annotator records) are split into five folds by observation, so all annotators of one image stay on the same side of a split.
- **Medoid reference.** For each observation the annotator with the highest mean semantic IoU to the other annotators is the training target and the validation reference.
- **Metric.** The competition PQ with matches at IoU > 0.5, plus a strict one-to-one variant, and counts of one-to-many and many-to-one relations at any positive overlap.
- **Pooling and uncertainty.** PQ over several folds is computed from summed TP, FP, FN and IoU totals. The 95% interval is a percentile bootstrap over the 707 held-out observations (1,000 resamples).
- **Selection.** The detector score and the minimum area were selected on fold 1 and confirmed on fold 0.
- **Provenance.** Every training run records hashes of the annotation file, each image, the source code, the settings and the environment; pooling refuses folds whose hashes or split membership disagree.

Held-out comparison on the same folds:

| Method | Data | PQ | One-to-many / many-to-one | Public PQ |
|---|---|---:|---:|---:|
| Semantic U-Net (ResNet-18) | fold 1 | 0.340 | 125 / 7 | |
| YOLO11-seg | fold 1 | 0.357 | 110 / 52 | |
| Detector + refiner, score 0.7 | fold 1 | 0.402 | 26 / 49 | |
| Detector + refiner, score 0.8 | fold 1 | 0.412 | 23 / 43 | |
| Detector + refiner, score 0.7 | fold 0 | 0.445 | 28 / 49 | 0.35 |
| **Detector + refiner, score 0.8 (submitted)** | fold 0 | **0.449** | 22 / 45 | **0.37** |
| Same, single refiner pass without flip averaging | fold 0 | 0.444 | 22 / 45 | |
| Same, detector masks without the refiner | fold 0 | 0.380 | 22 / 48 | |
| **Detector + refiner, score 0.8 (submitted)** | five folds | **0.412** | 102 / 211 | |

Public PQ is the leaderboard score of the fold-0 model run with that configuration on the test images. The refiner ablation (`python -m baseline.ablate_refiner`, output in `report/refiner-ablation-fold0.json`) keeps the detector, grouping, split, merge, area filter and retry fixed and changes only the source of the pixel mask: the refiner raises the mean IoU of matched instances from 0.637 to 0.681 and the number of matches from 611 to 673.

## Environment

Kaggle Python 3.13.16, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, NumPy 2.1.3, pandas 2.3.3, OpenCV 4.14.0.94, SciPy 1.16.3, pycocotools 2.0.10, on Tesla T4 GPUs. `requirements.txt` pins the package versions.

## Data and credits

Only the competition training images and annotations are used for training, and only the test images are read at inference. No external solar data, test annotation, lookup table or precomputed prediction is used.

The training and crop code derives from [Nomannic19's notebook](https://www.kaggle.com/code/nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner) (Apache-2.0), and the per-image inference code in `baseline/handoff.py` from [phuongncn's notebook](https://www.kaggle.com/code/phuongncn/lb-0-70-topology-safe-solar-filaments-handoff). The grouped held-out design was informed by [ektarr's notebook](https://www.kaggle.com/code/ektarr/new-metric-solar-filament). All weights in this repository were trained by us from generic COCO and ImageNet initialisations under the protocol above.

Code in this repository is released under the MIT licence (`LICENSE`); files derived from the notebooks above remain under their original terms.
