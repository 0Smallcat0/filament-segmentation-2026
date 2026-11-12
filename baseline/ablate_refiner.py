"""Refiner ablation on one held-out fold: detector masks only, single-pass refiner, guarded TTA.

All three variants share the detector, the grouping, the seeded split, the merge,
the area filter and the empty-image retry; only the source of the pixel mask changes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from baseline import handoff
from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices
from src.postprocess import assert_disjoint

VARIANTS = ('detector_only', 'refiner_single_pass', 'refiner_guarded_tta')
MIN_AREA = 256


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def detector_mask_only(probability, bounds, fallback, *, threshold):
    return np.asarray(fallback, dtype=bool)


def run(detector, refiner, image, device, detector_score):
    """Return {variant: masks} for one image at the given detector score."""
    handoff.DETECTOR_SCORE_THRESHOLD = detector_score
    _, single, tta, _ = handoff.infer_observation(detector, refiner, image, device=device)
    original = handoff._decode_task
    handoff._decode_task = detector_mask_only
    try:
        _, _, plain, _ = handoff.infer_observation(detector, refiner, image, device=device)
    finally:
        handoff._decode_task = original
    return dict(zip(VARIANTS, (plain, single, tta)))


def keep_large(masks):
    return [m for m in masks if int(mask_utils.area(m)) >= MIN_AREA]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--detector-score', type=float, default=0.8)
    parser.add_argument('--refiner-threshold', type=float, default=0.7)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [r for r in prepare_train(args.train_json) if r['fold'] == args.fold]
    manifest = json.loads((args.weights / f'provenance_fold{args.fold}.json').read_text(encoding='utf8'))
    held_out = {r['filename'] for r in records}
    if (manifest['status'] != 'complete' or manifest['fold'] != args.fold
            or set(manifest['held_out_filenames']) != held_out
            or set(manifest['train_filenames']) & held_out):
        raise ValueError('Invalid held-out training provenance')
    for name in ('final_detector.pt', 'final_refiner.pt'):
        if sha(args.weights / name) != manifest['outputs_sha256'][name]:
            raise ValueError('Checkpoint hash mismatch')
    handoff.CANDIDATE_REFINER_THRESHOLD = args.refiner_threshold
    handoff.REFINER_TTA_THRESHOLD = args.refiner_threshold
    device = torch.device('cuda:0')
    detector = handoff.build_detector().to(device)
    detector.load_state_dict(torch.load(args.weights / 'final_detector.pt', map_location='cpu', weights_only=True)['model'])
    refiner = handoff.CropUNet().to(device)
    refiner.load_state_dict(torch.load(args.weights / 'final_refiner.pt', map_location='cpu', weights_only=True)['model'])
    detector.eval(); refiner.eval()
    matrices = {v: [] for v in VARIANTS}
    empty = {v: 0 for v in VARIANTS}
    instances = {v: 0 for v in VARIANTS}
    for index, record in enumerate(records):
        image = cv2.imread(str(args.images / record['filename']), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f"Invalid held-out image {record['filename']}")
        detector.roi_heads.score_thresh = 0.05
        first = {v: keep_large(m) for v, m in run(detector, refiner, image, device, args.detector_score).items()}
        retry = None
        for variant in VARIANTS:
            masks = first[variant]
            if not masks:
                if retry is None:
                    detector.roi_heads.score_thresh = 0.001
                    retry = {v: keep_large(m) for v, m in run(detector, refiner, image, device, 0.01).items()}
                masks = retry[variant]
            assert_disjoint(masks)
            matrices[variant].append(overlap_matrix(record['rles'], masks))
            empty[variant] += int(not masks)
            instances[variant] += len(masks)
        if (index + 1) % 10 == 0:
            print(f'fold {args.fold}: {index + 1}/{len(records)} observations', flush=True)
    results = {}
    for variant in VARIANTS:
        score = score_matrices(matrices[variant])
        iou = np.asarray(score.pop('matched_iou'), dtype=float)
        dice = 2 * iou / (1 + iou)
        score['matched_iou_mean'] = float(iou.mean())
        score['matched_iou_quartiles'] = [float(q) for q in np.percentile(iou, (25, 50, 75))]
        score['matched_dice_mean'] = float(dice.mean())
        score['matched_dice_quartiles'] = [float(q) for q in np.percentile(dice, (25, 50, 75))]
        score['matched_iou_ge_0_6'] = float((iou >= 0.6).mean())
        results[variant] = {'empty': empty[variant], 'instances': instances[variant], 'score': score}
    args.output.write_text(json.dumps({
        'fold': args.fold, 'observations': len(records), 'detector_score': args.detector_score,
        'refiner_threshold': args.refiner_threshold, 'minimum_area': MIN_AREA,
        'annotation_sha256': sha(args.train_json),
        'checkpoint_sha256': {n: sha(args.weights / n) for n in ('final_detector.pt', 'final_refiner.pt')},
        'source_sha256': {'ablate_refiner.py': sha(Path(__file__)), 'handoff.py': sha(Path(handoff.__file__))},
        'results': results}, indent=2), encoding='utf8')
    print(args.output.read_text(encoding='utf8'), flush=True)


if __name__ == '__main__':
    main()
