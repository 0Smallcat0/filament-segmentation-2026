"""Score lower detector thresholds solely for empty held-out predictions."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from pycocotools import mask as mask_utils
from baseline import handoff
from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices
from src.postprocess import assert_disjoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--low-proposals', action='store_true')
    args = parser.parse_args()
    records = [r for r in prepare_train(args.train_json) if r['fold'] == args.fold]
    predictions = defaultdict(list)
    with (args.weights / f'oof_fold{args.fold}.csv').open(newline='', encoding='utf8') as stream:
        for row in csv.DictReader(stream):
            stem, _ = row['filament_id'].rsplit('_', 1)
            mask = {'size': [2048, 2048], 'counts': row['segmentation_rle'].encode('ascii')}
            if int(mask_utils.area(mask)) >= 256:
                predictions[stem].append(mask)
    allowed = {Path(r['filename']).stem for r in records}
    if set(predictions) - allowed:
        raise ValueError('OOF predictions outside held-out fold')
    device = torch.device('cuda:0')
    detector = handoff.build_detector().to(device)
    detector.load_state_dict(torch.load(args.weights / 'final_detector.pt', map_location='cpu', weights_only=True)['model'])
    if args.low_proposals:
        detector.roi_heads.score_thresh = 0.001
    refiner = handoff.CropUNet().to(device)
    refiner.load_state_dict(torch.load(args.weights / 'final_refiner.pt', map_location='cpu', weights_only=True)['model'])
    detector.eval(); refiner.eval()
    empty = [r for r in records if not predictions[Path(r['filename']).stem]]
    candidates = {}
    for threshold in ((0.05, 0.01, 0.001) if args.low_proposals else (0.5, 0.3, 0.1)):
        handoff.DETECTOR_SCORE_THRESHOLD = threshold
        fills = {}
        for record in empty:
            path = args.images / record['filename']
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (2048, 2048):
                raise ValueError(f'Invalid held-out image {path}')
            _, _, masks, _ = handoff.infer_observation(detector, refiner, image, device=device)
            assert_disjoint(masks)
            fills[Path(record['filename']).stem] = masks
        for area in ((16, 64, 256) if args.low_proposals else (256,)):
            filtered = {stem: [m for m in masks if int(mask_utils.area(m)) >= area]
                        for stem, masks in fills.items()}
            results = score_matrices(overlap_matrix(r['rles'], filtered.get(Path(r['filename']).stem,
                                                          predictions[Path(r['filename']).stem])) for r in records)
            results.pop('matched_iou')
            results['empty_before'] = len(empty)
            results['empty_after'] = sum(not masks for masks in filtered.values())
            candidates[f'{threshold}:{area}'] = results
            print(json.dumps({'fold': args.fold, 'threshold': threshold, 'area': area, **results}), flush=True)
    args.output.write_text(json.dumps(candidates, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
