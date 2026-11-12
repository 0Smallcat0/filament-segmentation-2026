"""Evaluate detector/refiner thresholds using held-out train images only."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import torch
from pycocotools import mask as mask_utils
from baseline import handoff
from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices
from src.postprocess import assert_disjoint


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detector-score', type=float)
    parser.add_argument('--refiner-threshold', type=float)
    parser.add_argument('--selection', type=Path)
    args = parser.parse_args()
    records = [r for r in prepare_train(args.train_json) if r['fold'] == args.fold]
    manifest_path = args.weights / f'provenance_fold{args.fold}.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    expected = {r['filename'] for r in records}
    if (manifest['status'] != 'complete' or manifest['fold'] != args.fold
            or manifest['settings'].get('training_scope') == 'all_competition_train_observations'
            or manifest['dataset']['annotation_sha256'] != sha(args.train_json)
            or set(manifest['held_out_filenames']) != expected
            or set(manifest['train_filenames']) & expected):
        raise ValueError('Invalid held-out training provenance')
    for name in ('final_detector.pt', 'final_refiner.pt'):
        if sha(args.weights / name) != manifest['outputs_sha256'][name]:
            raise ValueError('Checkpoint hash mismatch')
    image_hashes = {item['filename']: item['image_sha256'] for item in manifest['dataset']['observations']}
    settings = [(0.6, 0.7), (0.7, 0.6), (0.7, 0.7), (0.7, 0.8), (0.8, 0.7)]
    if (args.detector_score is None) != (args.refiner_threshold is None):
        raise ValueError('Specify both selected thresholds')
    if args.detector_score is not None:
        if args.fold != 0 or not args.selection:
            raise ValueError('Evaluation requires fold 0 and a fold-1 selection artifact')
        selection = json.loads(args.selection.read_text(encoding='utf8'))
        if (selection['fold'] != 1 or selection['role'] != 'selection'
                or selection['annotation_sha256'] != sha(args.train_json)
                or selection['source_sha256'] != {'tune_handoff.py': sha(Path(__file__)),
                                                   'handoff.py': sha(Path(handoff.__file__))}):
            raise ValueError('Invalid selection artifact')
        eligible = [r for r in selection['results'] if r['empty'] == 0]
        if not eligible:
            raise ValueError('No fully covered selection candidate')
        winner = max(eligible, key=lambda r: r['score']['official']['pq'])
        selected = (args.detector_score, args.refiner_threshold)
        if selected not in settings or selected != (winner['detector_score'], winner['refiner_threshold']):
            raise ValueError('Selected thresholds do not match the fold-1 winner')
        settings = [selected]
    elif args.fold != 1 or args.selection:
        raise ValueError('Grid selection must run on fold 1')
    matrices, empty = {s: [] for s in settings}, {s: 0 for s in settings}
    detector = handoff.build_detector().cuda()
    detector.load_state_dict(torch.load(args.weights / 'final_detector.pt', map_location='cpu', weights_only=True)['model'])
    refiner = handoff.CropUNet().cuda()
    refiner.load_state_dict(torch.load(args.weights / 'final_refiner.pt', map_location='cpu', weights_only=True)['model'])
    detector.eval(); refiner.eval()
    device = torch.device('cuda:0')
    started = time.monotonic()
    for index, record in enumerate(records):
        path = args.images / record['filename']
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f'Invalid held-out image {path}')
        if sha(path) != image_hashes[record['filename']]:
            raise ValueError(f'Held-out image hash mismatch: {path.name}')
        for confidence, refiner_threshold in settings:
            handoff.DETECTOR_SCORE_THRESHOLD = confidence
            handoff.CANDIDATE_REFINER_THRESHOLD = refiner_threshold
            handoff.REFINER_TTA_THRESHOLD = refiner_threshold
            detector.roi_heads.score_thresh = 0.05
            _, _, masks, _ = handoff.infer_observation(detector, refiner, image, device=device)
            masks = [m for m in masks if int(mask_utils.area(m)) >= 256]
            if not masks:
                detector.roi_heads.score_thresh = 0.001
                handoff.DETECTOR_SCORE_THRESHOLD = 0.01
                _, _, masks, _ = handoff.infer_observation(detector, refiner, image, device=device)
                masks = [m for m in masks if int(mask_utils.area(m)) >= 256]
            assert_disjoint(masks)
            matrices[(confidence, refiner_threshold)].append(overlap_matrix(record['rles'], masks))
            empty[(confidence, refiner_threshold)] += int(not masks)
        if (index + 1) % 10 == 0:
            print(f'fold {args.fold}: {index + 1}/{len(records)} observations', flush=True)
    results = []
    for confidence, refiner_threshold in settings:
        result = score_matrices(matrices[(confidence, refiner_threshold)])
        result.pop('matched_iou')
        results.append({'detector_score': confidence, 'refiner_threshold': refiner_threshold,
                        'empty': empty[(confidence, refiner_threshold)], 'score': result})
    args.output.write_text(json.dumps({'fold': args.fold, 'role': 'evaluation' if args.selection else 'selection',
                                       'annotation_sha256': sha(args.train_json), 'provenance_sha256': sha(manifest_path),
                                       'checkpoint_sha256': {name: sha(args.weights / name) for name in ('final_detector.pt', 'final_refiner.pt')},
                                       'source_sha256': {'tune_handoff.py': sha(Path(__file__)), 'handoff.py': sha(Path(handoff.__file__))},
                                       'selection_sha256': sha(args.selection) if args.selection else None,
                                       'observations': len(records),
                                       'seconds': time.monotonic() - started, 'results': results}, indent=2), encoding='utf8')
    print(args.output.read_text(encoding='utf8'), flush=True)


if __name__ == '__main__':
    main()
