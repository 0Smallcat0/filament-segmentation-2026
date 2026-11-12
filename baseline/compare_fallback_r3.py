"""Compare rescue strategies on fold 1; the fold-0 rescuer has seen this fold."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from pycocotools import mask as mask_utils
from baseline import handoff
from baseline.provenance import object_sha256
from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices
from src.postprocess import assert_disjoint


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(weights, fold, records, annotation, device):
    manifest_path = weights / f'provenance_fold{fold}.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    expected = {r['filename'] for r in records if r['fold'] == fold}
    all_names = {r['filename'] for r in records}
    dataset = manifest['dataset']
    if (manifest['status'] != 'complete' or manifest['fold'] != fold
            or manifest['dataset']['annotation_sha256'] != sha(annotation)
            or dataset['sha256'] != object_sha256({k: v for k, v in dataset.items() if k != 'sha256'})
            or len(manifest['held_out_filenames']) != len(expected)
            or set(manifest['held_out_filenames']) != expected
            or set(manifest['train_filenames']) != all_names - expected):
        raise ValueError('Invalid provenance')
    settings = manifest['settings']
    if (settings['detector_score_threshold'] != 0.7
            or settings['detector_mask_threshold'] != 0.5
            or settings['refiner_threshold'] != 0.7
            or settings['inference'] != 'handoff-topology-safe-flip4-tta'
            or manifest['source_code']['files']['baseline/handoff.py'] != sha(Path(handoff.__file__))):
        raise ValueError('OOF inference settings or handoff source differ')
    for name in ('final_detector.pt', 'final_refiner.pt', f'oof_fold{fold}.csv', f'processed_fold{fold}.json'):
        if sha(weights / name) != manifest['outputs_sha256'][name]:
            raise ValueError('Checkpoint hash mismatch')
    processed = json.loads((weights / f'processed_fold{fold}.json').read_text(encoding='utf8'))
    if len(processed) != len(expected) or set(processed) != {Path(name).stem for name in expected}:
        raise ValueError('Processed images differ from held-out set')
    detector = handoff.build_detector().to(device)
    detector.load_state_dict(torch.load(weights / 'final_detector.pt', map_location='cpu', weights_only=True)['model'])
    refiner = handoff.CropUNet().to(device)
    refiner.load_state_dict(torch.load(weights / 'final_refiner.pt', map_location='cpu', weights_only=True)['model'])
    detector.eval(); refiner.eval()
    return detector, refiner, manifest


def infer(detector, refiner, image, score, floor, device):
    detector.roi_heads.score_thresh = floor
    handoff.DETECTOR_SCORE_THRESHOLD = score
    _, _, masks, _ = handoff.infer_observation(detector, refiner, image, device=device)
    masks = [m for m in masks if int(mask_utils.area(m)) >= 256]
    assert_disjoint(masks)
    return masks


def main():
    p = argparse.ArgumentParser()
    for name in ('train-json', 'images', 'fold0', 'fold1', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    records = prepare_train(args.train_json)
    heldout = [r for r in records if r['fold'] == 1]
    device = torch.device('cuda:0')
    d0, r0, manifest0 = load(args.fold0, 0, records, args.train_json, device)
    d1, r1, manifest1 = load(args.fold1, 1, records, args.train_json, device)
    if manifest0['dataset']['sha256'] != manifest1['dataset']['sha256']:
        raise ValueError('Mixed training datasets')
    predictions = defaultdict(list)
    with (args.fold1 / 'oof_fold1.csv').open(newline='', encoding='utf8') as stream:
        for row in csv.DictReader(stream):
            stem, _ = row['filament_id'].rsplit('_', 1)
            m = {'size': [2048, 2048], 'counts': row['segmentation_rle'].encode('ascii')}
            if int(mask_utils.area(m)) >= 256:
                predictions[stem].append(m)
    allowed = {Path(r['filename']).stem for r in heldout}
    if set(predictions) - allowed:
        raise ValueError('Unexpected prediction')
    hashes = {x['filename']: x['image_sha256'] for x in manifest1['dataset']['observations']}
    choices = {'own_001': {}, 'fold0_08_then_001': {}, 'own_then_fold0': {}}
    empties = [r for r in heldout if not predictions[Path(r['filename']).stem]]
    for record in empties:
        path = args.images / record['filename']
        if sha(path) != hashes[record['filename']]:
            raise ValueError('Image hash mismatch')
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError('Invalid image')
        stem = path.stem
        choices['own_001'][stem] = infer(d1, r1, image, 0.01, 0.001, device)
        masks = infer(d0, r0, image, 0.8, 0.05, device)
        if not masks:
            masks = infer(d0, r0, image, 0.01, 0.001, device)
        choices['fold0_08_then_001'][stem] = masks
        choices['own_then_fold0'][stem] = choices['own_001'][stem] or masks
    results = {}
    for label, fills in choices.items():
        stats = score_matrices(overlap_matrix(r['rles'], fills.get(Path(r['filename']).stem,
                                 predictions[Path(r['filename']).stem])) for r in heldout)
        stats.pop('matched_iou')
        stats['empty_before'] = len(empties)
        stats['empty_after'] = sum(not m for m in fills.values())
        results[label] = stats
    result = {'fold': 1, 'primary_threshold': 0.7,
              'warning': 'fold0 rescuer was trained on fold1 images; this comparison is optimistic, not clean OOF',
              'annotation_sha256': sha(args.train_json),
              'provenance_sha256': [sha(args.fold0 / 'provenance_fold0.json'), sha(args.fold1 / 'provenance_fold1.json')],
              'results': results}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
