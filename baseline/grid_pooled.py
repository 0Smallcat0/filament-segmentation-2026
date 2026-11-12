"""Held-out five-fold threshold grid; all inputs must be competition train data."""
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from baseline.provenance import object_sha256
from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices
from src.postprocess import assert_disjoint

PAIRS = ((0.6, 0.7), (0.7, 0.6), (0.7, 0.7), (0.7, 0.8), (0.8, 0.7))
CONFIGS = tuple(itertools.product(PAIRS, (0, 256), (0, 2)))
DIAGNOSTICS = ('one_gt_to_many_predictions_iou_gt_0', 'many_gt_to_one_prediction_iou_gt_0')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def postprocess(rles, min_area, close_radius):
    """Filter by original area, then close survivors in order without stealing pixels."""
    rles = [r for r in rles if int(mask_utils.area(r)) >= min_area]
    if not close_radius:
        result = list(rles)
    else:
        originals = [mask_utils.decode(r).astype(bool) for r in rles]
        if not originals:
            return []
        occupied = np.logical_or.reduce(originals)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close_radius + 1,) * 2)
        result = []
        for original in originals:
            closed = cv2.morphologyEx(original.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)
            added = closed & ~occupied
            occupied |= added
            result_mask = original | added
            result.append(mask_utils.encode(np.asfortranarray(result_mask.astype(np.uint8))))
    assert_disjoint(result)
    return result


def compact_score(matrices):
    score = score_matrices(matrices)
    score.pop('matched_iou')
    return score


def run(args):
    import torch
    import torchvision
    from baseline import handoff
    from baseline import provenance
    from src import data, metrics, postprocess as postprocess_module
    records = prepare_train(args.train_json)
    held = [r for r in records if r['fold'] == args.fold]
    manifest_path = args.weights / f'provenance_fold{args.fold}.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    expected = {r['filename'] for r in held}
    all_names = {r['filename'] for r in records}
    dataset = manifest['dataset']
    if (manifest['status'] != 'complete' or manifest['fold'] != args.fold
            or manifest['settings'].get('training_scope') == 'all_competition_train_observations'
            or dataset['annotation_sha256'] != sha(args.train_json)
            or dataset['sha256'] != object_sha256({k: v for k, v in dataset.items() if k != 'sha256'})
            or len(manifest['held_out_filenames']) != len(expected)
            or set(manifest['held_out_filenames']) != expected
            or set(manifest['train_filenames']) != all_names - expected):
        raise ValueError('Invalid held-out training provenance')
    for name in ('final_detector.pt', 'final_refiner.pt'):
        if sha(args.weights / name) != manifest['outputs_sha256'][name]:
            raise ValueError('Checkpoint hash mismatch')
    image_hashes = {item['filename']: item['image_sha256'] for item in dataset['observations']}
    detector = handoff.build_detector().cuda()
    detector.load_state_dict(torch.load(args.weights / 'final_detector.pt', map_location='cpu', weights_only=True)['model'])
    refiner = handoff.CropUNet().cuda()
    refiner.load_state_dict(torch.load(args.weights / 'final_refiner.pt', map_location='cpu', weights_only=True)['model'])
    detector.eval()
    refiner.eval()
    device = torch.device('cuda:0')
    environment = {'python': platform.python_version(), 'numpy': np.__version__,
                   'opencv': cv2.__version__, 'torch': torch.__version__,
                   'torchvision': torchvision.__version__, 'torch_cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0),
                   'scipy': importlib.metadata.version('scipy'),
                   'pycocotools': importlib.metadata.version('pycocotools')}
    started = time.monotonic()
    per_config = {str(i): [] for i in range(len(CONFIGS))}
    for index, record in enumerate(held):
        path = args.images / record['filename']
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048) or sha(path) != image_hashes[record['filename']]:
            raise ValueError(f'Invalid held-out image {path}')
        fallback = {}
        for pair in PAIRS:
            confidence, threshold = pair
            handoff.DETECTOR_SCORE_THRESHOLD = confidence
            handoff.CANDIDATE_REFINER_THRESHOLD = threshold
            handoff.REFINER_TTA_THRESHOLD = threshold
            detector.roi_heads.score_thresh = 0.05
            _, _, raw, _ = handoff.infer_observation(detector, refiner, image, device=device)
            if any(not postprocess(raw, area, radius) for area, radius in itertools.product((0, 256), (0, 2))):
                if threshold not in fallback:
                    detector.roi_heads.score_thresh = 0.001
                    handoff.DETECTOR_SCORE_THRESHOLD = 0.01
                    _, _, fallback[threshold], _ = handoff.infer_observation(detector, refiner, image, device=device)
            for area, radius in itertools.product((0, 256), (0, 2)):
                config_index = CONFIGS.index((pair, area, radius))
                masks = postprocess(raw, area, radius)
                if not masks:
                    masks = postprocess(fallback[threshold], area, radius)
                per_config[str(config_index)].append({'filename': record['filename'], 'empty': not bool(masks),
                    'score': compact_score([overlap_matrix(record['rles'], masks)])})
        if (index + 1) % 10 == 0:
            print(f'fold {args.fold}: {index + 1}/{len(held)}', flush=True)
    configs = [{'detector_score': pair[0], 'refiner_threshold': pair[1], 'min_area': area,
                'close_radius': radius} for pair, area, radius in CONFIGS]
    result = {'fold': args.fold, 'annotation_sha256': sha(args.train_json),
              'dataset_sha256': dataset['sha256'], 'provenance_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
              'checkpoint_sha256': {name: sha(args.weights / name) for name in ('final_detector.pt', 'final_refiner.pt')},
              'source_sha256': {name: sha(Path(path)) for name, path in [('grid_pooled.py', __file__),
                  ('handoff.py', handoff.__file__), ('provenance.py', provenance.__file__),
                  ('data.py', data.__file__), ('metrics.py', metrics.__file__),
                  ('postprocess.py', postprocess_module.__file__)]},
              'environment': environment,
              'configs': configs, 'observations': per_config, 'seconds': time.monotonic() - started}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf8')


def pool(paths, train_json, manifest_paths):
    records = prepare_train(train_json)
    expected_by_fold = {f: {r['filename'] for r in records if r['fold'] == f} for f in range(5)}
    docs = [json.loads(p.read_text(encoding='utf8')) for p in paths]
    manifests = {json.loads(p.read_text(encoding='utf8'))['fold']: p for p in manifest_paths}
    if sorted(d['fold'] for d in docs) != list(range(5)) or len(manifest_paths) != 5 or set(manifests) != set(range(5)):
        raise ValueError('Exactly one result and manifest per fold required')
    if len({d['dataset_sha256'] for d in docs}) != 1 or len({json.dumps(d['source_sha256'], sort_keys=True) for d in docs}) != 1:
        raise ValueError('Mixed data or source versions')
    if any('environment' not in d for d in docs) or len({json.dumps(d['environment'], sort_keys=True) for d in docs}) != 1:
        raise ValueError('Mixed or missing execution environments')
    if len({json.dumps(d['configs'], sort_keys=True) for d in docs}) != 1:
        raise ValueError('Mixed grid configurations')
    training_manifests = []
    for d in docs:
        manifest_path = manifests[d['fold']]
        manifest = json.loads(manifest_path.read_text(encoding='utf8'))
        training_manifests.append(manifest)
        if (d['annotation_sha256'] != sha(train_json) or d['provenance_sha256'] != sha(manifest_path)
                or manifest['dataset']['sha256'] != d['dataset_sha256']
                or any(manifest['outputs_sha256'][n] != h for n, h in d['checkpoint_sha256'].items())):
            raise ValueError('Artifact provenance mismatch')
        for field in ('settings', 'environment'):
            if object_sha256(manifest[field]) != manifest[field + '_sha256']:
                raise ValueError(f'Training {field} hash mismatch')
        dataset = manifest['dataset']
        if object_sha256({k: v for k, v in dataset.items() if k != 'sha256'}) != dataset['sha256']:
            raise ValueError('Training dataset hash mismatch')
        if object_sha256(manifest['source_code']['files']) != manifest['source_code']['sha256']:
            raise ValueError('Training source hash mismatch')
        expected = expected_by_fold[d['fold']]
        all_names = {r['filename'] for r in records}
        if (manifest['status'] != 'complete' or manifest['fold'] != d['fold']
                or manifest['dataset']['annotation_sha256'] != sha(train_json)
                or len(manifest['held_out_filenames']) != len(expected)
                or set(manifest['held_out_filenames']) != expected
                or set(manifest['train_filenames']) != all_names - expected):
            raise ValueError('Training split mismatch')
        for i in range(len(d['configs'])):
            items = d['observations'][str(i)]
            if len(items) != len(expected_by_fold[d['fold']]) or {x['filename'] for x in items} != expected_by_fold[d['fold']]:
                raise ValueError('Held-out observations mismatch')
    for name in ('dataset', 'source_code'):
        if len({m[name]['sha256'] for m in training_manifests}) != 1:
            raise ValueError(f'Mixed training {name} hashes')
    for name in ('settings_sha256', 'environment_sha256'):
        if len({m[name] for m in training_manifests}) != 1:
            raise ValueError(f'Mixed training {name}')
    pooled = []
    for i, config in enumerate(docs[0]['configs']):
        items = [x for d in docs for x in d['observations'][str(i)]]
        result = {'config': config, 'empty': sum(x['empty'] for x in items)}
        for variant in ('official', 'one_to_one'):
            counts = {k: sum(x['score'][variant][k] for x in items) for k in ('tp', 'fp', 'fn', 'sum_iou')}
            den = counts['tp'] + .5 * (counts['fp'] + counts['fn'])
            counts['pq'] = counts['sum_iou'] / den if den else 0.0
            result[variant] = counts
        for key in DIAGNOSTICS:
            result[key] = sum(x['score'][key] for x in items)
        pooled.append(result)
    return sorted(pooled, key=lambda x: x['official']['pq'], reverse=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('run')
    r.add_argument('--fold', type=int, choices=range(5), required=True)
    for name in ('train-json', 'images', 'weights', 'output'):
        r.add_argument('--' + name, type=Path, required=True)
    q = sub.add_parser('pool')
    q.add_argument('--train-json', type=Path, required=True)
    q.add_argument('--manifests', type=Path, nargs=5, required=True)
    q.add_argument('results', type=Path, nargs=5)
    args = p.parse_args()
    if args.action == 'run':
        run(args)
    else:
        print(json.dumps(pool(args.results, args.train_json, args.manifests), indent=2))


if __name__ == '__main__':
    main()
