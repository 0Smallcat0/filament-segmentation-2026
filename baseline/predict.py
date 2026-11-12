"""Run the retrained detector and refiner on an explicit test image directory."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import torch
from pycocotools import mask as mask_utils

from baseline import handoff
from baseline import grid_pooled
from src.postprocess import assert_disjoint


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        for field in (name, str(tensor.dtype), str(tuple(tensor.shape))):
            digest.update(field.encode() + bytes([0]))
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_models(weights, detector_hash, refiner_hash, device):
    detector = handoff.build_detector().to(device)
    parts = [weights / f'detector-part{i}.pt' for i in range(2)]
    detector_state = {}
    for path in parts:
        part = torch.load(path, map_location='cpu', weights_only=True)
        if detector_state.keys() & part.keys():
            raise ValueError('Duplicate detector tensor in checkpoint parts')
        detector_state.update(part)
    if state_hash(detector_state) != detector_hash:
        raise ValueError('Detector tensor hash mismatch')
    if sha256(weights / 'final_refiner.pt') != refiner_hash:
        raise ValueError('Refiner checkpoint hash mismatch')
    detector.load_state_dict(detector_state)
    refiner = handoff.CropUNet().to(device)
    refiner.load_state_dict(torch.load(weights / 'final_refiner.pt', map_location='cpu', weights_only=True)['model'])
    detector.eval(); refiner.eval()
    return detector, refiner, parts


def grid_inference(detector, refiner, image, device, config):
    """Apply exactly the grid's normal prediction and same-model empty retry."""
    handoff.DETECTOR_SCORE_THRESHOLD = config['detector_score']
    handoff.CANDIDATE_REFINER_THRESHOLD = config['refiner_threshold']
    handoff.REFINER_TTA_THRESHOLD = config['refiner_threshold']
    detector.roi_heads.score_thresh = 0.05
    _, _, masks, _ = handoff.infer_observation(detector, refiner, image, device=device)
    masks = grid_pooled.postprocess(masks, config['min_area'], config['close_radius'])
    if not masks:
        detector.roi_heads.score_thresh = 0.001
        handoff.DETECTOR_SCORE_THRESHOLD = 0.01
        _, _, masks, _ = handoff.infer_observation(detector, refiner, image, device=device)
        masks = grid_pooled.postprocess(masks, config['min_area'], config['close_radius'])
    return masks


def read_grid_config(path):
    document = json.loads(path.read_text(encoding='utf8'))
    config = document.get('config', document)
    if set(config) != {'detector_score', 'refiner_threshold', 'min_area', 'close_radius'}:
        raise ValueError('Invalid grid configuration keys')
    key = ((config['detector_score'], config['refiner_threshold']), config['min_area'], config['close_radius'])
    if key not in grid_pooled.CONFIGS:
        raise ValueError('Configuration was not in the evaluated grid')
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--profile', type=Path, help='explicit trusted checkpoint hashes for another trained model')
    parser.add_argument('--fallback-weights', type=Path, help='verified fold-0 model for otherwise empty images')
    parser.add_argument('--detector-score', type=float, choices=(0.7, 0.8), default=0.7)
    parser.add_argument('--grid-config', type=Path, help='selected grid result or config JSON')
    args = parser.parse_args()
    if args.grid_config and args.fallback_weights:
        raise ValueError('Grid evaluation only used same-model fallback')
    grid_config = read_grid_config(args.grid_config) if args.grid_config else None
    expected_detector = '4409a5d50d18139097693c26fd2f06efa19a6463606d9c16eceda77cfe5ee2b2'
    expected_refiner = '95ff4976ae06e7022c1f207290a355fdf5a8758a1d47dd660b9bb1f4e3ecfdc4'
    if args.profile:
        profile = json.loads(args.profile.read_text(encoding='utf8'))
        expected_detector = profile['detector_state_sha256']
        expected_refiner = profile['refiner_sha256']
    inventory = json.loads(args.inventory.read_text(encoding='utf8'))
    print('inventory schema', type(inventory).__name__, flush=True)
    expected = inventory['files']
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {'.jpeg', '.jpg', '.png'})
    if len(images) != 180 or sorted(p.name for p in images) != sorted(expected):
        raise ValueError('Test image inventory mismatch')
    if args.output.exists():
        raise FileExistsError(args.output)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detector, refiner, parts = load_models(args.weights, expected_detector, expected_refiner, device)
    fallback = None
    if args.fallback_weights:
        fallback = load_models(args.fallback_weights,
                               '4409a5d50d18139097693c26fd2f06efa19a6463606d9c16eceda77cfe5ee2b2',
                               '95ff4976ae06e7022c1f207290a355fdf5a8758a1d47dd660b9bb1f4e3ecfdc4', device)
    handoff.DETECTOR_SCORE_THRESHOLD = args.detector_score
    rows, image_hashes, fallback_images = [], {}, []
    for index, path in enumerate(images):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048):
            raise ValueError(f'Invalid test image: {path.name}')
        image_hashes[path.name] = sha256(path)
        if grid_config is not None:
            masks = grid_inference(detector, refiner, image, device, grid_config)
        else:
            _, _, masks, _ = handoff.infer_observation(detector, refiner, image, device=device)
            masks = [mask for mask in masks if int(mask_utils.area(mask)) >= 256]
        if not masks and grid_config is None:
            old_score = handoff.DETECTOR_SCORE_THRESHOLD
            old_floor = detector.roi_heads.score_thresh
            try:
                detector.roi_heads.score_thresh = 0.001
                handoff.DETECTOR_SCORE_THRESHOLD = 0.01
                _, _, masks, _ = handoff.infer_observation(detector, refiner, image, device=device)
                masks = [mask for mask in masks if int(mask_utils.area(mask)) >= 256]
            finally:
                handoff.DETECTOR_SCORE_THRESHOLD = old_score
                detector.roi_heads.score_thresh = old_floor
        if not masks:
            if fallback is not None:
                fallback_detector, fallback_refiner, _ = fallback
                old_score = handoff.DETECTOR_SCORE_THRESHOLD
                try:
                    handoff.DETECTOR_SCORE_THRESHOLD = 0.8
                    _, _, masks, _ = handoff.infer_observation(fallback_detector, fallback_refiner, image, device=device)
                    masks = [mask for mask in masks if int(mask_utils.area(mask)) >= 256]
                    if not masks:
                        fallback_detector.roi_heads.score_thresh = 0.001
                        handoff.DETECTOR_SCORE_THRESHOLD = 0.01
                        _, _, masks, _ = handoff.infer_observation(fallback_detector, fallback_refiner, image, device=device)
                        masks = [mask for mask in masks if int(mask_utils.area(mask)) >= 256]
                finally:
                    handoff.DETECTOR_SCORE_THRESHOLD = old_score
                    fallback_detector.roi_heads.score_thresh = 0.05
                print(f'fold0 fallback: {path.name}, masks={len(masks)}', flush=True)
                fallback_images.append(path.name)
        if not masks:
            raise ValueError(f'No predictions for {path.name}')
        rows.append((path.stem, masks))
        if (index + 1) % 10 == 0:
            print(f'inferred {index + 1}/{len(images)}', flush=True)
    with args.output.open('x', newline='', encoding='utf8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['filament_id', 'segmentation_rle'])
        for stem, masks in rows:
            assert_disjoint(masks)
            for index, mask in enumerate(masks, 1):
                writer.writerow([f'{stem}_{index}', mask['counts'].decode('ascii')])
    fallback_provenance = None
    if fallback is not None:
        fallback_provenance = {'detector_part_sha256': [sha256(path) for path in fallback[2]],
                               'refiner_sha256': sha256(args.fallback_weights / 'final_refiner.pt'),
                               'detector_score': 0.8, 'retry_floor': 0.001, 'retry_score': 0.01,
                               'minimum_area': 256, 'images': fallback_images}
    print(json.dumps({'output_sha256': sha256(args.output), 'images': len(images), 'image_sha256': image_hashes,
                      'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'device': str(device), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      'detector_part_sha256': [sha256(path) for path in parts],
                      'refiner_sha256': sha256(args.weights / 'final_refiner.pt'),
                      'detector_score': grid_config['detector_score'] if grid_config else args.detector_score,
                      'refiner_threshold': grid_config['refiner_threshold'] if grid_config else 0.7,
                      'retry_floor': 0.001, 'retry_score': 0.01,
                      'minimum_area': grid_config['min_area'] if grid_config else 256,
                      'close_radius': grid_config['close_radius'] if grid_config else 0,
                      'grid_config': grid_config,
                      'grid_config_sha256': sha256(args.grid_config) if args.grid_config else None,
                      'grid_source_sha256': sha256(Path(grid_pooled.__file__)) if grid_config else None,
                      'fallback': fallback_provenance}), flush=True)


if __name__ == '__main__':
    main()
