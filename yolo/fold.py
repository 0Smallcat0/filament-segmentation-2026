"""Train a single group-held-out YOLO11-seg fold on private Kaggle GPU."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from ultralytics import YOLO
from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices
from src.postprocess import assert_disjoint
from src.provenance import sha256_file
from yolo.common import polygon_from_rle, disjoint


def prepare_data(records, image_dir, work, fold):
    """Only the other four folds enter YOLO train and its internal validation."""
    root = work / 'dataset'
    images, labels = root / 'images' / 'train', root / 'labels' / 'train'
    images.mkdir(parents=True, exist_ok=False)
    labels.mkdir(parents=True, exist_ok=False)
    stats = {'annotations': 0, 'dropped_degenerate': 0}
    for rec in records:
        if rec['fold'] == fold:
            continue
        filename = Path(rec['filename'])
        if filename.name != str(filename):
            raise ValueError(f'unsafe image filename: {filename}')
        source = image_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        (images / filename).symlink_to(source.resolve())
        lines = []
        for rle in rec['rles']:
            stats['annotations'] += 1
            polygon = polygon_from_rle(rle)
            if polygon is None:
                stats['dropped_degenerate'] += 1
            else:
                lines.append(polygon)
        (labels / (filename.stem + '.txt')).write_text('\n'.join(lines), encoding='utf8')
    data = root / 'data.yaml'
    data.write_text(f'path: {root}\ntrain: images/train\nval: images/train\nnames:\n  0: filament\n', encoding='utf8')
    return data, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=Path, required=True)
    parser.add_argument('--train-images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--batch', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261109)
    args = parser.parse_args()
    if not os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or not torch.cuda.is_available():
        raise RuntimeError('training is restricted to Kaggle GPU')
    start = time.monotonic()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    records = prepare_train(args.train_json, seed=args.seed)
    if len({Path(r['filename']).stem for r in records}) != len(records):
        raise ValueError('image stems must be unique')
    train = [r for r in records if r['fold'] != args.fold]
    held = [r for r in records if r['fold'] == args.fold]
    if not train or not held: raise ValueError('empty fold')
    args.output.mkdir(parents=True, exist_ok=False)
    # Store per-epoch checkpoints under Kaggle output, not ephemeral scratch.
    # On a failed/interrupted version, inspect the actual retained outputs before
    # assuming they are available; the final manifest is written only on success.
    environment = subprocess.run([sys.executable, '-m', 'pip', 'freeze', '--all'],
                                 check=True, capture_output=True, text=True).stdout
    (args.output / 'environment.txt').write_text(environment, encoding='utf8')
    image_hashes = {r['filename']: sha256_file(args.train_images / r['filename']) for r in train}
    scratch = Path('/kaggle/temp') / f'filament-yolo-fold{args.fold}'
    scratch.mkdir(parents=True, exist_ok=False)
    yaml, stats = prepare_data(records, args.train_images, scratch, args.fold)
    # COCO pretrained generic visual weights only; no solar data or test labels.
    model = YOLO('yolo11s-seg.pt')
    pretrained = Path(model.ckpt_path)
    if not pretrained.is_file(): raise FileNotFoundError(f'pretrained weights: {pretrained}')
    pretrained_hash = sha256_file(pretrained)
    train_options = {'data': str(yaml), 'epochs': args.epochs, 'imgsz': args.size,
                     'batch': args.batch, 'device': 0, 'patience': 0, 'seed': args.seed,
                     'deterministic': True, 'single_cls': True, 'overlap_mask': False,
                     'val': False, 'workers': 2, 'amp': False, 'save': True,
                     'save_period': 1, 'project': str(args.output / 'runs'),
                     'name': 'train', 'exist_ok': False}
    (args.output / 'requested_training_options.json').write_text(json.dumps(train_options, indent=2), encoding='utf8')
    model.train(**train_options)
    actual_options = json.loads(json.dumps(vars(model.trainer.args), default=str))
    (args.output / 'resolved_training_options.json').write_text(json.dumps(actual_options, indent=2), encoding='utf8')
    last = args.output / 'runs' / 'train' / 'weights' / 'last.pt'
    if not last.is_file(): raise FileNotFoundError(last)
    final = args.output / f'fold{args.fold}.pt'
    shutil.copyfile(last, final)
    model = YOLO(str(final))
    raw_rows, matrices, rows, processed = {}, [], [], []
    for index, rec in enumerate(held):
        image_path = args.train_images / rec['filename']
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (2048, 2048):
            raise ValueError(f'invalid train image {image_path}')
        result = model.predict(source=image, imgsz=args.size, conf=0.01, iou=0.7,
                               max_det=300, retina_masks=True, device=0, verbose=False)[0]
        raw = []
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            if len(masks) != len(scores): raise ValueError('inconsistent detector outputs')
            for mask, score in zip(masks, scores):
                if mask.shape != image.shape[:2]: raise ValueError('incorrect mask shape')
                encoded = mask_utils.encode(np.asfortranarray((mask > 0.5).astype(np.uint8)))
                raw.append({'size': encoded['size'], 'counts': encoded['counts'].decode('ascii'), 'score': float(score)})
        stem = Path(rec['filename']).stem
        raw_rows[stem] = raw
        # These are provisional fixed defaults, not tuned or submission-ready.
        rles = disjoint(raw, confidence=0.25, min_area=100, close_radius=0)
        assert_disjoint(rles)
        rows.append((stem, rles)); processed.append(stem)
        matrices.append(overlap_matrix(rec['rles'], rles))
        if (index + 1) % 10 == 0: print('OOF', index + 1, '/', len(held), flush=True)
    with (args.output / f'oof_fold{args.fold}.csv').open('w', encoding='utf8', newline='') as f:
        writer = csv.writer(f); writer.writerow(['filament_id', 'segmentation_rle'])
        for stem, rles in rows:
            for index, rle in enumerate(rles, 1): writer.writerow([f'{stem}_{index}', rle['counts'].decode('ascii')])
    (args.output / f'raw_fold{args.fold}.json').write_text(json.dumps(raw_rows), encoding='utf8')
    (args.output / f'processed_fold{args.fold}.json').write_text(json.dumps(processed), encoding='utf8')
    (args.output / f'score_fold{args.fold}.json').write_text(json.dumps(score_matrices(matrices), indent=2), encoding='utf8')
    root = Path(__file__).resolve().parents[1]
    sources = [*sorted((root / 'src').glob('*.py')), *sorted((root / 'yolo').glob('*.py'))]
    metadata = {'fold': args.fold, 'seed': args.seed, 'epochs': args.epochs, 'size': args.size,
                'batch': args.batch, 'train_json_sha256': sha256_file(args.train_json),
                'pretrained_sha256': pretrained_hash, 'checkpoint_sha256': sha256_file(final),
                'raw_sha256': sha256_file(args.output / f'raw_fold{args.fold}.json'),
                'training_filenames': sorted(r['filename'] for r in train),
                'training_image_sha256': image_hashes,
                'held_out_filenames': sorted(r['filename'] for r in held),
                'source_sha256': {str(p.relative_to(root)): sha256_file(p) for p in sources},
                'polygon_conversion': stats, 'python': platform.python_version(),
                'ultralytics': importlib.metadata.version('ultralytics'),
                'torch': torch.__version__, 'cuda': torch.version.cuda,
                'cudnn': torch.backends.cudnn.version(),
                'environment_sha256': sha256_file(args.output / 'environment.txt'),
                'resolved_training_options_sha256': sha256_file(args.output / 'resolved_training_options.json'),
                'resolved_training_options': actual_options,
                'gpu': torch.cuda.get_device_name(0), 'seconds': time.monotonic() - start,
                'proposal_conf': 0.01, 'proposal_iou': 0.7, 'proposal_max_det': 300,
                'provisional_defaults': {'confidence': 0.25, 'min_area': 100, 'close_radius': 0}}
    (args.output / f'manifest_fold{args.fold}.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    print('completed', json.dumps(metadata), flush=True)


if __name__ == '__main__': main()
