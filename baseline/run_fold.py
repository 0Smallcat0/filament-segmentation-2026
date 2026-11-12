"""Train a fresh held-out baseline fold on Kaggle, then run the 0.34 handoff."""
import argparse
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools.coco import COCO
from torch.utils.data import DataLoader

from src.data import prepare_train
from src.inference import write_submission
from src.metrics import overlap_matrix, score_matrices
from src.postprocess import assert_disjoint
from baseline import handoff, refiner, provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=range(5), required=True)
    parser.add_argument('--detector-epochs', type=int, default=6)
    parser.add_argument('--refiner-epochs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--full-train', action='store_true', help='train all observations without an OOF score')
    args = parser.parse_args()
    if not os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or torch.cuda.device_count() < 2:
        raise RuntimeError('This training is restricted to Kaggle GPU T4 x2')
    if any('T4' not in torch.cuda.get_device_name(i) for i in range(2)):
        raise RuntimeError('Expected two T4 GPUs')
    started = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    data_root = Path(os.environ['FILAMENT_DATA_ROOT'])
    train_root = data_root / 'train'
    annotation = train_root / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
    image_root = train_root / 'train_images'
    work = Path('/kaggle/working/solar_filaments_final')
    if work.exists() and any(work.iterdir()):
        raise FileExistsError(f'Refusing to reuse nonempty output directory: {work}')
    work.mkdir(parents=True, exist_ok=True)
    records = prepare_train(annotation, seed=args.seed)
    if len({Path(r['filename']).stem for r in records}) != len(records):
        raise ValueError('Observation stems must be unique')
    train = records if args.full_train else [r for r in records if r['fold'] != args.fold]
    held_out = [] if args.full_train else [r for r in records if r['fold'] == args.fold]
    assert len(train) + len(held_out) == len(records) and train and (held_out or args.full_train)
    assert not ({r['filename'] for r in train} & {r['filename'] for r in held_out})
    settings = {'fold_seed': args.seed, 'detector_seed': 2026,
                'detector_epochs': args.detector_epochs, 'refiner_epochs': args.refiner_epochs,
                'detector_gpus': 2, 'detector_batch_per_gpu': 1, 'refiner_batch': 16,
                'detector_score_threshold': handoff.DETECTOR_SCORE_THRESHOLD,
                'detector_mask_threshold': handoff.DETECTOR_MASK_THRESHOLD,
                'refiner_threshold': handoff.CANDIDATE_REFINER_THRESHOLD,
                'model': 'maskrcnn-resnet50-fpn-v2+resnet18-crop-unet',
                'inference': 'handoff-topology-safe-flip4-tta'}
    if args.full_train:
        settings['training_scope'] = 'all_competition_train_observations'
    versions = {}
    for name in ('torch', 'torchvision', 'numpy', 'pandas', 'opencv-python', 'opencv-python-headless',
                 'pycocotools', 'scipy', 'segmentation-models-pytorch'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    versions.update({'python': sys.version, 'cuda_runtime': torch.version.cuda, 'opencv_runtime': cv2.__version__,
                     'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    run_manifest = {'schema': 1, 'fold': args.fold, 'source_dataset': 'Kaggle competition filament-segmentation-2026 train',
                    'dataset': provenance.dataset_manifest(annotation, image_root, records),
                    'source_code': provenance.source_manifest(Path(__file__).resolve().parents[1]),
                    'settings': settings, 'settings_sha256': provenance.object_sha256(settings),
                    'environment': versions, 'environment_sha256': provenance.object_sha256(versions),
                    'train_filenames': sorted(r['filename'] for r in train),
                    'held_out_filenames': sorted(r['filename'] for r in held_out),
                    'status': 'training'}
    provenance_path = work / f'provenance_fold{args.fold}.json'
    provenance_path.write_text(json.dumps(run_manifest, indent=2), encoding='utf8')
    manifest = pd.DataFrame([{'id': r['image_id'], 'file_name': r['filename'],
                              'observation_id': Path(r['filename']).stem} for r in train])
    manifest.to_csv(work / 'training_manifest.csv', index=False)
    # The upstream final mode trains on exactly the supplied manifest.
    # Its validation branch is never invoked.
    subprocess.run(['torchrun', '--standalone', '--nproc_per_node=2',
                    str(Path(__file__).with_name('detector.py')), '--mode', 'final',
                    '--epochs', str(args.detector_epochs)], check=True)
    refiner.WORK_DIR, refiner.TRAIN_DIR, refiner.coco = work, image_root, COCO(str(annotation))
    images, masks, count = refiner.build_crop_cache(manifest, 'train_only')
    if not count:
        raise RuntimeError('No training crops')
    dataset = refiner.CropDataset(images, masks, augment=True)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4,
                        pin_memory=True, persistent_workers=True)
    model = refiner.CropUNet(pretrained=True)
    refiner.train_refiner(model, loader, args.refiner_epochs, work / 'final_refiner.pt')
    del model
    if args.full_train:
        outputs = [work / 'training_manifest.csv', work / 'final_detector.pt', work / 'final_refiner.pt']
        run_manifest['outputs_sha256'] = {p.name: provenance.file_sha256(p) for p in outputs}
        run_manifest['status'] = 'complete'
        run_manifest['elapsed_seconds'] = time.monotonic() - started
        provenance_path.write_text(json.dumps(run_manifest, indent=2), encoding='utf8')
        print('Full-train weights complete; no OOF score is claimed', flush=True)
        return
    torch.cuda.empty_cache()
    device = torch.device('cuda:0')
    detector = handoff.build_detector().to(device)
    detector.load_state_dict(torch.load(work / 'final_detector.pt', map_location=device,
                                        weights_only=True)['model'], strict=True)
    crop_model = handoff.CropUNet().to(device)
    crop_model.load_state_dict(torch.load(work / 'final_refiner.pt', map_location=device,
                                          weights_only=True)['model'], strict=True)
    detector.eval(); crop_model.eval()
    rows, matrices, processed = [], [], []
    for index, record in enumerate(held_out):
        image = cv2.imread(str(image_root / record['filename']), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(record['filename'])
        _, _, pred, _ = handoff.infer_observation(detector, crop_model, image, device=device)
        assert_disjoint(pred)
        stem = Path(record['filename']).stem
        rows.append((stem, pred)); processed.append(stem)
        matrices.append(overlap_matrix(record['rles'], pred))
        if (index + 1) % 10 == 0:
            print(f'OOF {index + 1}/{len(held_out)}', flush=True)
    if len(set(processed)) != len(held_out):
        raise ValueError('Duplicate or missing held-out observations')
    write_submission(rows, work / f'oof_fold{args.fold}.csv')
    (work / f'processed_fold{args.fold}.json').write_text(json.dumps(processed), encoding='utf8')
    report = {'fold': args.fold, 'seed': args.seed, 'detector_epochs': args.detector_epochs,
              'refiner_epochs': args.refiner_epochs, 'train_count': len(train),
              'held_out_count': len(held_out), 'elapsed_seconds': time.monotonic()-started,
              'score': score_matrices(matrices)}
    (work / f'score_fold{args.fold}.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    outputs = [work / f'oof_fold{args.fold}.csv', work / f'processed_fold{args.fold}.json',
               work / f'score_fold{args.fold}.json', work / 'training_manifest.csv',
               work / 'final_detector.pt', work / 'final_refiner.pt']
    run_manifest['outputs_sha256'] = {p.name: provenance.file_sha256(p) for p in outputs}
    run_manifest['status'] = 'complete'
    provenance_path.write_text(json.dumps(run_manifest, indent=2), encoding='utf8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
