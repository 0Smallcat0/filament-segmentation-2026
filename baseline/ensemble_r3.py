"""Five independently trained folds; binary-mask consensus, no test labels."""
import argparse
import csv
import gc
import json
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils


def encode(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def cluster_masks(by_fold, threshold=0.5):
    """Greedy largest-first matching, at most one mask from each fold per cluster."""
    candidates = [(int(mask_utils.area(r)), fold, index, r)
                  for fold, masks in enumerate(by_fold) for index, r in enumerate(masks)]
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    clusters = []
    for _, fold, _, rle in candidates:
        eligible = [c for c in clusters if fold not in c['folds']]
        best, best_iou = None, threshold
        if eligible:
            for c, iou in zip(eligible, mask_utils.iou([rle], [c['representative'] for c in eligible], [0] * len(eligible))[0]):
                if iou > best_iou:
                    best, best_iou = c, float(iou)
        if best is None:
            clusters.append({'folds': {fold}, 'rles': [rle], 'representative': rle})
        else:
            best['folds'].add(fold)
            best['rles'].append(rle)
            counts = sum((mask_utils.decode(x).astype(np.uint8) for x in best['rles']))
            best['representative'] = encode(counts * 2 >= len(best['rles']))
    return clusters


def consensus(by_fold, k, min_area=256):
    clusters = [c for c in cluster_masks(by_fold) if len(c['folds']) >= k]
    if not clusters:
        return [], False
    shape = tuple(clusters[0]['rles'][0]['size'])
    owner = np.full(shape, -1, dtype=np.int16)
    best_score = np.zeros(shape, dtype=np.float32)
    for i, c in enumerate(clusters):
        votes = sum((mask_utils.decode(r).astype(np.uint8) for r in c['rles']))
        probability = votes.astype(np.float32) / len(c['rles'])
        update = (probability >= 0.5) & (probability > best_score)
        owner[update] = i
        best_score[update] = probability[update]
    masks = [encode(owner == i) for i in range(len(clusters)) if int(np.count_nonzero(owner == i)) >= min_area]
    return masks, bool(masks)


def main():
    import torch
    from baseline import handoff
    from baseline.detector_tta_r3 import infer
    from baseline.provenance import file_sha256
    from src.postprocess import assert_disjoint

    p = argparse.ArgumentParser()
    p.add_argument('--images', type=Path, required=True)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--profiles', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    images = sorted(x for x in args.images.iterdir() if x.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    inventory = json.loads(args.inventory.read_text(encoding='utf8'))['files']
    if len(images) != 180 or sorted(x.name for x in images) != sorted(inventory):
        raise ValueError('Test image inventory mismatch')
    profiles = json.loads(args.profiles.read_text(encoding='utf8'))
    if len(profiles) != 5 or not torch.cuda.is_available():
        raise ValueError('Exactly five folds and Kaggle GPU required')
    device = torch.device('cuda:0')
    predictions = [[] for _ in images]
    for fold, profile in enumerate(profiles):
        manifests = list(Path('/kaggle/input').rglob(f'provenance_fold{fold}.json'))
        if len(manifests) != 1 or file_sha256(manifests[0]) != profile['provenance_sha256']:
            raise ValueError(f'Fold {fold} provenance mismatch')
        folder = manifests[0].parent
        for filename in ('final_detector.pt', 'final_refiner.pt'):
            if file_sha256(folder / filename) != profile[filename]:
                raise ValueError(f'Fold {fold} checkpoint mismatch: {filename}')
        detector = handoff.build_detector().to(device)
        detector.load_state_dict(torch.load(folder / 'final_detector.pt', map_location='cpu', weights_only=True)['model'])
        refiner = handoff.CropUNet().to(device)
        refiner.load_state_dict(torch.load(folder / 'final_refiner.pt', map_location='cpu', weights_only=True)['model'])
        detector.eval(); refiner.eval()
        for i, path in enumerate(images):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.shape != (2048, 2048):
                raise ValueError(f'Invalid test image {path}')
            masks = infer(detector, refiner, image, device)
            assert_disjoint(masks)
            predictions[i].append(masks)
            if (i + 1) % 20 == 0:
                print(f'fold {fold}: {i + 1}/180', flush=True)
        del detector, refiner
        gc.collect(); torch.cuda.empty_cache()
    args.output.mkdir(parents=True, exist_ok=True)
    stats = {}
    for k in (2, 3):
        fallback = []
        with (args.output / f'submission-k{k}.csv').open('x', newline='', encoding='utf8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['filament_id', 'segmentation_rle'])
            for path, folds in zip(images, predictions):
                masks, ok = consensus(folds, k)
                if not ok:
                    masks = folds[0]
                    fallback.append(path.name)
                if not masks:
                    raise ValueError(f'No consensus or fold0 fallback for {path.name}')
                assert_disjoint(masks)
                for i, mask in enumerate(masks, 1):
                    writer.writerow([f'{path.stem}_{i}', mask['counts'].decode('ascii')])
        stats[str(k)] = {'fold0_fallback_images': fallback}
    (args.output / 'ensemble-stats.json').write_text(json.dumps(stats, indent=2), encoding='utf8')
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
