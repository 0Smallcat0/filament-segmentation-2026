"""Four full-image detector views and a paired fold-0 OOF evaluation."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from baseline.grid_pooled import compact_score, postprocess, sha
from baseline.provenance import object_sha256
from src.data import prepare_train
from src.metrics import overlap_matrix

VIEWS = ((), (-1,), (-2,), (-2, -1))
MAX_CANDIDATES = 128  # Fail closed rather than silently discard detections.


def cluster_views(views, iou_threshold=0.5, mask_threshold=0.5):
    """Greedy score-ordered matching; one instance per view per cluster.

    Ties use view then within-view index. Compare against the current mean mask;
    retain singleton detections and average only present probabilities and scores.
    """
    candidates = []
    for view, (scores, masks) in enumerate(views):
        if len(scores) != len(masks):
            raise ValueError('Mismatched score/mask lengths')
        for index, (score, mask) in enumerate(zip(scores, masks)):
            probability = np.asarray(mask, dtype=np.float32)
            if probability.ndim != 2 or not np.isfinite(probability).all() or not np.isfinite(score):
                raise ValueError('Invalid detector output')
            binary = probability >= mask_threshold
            if binary.any():
                rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
                candidates.append((float(score), view, index, probability, rle))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    clusters = []
    for score, view, index, probability, rle in candidates:
        eligible = [(i, cluster) for i, cluster in enumerate(clusters) if view not in cluster['views']]
        best_index, best_iou = None, iou_threshold
        if eligible:
            overlaps = mask_utils.iou([rle], [cluster['rle'] for _, cluster in eligible], [0] * len(eligible))[0]
            for (cluster_index, _), overlap in zip(eligible, overlaps):
                if overlap > best_iou:
                    best_index, best_iou = cluster_index, float(overlap)
        if best_index is None:
            clusters.append({'scores': [score], 'views': {view}, 'sum': probability.copy(), 'rle': rle})
        else:
            cluster = clusters[best_index]
            cluster['scores'].append(score)
            cluster['views'].add(view)
            cluster['sum'] += probability
            binary = cluster['sum'] / len(cluster['scores']) >= mask_threshold
            cluster['rle'] = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    return [{'score': float(np.mean(c['scores'])), 'probability': c['sum'] / len(c['scores']),
             'views': sorted(c['views'])} for c in clusters]


class FlipDetector:
    def __init__(self, model):
        self.model = model
        self.roi_heads = model.roi_heads

    def __call__(self, images):
        import torch
        from baseline import handoff
        if len(images) != 1:
            raise ValueError('Expected one full image')
        image = images[0]
        views = []
        candidate_count = 0
        for axes in VIEWS:
            transformed = torch.flip(image, axes) if axes else image
            with torch.inference_mode():
                out = self.model([transformed])[0]
            keep = out['scores'] >= handoff.DETECTOR_SCORE_THRESHOLD
            candidate_count += len(out['scores'][keep])
            if candidate_count > MAX_CANDIDATES:
                raise RuntimeError(f'Detector TTA exceeds {MAX_CANDIDATES} candidates; refusing unbounded memory use')
            scores = out['scores'][keep].detach().cpu().numpy()
            masks = out['masks'][keep, 0]
            if axes:
                masks = torch.flip(masks, axes)
            views.append((scores, masks.detach().cpu().numpy()))
        clusters = cluster_views(views, mask_threshold=handoff.DETECTOR_MASK_THRESHOLD)
        if not clusters:
            return [{'scores': torch.empty(0, device=image.device),
                     'masks': torch.empty((0, 1, *image.shape[-2:]), device=image.device)}]
        return [{'scores': torch.tensor([c['score'] for c in clusters], dtype=torch.float32, device=image.device),
                 'masks': torch.from_numpy(np.stack([c['probability'] for c in clusters])[:, None]).to(image.device)}]


def infer(detector, refiner, image, device):
    from baseline import handoff
    handoff.CANDIDATE_REFINER_THRESHOLD = 0.7
    handoff.REFINER_TTA_THRESHOLD = 0.7
    handoff.DETECTOR_SCORE_THRESHOLD = 0.8
    detector.roi_heads.score_thresh = 0.05
    _, _, masks, _ = handoff.infer_observation(detector, refiner, image, device=device)
    masks = postprocess(masks, 256, 0)
    if not masks:
        detector.roi_heads.score_thresh = 0.001
        handoff.DETECTOR_SCORE_THRESHOLD = 0.01
        _, _, masks, _ = handoff.infer_observation(detector, refiner, image, device=device)
        masks = postprocess(masks, 256, 0)
    return masks


def main():
    import torch
    import torchvision
    from baseline import handoff, provenance
    from baseline import grid_pooled
    from src import data, metrics, postprocess as postprocess_module
    p = argparse.ArgumentParser()
    for name in ('train-json', 'images', 'weights', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    records = prepare_train(args.train_json)
    held = [r for r in records if r['fold'] == 0]
    expected = {r['filename'] for r in held}
    manifest_path = args.weights / 'provenance_fold0.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    dataset = manifest['dataset']
    if (manifest['status'] != 'complete' or manifest['fold'] != 0
            or manifest['settings'].get('training_scope') == 'all_competition_train_observations'
            or dataset['annotation_sha256'] != sha(args.train_json)
            or dataset['sha256'] != object_sha256({k: v for k, v in dataset.items() if k != 'sha256'})
            or len(manifest['held_out_filenames']) != len(expected)
            or set(manifest['held_out_filenames']) != expected
            or set(manifest['train_filenames']) != {r['filename'] for r in records} - expected):
        raise ValueError('Invalid held-out provenance')
    for field in ('settings', 'environment'):
        if object_sha256(manifest[field]) != manifest[field + '_sha256']:
            raise ValueError(f'{field} manifest hash mismatch')
    if object_sha256(manifest['source_code']['files']) != manifest['source_code']['sha256']:
        raise ValueError('Source manifest hash mismatch')
    dataset_rows = [(r['filename'], str(r['medoid_image_id']), r['fold']) for r in dataset['observations']]
    prepared_rows = [(r['filename'], str(r['image_id']), r['fold']) for r in records]
    if len(dataset_rows) != len(prepared_rows) or set(dataset_rows) != set(prepared_rows):
        raise ValueError('Dataset filename/medoid/fold mismatch')
    for name in ('final_detector.pt', 'final_refiner.pt', 'training_manifest.csv'):
        if sha(args.weights / name) != manifest['outputs_sha256'][name]:
            raise ValueError(f'Training output hash mismatch: {name}')
    with (args.weights / 'training_manifest.csv').open(newline='', encoding='utf8') as stream:
        rows = list(csv.DictReader(stream))
    expected_train = {(str(r['image_id']), r['filename'], Path(r['filename']).stem)
                      for r in records if r['fold'] != 0}
    actual_train = {(r['id'], r['file_name'], r['observation_id']) for r in rows}
    if len(rows) != len(expected_train) or len(actual_train) != len(rows) or actual_train != expected_train:
        raise ValueError('Training manifest membership mismatch')
    hashes = {item['filename']: item['image_sha256'] for item in dataset['observations']}
    detector = handoff.build_detector().cuda()
    detector.load_state_dict(torch.load(args.weights / 'final_detector.pt', map_location='cpu', weights_only=True)['model'])
    refiner = handoff.CropUNet().cuda()
    refiner.load_state_dict(torch.load(args.weights / 'final_refiner.pt', map_location='cpu', weights_only=True)['model'])
    detector.eval()
    refiner.eval()
    device = torch.device('cuda:0')
    flipped = FlipDetector(detector)
    matrices = {'single': [], 'flip4': []}
    per_image = []
    started = time.monotonic()
    for i, record in enumerate(held):
        path = args.images / record['filename']
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != (2048, 2048) or sha(path) != hashes[record['filename']]:
            raise ValueError(f'Invalid image: {path}')
        result = {'filename': record['filename']}
        for label, model in [('single', detector), ('flip4', flipped)]:
            masks = infer(model, refiner, image, device)
            matrix = overlap_matrix(record['rles'], masks)
            matrices[label].append(matrix)
            result[label] = {'empty': not bool(masks), 'score': compact_score([matrix])}
        per_image.append(result)
        if (i + 1) % 10 == 0:
            print(f'fold 0: {i + 1}/{len(held)}', flush=True)
    environment = {'python': platform.python_version(), 'numpy': np.__version__, 'opencv': cv2.__version__,
                   'torch': torch.__version__, 'torchvision': torchvision.__version__, 'torch_cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0), 'scipy': importlib.metadata.version('scipy'),
                   'pycocotools': importlib.metadata.version('pycocotools')}
    sources = [__file__, handoff.__file__, grid_pooled.__file__, provenance.__file__, data.__file__,
               metrics.__file__, postprocess_module.__file__]
    results = {name: {'empty': sum(row[name]['empty'] for row in per_image), 'score': compact_score(matrix)}
               for name, matrix in matrices.items()}
    output = {'fold': 0, 'config': {'detector_score': 0.8, 'refiner_threshold': 0.7, 'min_area': 256,
              'close_radius': 0, 'fallback_floor': 0.001, 'fallback_score': 0.01},
              'clustering': 'score-descending, view/index tie break, current mean mask IoU>0.5, at most one per view; retain singletons',
              'selection_bias': 'Fold 0 informed earlier threshold evaluation and pooled selection; this comparison is not independent of that selection.',
              'candidate_limit': MAX_CANDIDATES,
              'annotation_sha256': sha(args.train_json), 'provenance_sha256': sha(manifest_path),
              'checkpoint_sha256': {n: sha(args.weights / n) for n in ('final_detector.pt', 'final_refiner.pt')},
              'source_sha256': {Path(x).name: sha(Path(x)) for x in sources}, 'environment': environment,
              'seconds': time.monotonic() - started, 'results': results, 'observations': per_image}
    args.output.write_text(json.dumps(output, indent=2), encoding='utf8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
