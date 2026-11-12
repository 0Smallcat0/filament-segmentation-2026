"""Tune semantic postprocessing against five independently held-out folds."""
import argparse
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices
from src.postprocess import instances
from src.provenance import sha256_file


def validate_manifests(root, records, train_json, seed):
    """Reject incomplete, changed, or mixed-version folds before evaluating."""
    root = Path(root)
    expected_input_hash = sha256_file(train_json)
    shared = None
    identities = []
    for fold in range(5):
        manifest_path = root / f'manifest_fold{fold}.json'
        if not manifest_path.is_file():
            raise ValueError(f'missing {manifest_path}: legacy/incomplete runs cannot be mixed or scored')
        manifest = json.loads(manifest_path.read_text(encoding='utf8'))
        if manifest.get('schema') != 1 or manifest.get('fold') != fold or manifest.get('folds') != 5 or manifest.get('seed') != seed:
            raise ValueError(f'invalid fold metadata: {manifest_path}')
        if manifest.get('train_json_sha256') != expected_input_hash:
            raise ValueError(f'train annotation hash mismatch: {manifest_path}')
        val = [r for r in records if r['fold'] == fold]
        if manifest.get('validation_filenames') != sorted(r['filename'] for r in val):
            raise ValueError(f'validation split mismatch: {manifest_path}')
        signatures = {key: manifest.get(key) for key in ('schema', 'folds', 'seed', 'config', 'source_sha256', 'train_json_sha256', 'probability_format')}
        env = manifest.get('environment', {})
        signatures['environment'] = {key: env.get(key) for key in ('python', 'packages', 'cuda_runtime', 'cudnn', 'gpu_name')}
        if any(signatures[key] is None for key in ('config', 'source_sha256', 'probability_format')):
            raise ValueError(f'incomplete manifest: {manifest_path}')
        if shared is None: shared = signatures
        elif shared != signatures: raise ValueError(f'mixed training versions/configurations/environments: {manifest_path}')
        checkpoint = root / f'fold{fold}.pt'
        checkpoint_hash = sha256_file(checkpoint) if checkpoint.is_file() else None
        if checkpoint_hash != manifest.get('checkpoint_sha256') or checkpoint_hash is None:
            raise ValueError(f'checkpoint missing or modified: {checkpoint}')
        identities.append({'fold': fold, 'checkpoint': checkpoint.name, 'checkpoint_sha256': checkpoint_hash,
                           'manifest': manifest_path.name, 'manifest_sha256': sha256_file(manifest_path)})
        expected_maps = {Path(r['filename']).stem for r in val}
        map_hashes = manifest.get('probability_sha256', {})
        if set(map_hashes) != expected_maps:
            raise ValueError(f'probability manifest does not cover validation fold: {manifest_path}')
        directory = root / f'probabilities_fold{fold}'
        if {p.stem for p in directory.glob('*.png')} != expected_maps:
            raise ValueError(f'probability directory has missing or extra maps: {directory}')
        for stem, expected_hash in map_hashes.items():
            path = directory / f'{stem}.png'
            if sha256_file(path) != expected_hash: raise ValueError(f'probability map modified: {path}')
    return identities


def cross_fit_settings(settings, matrices, folds):
    """Choose settings on four folds and aggregate matches from the fifth.

    This avoids using the evaluated fold's OOF scores for setting selection.
    Other fold models may still have trained on that fold: this is not nested CV.
    """
    if not settings or len(settings) != len(matrices) or set(folds) != set(range(5)):
        raise ValueError('expected candidate settings and all five folds')
    if any(len(candidate) != len(folds) for candidate in matrices):
        raise ValueError('each candidate must cover all observations')
    chosen_matrices = []
    fold_results = []
    for fold in range(5):
        select = [i for i, value in enumerate(folds) if value != fold]
        held = [i for i, value in enumerate(folds) if value == fold]
        scores = [score_matrices(candidate[i] for i in select) for candidate in matrices]
        # max() resolves ties by the original grid order.
        chosen = max(range(len(settings)), key=lambda i: scores[i]['one_to_one']['pq'])
        held_matrices = [matrices[chosen][i] for i in held]
        chosen_matrices.extend(held_matrices)
        threshold, min_area, radius = settings[chosen]
        fold_results.append({'fold': fold, 'selection_folds': [i for i in range(5) if i != fold],
            'threshold': threshold, 'min_area': min_area, 'close_radius': radius,
            'selection_score': scores[chosen], 'held_out_observations': len(held),
            'held_out_score': score_matrices(held_matrices)})
    return {'folds': fold_results, 'score': score_matrices(chosen_matrices)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', required=True)
    parser.add_argument('--runs', required=True, help='directory with probabilities_fold0 through probabilities_fold4')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.3, 0.4, 0.5, 0.6])
    parser.add_argument('--min-areas', type=int, nargs='+', default=[100, 500, 2000])
    parser.add_argument('--close-radii', type=int, nargs='+', default=[0, 2, 5])
    args = parser.parse_args()
    records = prepare_train(args.train_json, seed=args.seed)
    if len({Path(r['filename']).stem for r in records}) != len(records):
        raise ValueError('observation stems must be distinct')
    identities = validate_manifests(args.runs, records, args.train_json, args.seed)
    settings = list(itertools.product(args.thresholds, args.min_areas, args.close_radii))
    matrices = [[] for _ in settings]
    for index, record in enumerate(records):
        path = Path(args.runs) / f"probabilities_fold{record['fold']}" / (Path(record['filename']).stem + '.png')
        raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if raw is None or raw.dtype != np.uint16 or raw.ndim != 2:
            raise ValueError(f'missing or invalid uint16 probability map: {path}')
        prob = raw.astype(np.float32) / 65535
        for n, (threshold, min_area, radius) in enumerate(settings):
            masks = instances(prob, threshold=threshold, min_area=min_area, close_radius=radius)
            matrices[n].append(overlap_matrix(record['rles'], masks))
        if (index + 1) % 25 == 0: print(f'processed {index+1}/{len(records)}', flush=True)
    results = []
    for (threshold, min_area, radius), overlaps in zip(settings, matrices):
        score = score_matrices(overlaps)
        results.append({'threshold': threshold, 'min_area': min_area, 'close_radius': radius, 'score': score})
    results.sort(key=lambda result: result['score']['one_to_one']['pq'], reverse=True)
    cross_fitted = cross_fit_settings(settings, matrices, [record['fold'] for record in records])
    Path(args.output).write_text(json.dumps({'observations': len(records), 'seed': args.seed,
        'fold_artifacts': identities,
        'checkpoint_sha256': [identity['checkpoint_sha256'] for identity in identities],
        'selection_bias': 'The all-fold best result is optimistically biased: the same OOF maps select and score it.',
        'cross_fitted_limitation': 'Four-fold setting selection excludes the scored fold, but models for other folds may have trained on the scored fold. This is not fully nested cross-validation or an independent test.',
        'cross_fitted': cross_fitted, 'final_inference_setting': results[0], 'results': results}, indent=2), encoding='utf8')
    print(json.dumps({'cross_fitted': cross_fitted, 'final_inference_setting': results[0]}, indent=2))


if __name__ == '__main__':
    main()
