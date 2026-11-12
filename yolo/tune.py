"""Tune provisional YOLO postprocessing from five held-out raw outputs."""
import argparse
import itertools
import json
from pathlib import Path

from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices
from src.provenance import sha256_file
from yolo.common import disjoint


def pooled(folds):
    return score_matrices([matrix for fold in folds for matrix in fold])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-json', required=True, type=Path)
    p.add_argument('--runs', required=True, type=Path, help='contains fold0 ... fold4 directories')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=20261109)
    args = p.parse_args()
    records = prepare_train(args.train_json, seed=args.seed)
    shared = None
    raw_by_fold = []
    image_hashes = {}
    fold_artifacts = []
    for fold in range(5):
        directory = args.runs / f'fold{fold}'
        metadata = json.loads((directory / f'manifest_fold{fold}.json').read_text(encoding='utf8'))
        expected_train = sorted(r['filename'] for r in records if r['fold'] != fold)
        expected_held = sorted(r['filename'] for r in records if r['fold'] == fold)
        if (metadata['fold'] != fold or metadata['seed'] != args.seed
                or metadata['train_json_sha256'] != sha256_file(args.train_json)
                or metadata['training_filenames'] != expected_train
                or metadata['held_out_filenames'] != expected_held):
            raise ValueError(f'mismatched fold or data: {directory}')
        if set(metadata['training_image_sha256']) != set(expected_train):
            raise ValueError(f'incomplete training image hashes: {directory}')
        for filename, digest in metadata['training_image_sha256'].items():
            if filename in image_hashes and image_hashes[filename] != digest:
                raise ValueError(f'training image changed between folds: {filename}')
            image_hashes[filename] = digest
        for name, key in [(f'fold{fold}.pt', 'checkpoint_sha256'), (f'raw_fold{fold}.json', 'raw_sha256'),
                          ('environment.txt', 'environment_sha256'),
                          ('resolved_training_options.json', 'resolved_training_options_sha256')]:
            if sha256_file(directory / name) != metadata[key]: raise ValueError(f'hash mismatch: {directory / name}')
        actual_options = json.loads((directory / 'resolved_training_options.json').read_text(encoding='utf8'))
        if actual_options != metadata['resolved_training_options']:
            raise ValueError(f'resolved options mismatch: {directory}')
        # Per-fold scratch and output directory names are intentionally different.
        comparable_options = {k: v for k, v in actual_options.items() if k not in ('data', 'project', 'save_dir')}
        keys = ('seed', 'epochs', 'size', 'batch', 'pretrained_sha256', 'source_sha256', 'python', 'ultralytics', 'torch', 'cuda', 'cudnn', 'gpu', 'proposal_conf', 'proposal_iou', 'proposal_max_det', 'environment_sha256')
        signature = {key: metadata[key] for key in keys}
        signature['resolved_options'] = comparable_options
        if shared is None: shared = signature
        elif shared != signature: raise ValueError(f'mixed source, configuration or environment: {directory}')
        raw = json.loads((directory / f'raw_fold{fold}.json').read_text(encoding='utf8'))
        if set(raw) != {Path(name).stem for name in expected_held}:
            raise ValueError(f'incomplete fold predictions: {directory}')
        raw_by_fold.append(raw)
        fold_artifacts.append({'fold': fold, 'manifest_sha256': sha256_file(directory / f'manifest_fold{fold}.json'),
                               'checkpoint_sha256': metadata['checkpoint_sha256'], 'raw_sha256': metadata['raw_sha256']})
    if set(image_hashes) != {r['filename'] for r in records}:
        raise ValueError('training image hashes do not cover all observations')
    settings = list(itertools.product((0.05, 0.15, 0.25, 0.4), (100, 500, 2000), (0, 2, 5)))
    matrices = [[[] for _ in range(5)] for _ in settings]
    for index, record in enumerate(records):
        fold = record['fold']; raw = raw_by_fold[fold][Path(record['filename']).stem]
        for i, (confidence, area, radius) in enumerate(settings):
            masks = disjoint(raw, confidence, area, radius)
            matrices[i][fold].append(overlap_matrix(record['rles'], masks))
        if (index + 1) % 25 == 0: print('processed', index + 1, '/', len(records), flush=True)
    pooled_scores = [pooled(group) for group in matrices]
    best = max(range(len(settings)), key=lambda i: pooled_scores[i]['one_to_one']['pq'])
    chosen = []
    for held in range(5):
        i = max(range(len(settings)), key=lambda j: pooled([matrices[j][fold] for fold in range(5) if fold != held])['one_to_one']['pq'])
        chosen.append(i)
    cross = pooled([matrices[chosen[fold]][fold] for fold in range(5)])
    def label(i):
        confidence, area, radius = settings[i]
        return {'confidence': confidence, 'min_area': area, 'close_radius': radius}
    result = {'observations': len(records), 'fold_artifacts': fold_artifacts,
              'selection_metric': 'strict one-to-one pooled PQ at IoU > 0.5',
              'cross_fitted_score': cross,
              'settings_by_held_out_fold': [label(i) for i in chosen],
              'all_folds_setting': label(best), 'all_folds_score_selection_biased': pooled_scores[best],
              'limitation': 'Other fold models may have trained on the evaluated fold; cross-fitting is not fully nested validation. The raw proposal NMS and proposal threshold were fixed, not tuned.',
              'runs': str(args.runs)}
    args.output.write_text(json.dumps(result, indent=2), encoding='utf8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
