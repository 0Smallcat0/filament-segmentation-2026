"""Verify held-out fold artifacts and pool sufficient statistics."""
import argparse
import json
from pathlib import Path

from baseline.provenance import file_sha256, object_sha256


def pool(directories):
    manifests = []
    for directory in directories:
        paths = list(directory.glob('provenance_fold*.json'))
        if len(paths) != 1:
            raise ValueError(f'Expected exactly one provenance manifest in {directory}')
        manifest = json.loads(paths[0].read_text(encoding='utf8'))
        fold = manifest['fold']
        if manifest['status'] != 'complete':
            raise ValueError('Fold was not completed')
        for name in (f'oof_fold{fold}.csv', f'processed_fold{fold}.json', f'score_fold{fold}.json'):
            if file_sha256(directory / name) != manifest['outputs_sha256'][name]:
                raise ValueError(f'Output hash mismatch: {directory / name}')
        for field in ('settings', 'environment'):
            if object_sha256(manifest[field]) != manifest[field + '_sha256']:
                raise ValueError(f'{field} hash mismatch')
        dataset = manifest['dataset']
        if object_sha256({k: v for k, v in dataset.items() if k != 'sha256'}) != dataset['sha256']:
            raise ValueError('Dataset manifest hash mismatch')
        if object_sha256(manifest['source_code']['files']) != manifest['source_code']['sha256']:
            raise ValueError('Source manifest hash mismatch')
        observations = dataset['observations']
        expected = {item['filename'] for item in observations if item['fold'] == fold}
        all_files = {item['filename'] for item in observations}
        held = manifest['held_out_filenames']
        training = manifest['train_filenames']
        if len(held) != len(expected) or set(held) != expected or set(training) != all_files - expected:
            raise ValueError('Split membership mismatch')
        processed = json.loads((directory / f'processed_fold{fold}.json').read_text(encoding='utf8'))
        if len(processed) != len(expected) or set(processed) != {Path(name).stem for name in expected}:
            raise ValueError('Processed observation mismatch')
        score = json.loads((directory / f'score_fold{fold}.json').read_text(encoding='utf8'))
        if score['fold'] != fold or score['held_out_count'] != len(expected):
            raise ValueError('Score fold metadata mismatch')
        manifests.append((manifest, score['score']))
    folds = [m['fold'] for m, _ in manifests]
    if len(folds) != len(set(folds)):
        raise ValueError('Duplicate fold')
    for name in ('dataset', 'source_code'):
        if len({m[name]['sha256'] for m, _ in manifests}) != 1:
            raise ValueError(f'Mixed {name} hashes')
    for name in ('settings_sha256', 'environment_sha256'):
        if len({m[name] for m, _ in manifests}) != 1:
            raise ValueError(f'Mixed {name}')
    result = {'folds': sorted(folds), 'observations': sum(len(m['held_out_filenames']) for m, _ in manifests)}
    for variant in ('official', 'one_to_one'):
        counts = {key: sum(score[variant][key] for _, score in manifests) for key in ('tp', 'fp', 'fn', 'sum_iou')}
        denominator = counts['tp'] + 0.5 * (counts['fp'] + counts['fn'])
        counts['pq'] = counts['sum_iou'] / denominator if denominator else 0.0
        result[variant] = counts
    for key in ('one_gt_to_many_predictions_iou_gt_0', 'many_gt_to_one_prediction_iou_gt_0'):
        result[key] = sum(score[key] for _, score in manifests)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps(pool(args.directories), indent=2))


if __name__ == '__main__':
    main()
