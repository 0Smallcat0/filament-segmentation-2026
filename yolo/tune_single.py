"""Exploratory threshold grid on one YOLO OOF fold; scores are selection-biased."""
import argparse
import json
from itertools import product
from pathlib import Path

from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices
from src.provenance import sha256_file
from yolo.common import disjoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.run / f'manifest_fold{args.fold}.json').read_text(encoding='utf8'))
    records = prepare_train(args.train_json)
    held = [r for r in records if r['fold'] == args.fold]
    expected_held = sorted(r['filename'] for r in held)
    expected_train = sorted(r['filename'] for r in records if r['fold'] != args.fold)
    raw_path, checkpoint = args.run / f'raw_fold{args.fold}.json', args.run / f'fold{args.fold}.pt'
    if (manifest['fold'] != args.fold or manifest['train_json_sha256'] != sha256_file(args.train_json)
            or manifest['held_out_filenames'] != expected_held or manifest['training_filenames'] != expected_train
            or manifest['raw_sha256'] != sha256_file(raw_path)
            or manifest['checkpoint_sha256'] != sha256_file(checkpoint)):
        raise ValueError('YOLO fold provenance mismatch')
    raw = json.loads(raw_path.read_text(encoding='utf8'))
    if set(raw) != {Path(r['filename']).stem for r in held}:
        raise ValueError('Raw predictions do not cover held-out observations')
    grid = list(product((0.05, 0.15, 0.25, 0.4), (100, 500), (0,)))
    matrices = [[] for _ in grid]
    for index, record in enumerate(held):
        candidates = raw[Path(record['filename']).stem]
        for i, (confidence, area, radius) in enumerate(grid):
            matrices[i].append(overlap_matrix(record['rles'], disjoint(candidates, confidence, area, radius)))
        if (index + 1) % 10 == 0:
            print(f'{index + 1}/{len(held)}', flush=True)
    result = []
    for settings, values in zip(grid, matrices):
        score = score_matrices(values)
        score.pop('matched_iou')
        result.append({'confidence': settings[0], 'min_area': settings[1], 'close_radius': settings[2], 'score': score})
    args.output.write_text(json.dumps({'fold': args.fold, 'selection_biased': True, 'results': result}, indent=2), encoding='utf8')
    print(args.output.read_text(encoding='utf8'), flush=True)


if __name__ == '__main__':
    main()
