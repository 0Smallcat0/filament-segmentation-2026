"""Evaluate final mask-area filtering on held-out detector predictions."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools import mask as mask_utils
from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices


def evaluate(annotation, fold, csv_path):
    predictions = defaultdict(list)
    with csv_path.open(newline='', encoding='utf8') as stream:
        for row in csv.DictReader(stream):
            stem, _ = row['filament_id'].rsplit('_', 1)
            predictions[stem].append({'size': [2048, 2048], 'counts': row['segmentation_rle'].encode('ascii')})
    observations = [r for r in prepare_train(annotation) if r['fold'] == fold]
    allowed = {Path(r['filename']).stem for r in observations}
    if set(predictions) - allowed:
        raise ValueError('Prediction outside the selected fold')
    values = []
    for record in observations:
        rles = predictions[Path(record['filename']).stem]
        values.append((overlap_matrix(record['rles'], rles), np.array([int(mask_utils.area(x)) for x in rles])))
    return {str(area): score_matrices(matrix[:, sizes >= area] for matrix, sizes in values)
            for area in (0, 16, 64, 256, 1024, 4096)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    args = parser.parse_args()
    results = evaluate(args.train_json, args.fold, args.predictions)
    for result in results.values():
        result.pop('matched_iou')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
