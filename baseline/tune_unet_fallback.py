"""Cross-model fallback comparison on one shared held-out fold."""
import argparse
import csv
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices
from src.postprocess import instances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--maps', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [r for r in prepare_train(args.train_json) if r['fold'] == args.fold]
    predictions = defaultdict(list)
    with args.predictions.open(newline='', encoding='utf8') as stream:
        for row in csv.DictReader(stream):
            stem, _ = row['filament_id'].rsplit('_', 1)
            rle = {'size': [2048, 2048], 'counts': row['segmentation_rle'].encode('ascii')}
            if int(mask_utils.area(rle)) >= 256:
                predictions[stem].append(rle)
    allowed = {Path(r['filename']).stem for r in records}
    if set(predictions) - allowed:
        raise ValueError('Predictions outside held-out fold')
    empty = [r for r in records if not predictions[Path(r['filename']).stem]]
    empty_stems = {Path(r['filename']).stem for r in empty}
    matrices = {Path(r['filename']).stem: overlap_matrix(r['rles'], predictions[Path(r['filename']).stem])
                for r in records if Path(r['filename']).stem not in empty_stems}
    maps = {}
    for record in empty:
        stem = Path(record['filename']).stem
        path = args.maps / (stem + '.png')
        probability = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if probability is None or probability.shape != (2048, 2048) or probability.dtype != np.uint16:
            raise ValueError(f'Invalid probability map {path}')
        maps[stem] = probability.astype(np.float32) / 65535
    results = []
    for threshold, area, radius in product((0.001, 0.002, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3), (16, 256, 1024), (0, 2)):
        filled = {Path(r['filename']).stem: instances(maps[Path(r['filename']).stem], threshold, area, radius)
                  for r in empty}
        m = dict(matrices)
        for r in empty:
            stem = Path(r['filename']).stem
            m[stem] = overlap_matrix(r['rles'], filled[stem])
        score = score_matrices(m.values())
        score.pop('matched_iou')
        results.append({'threshold': threshold, 'min_area': area, 'close_radius': radius,
                        'empty_before': len(empty), 'empty_after': sum(not x for x in filled.values()),
                        'score': score})
        print(json.dumps(results[-1]), flush=True)
    args.output.write_text(json.dumps(results, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
