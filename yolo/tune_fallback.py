"""Compare YOLO fallback settings only on empty baseline OOF observations."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from itertools import product
from pathlib import Path

from pycocotools import mask as mask_utils
from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices
from yolo.common import disjoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', type=Path, required=True)
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--yolo-raw', type=Path, required=True)
    parser.add_argument('--yolo-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = [r for r in prepare_train(args.train_json) if r['fold'] == args.fold]
    manifest = json.loads(args.yolo_manifest.read_text(encoding='utf8'))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if (manifest['fold'] != args.fold or manifest['raw_sha256'] != sha(args.yolo_raw)
            or manifest['train_json_sha256'] != sha(args.train_json)):
        raise ValueError('YOLO provenance mismatch')
    predictions = defaultdict(list)
    with args.baseline.open(newline='', encoding='utf8') as stream:
        for row in csv.DictReader(stream):
            stem, _ = row['filament_id'].rsplit('_', 1)
            rle = {'size': [2048, 2048], 'counts': row['segmentation_rle'].encode('ascii')}
            if int(mask_utils.area(rle)) >= 256:
                predictions[stem].append(rle)
    allowed = {Path(r['filename']).stem for r in records}
    if set(manifest['held_out_filenames']) != {r['filename'] for r in records}:
        raise ValueError('YOLO held-out filenames mismatch')
    raw = json.loads(args.yolo_raw.read_text(encoding='utf8'))
    if set(raw) != allowed or set(predictions) - allowed:
        raise ValueError('Predictions do not match held-out observations')
    empty = [r for r in records if not predictions[Path(r['filename']).stem]]
    empty_stems = {Path(r['filename']).stem for r in empty}
    matrices = {Path(r['filename']).stem: overlap_matrix(r['rles'], predictions[Path(r['filename']).stem])
                for r in records if Path(r['filename']).stem not in empty_stems}
    results = []
    for confidence, area, radius in product((0.01, 0.05, 0.1, 0.25), (16, 100, 256), (0, 2)):
        filled = {stem: disjoint(raw[stem], confidence, area, radius) for stem in empty_stems}
        m = dict(matrices)
        for r in empty:
            stem = Path(r['filename']).stem
            m[stem] = overlap_matrix(r['rles'], filled[stem])
        score = score_matrices(m.values())
        score.pop('matched_iou')
        results.append({'confidence': confidence, 'min_area': area, 'close_radius': radius,
                        'empty_before': len(empty), 'empty_after': sum(not x for x in filled.values()),
                        'score': score})
    args.output.write_text(json.dumps(results, indent=2), encoding='utf8')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
