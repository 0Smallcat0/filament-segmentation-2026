"""Score held-out prediction CSV against medoid train records and emit JSON."""
import argparse, csv, json
from pathlib import Path
from collections import defaultdict
from src.data import prepare_train
from src.metrics import overlap_matrix, score_matrices
from pycocotools import mask as mask_utils

p = argparse.ArgumentParser()
p.add_argument('--train-json', required=True)
p.add_argument('--predictions', required=True)
p.add_argument('--fold', type=int, required=True)
p.add_argument('--folds', type=int, default=5)
p.add_argument('--seed', type=int, default=20261109)
args = p.parse_args()
records = [r for r in prepare_train(args.train_json, args.folds, args.seed) if r['fold'] == args.fold]
if not records: raise ValueError('fold has no observations')
preds = defaultdict(list)
with open(args.predictions, newline='', encoding='utf8') as f:
    reader = csv.DictReader(f)
    if reader.fieldnames != ['filament_id', 'segmentation_rle']: raise ValueError('unexpected columns')
    seen = set()
    for row in reader:
        if row['filament_id'] in seen: raise ValueError('duplicate filament_id')
        seen.add(row['filament_id'])
        image_id, number = row['filament_id'].rsplit('_', 1)
        if not image_id or not number.isdigit() or int(number) < 1: raise ValueError('invalid filament_id')
        rle = {'size': [2048, 2048], 'counts': row['segmentation_rle'].encode('ascii')}
        area = int(mask_utils.area(rle))
        if not 0 < area <= 2048 * 2048: raise ValueError('invalid or empty RLE')
        preds[image_id].append(rle)
allowed = {Path(r['filename']).stem for r in records}
if len(allowed) != len(records): raise ValueError('held-out filenames must have distinct stems')
if set(preds) - allowed: raise ValueError('predictions include images outside held-out fold')
result = score_matrices(overlap_matrix(r['rles'], preds[Path(r['filename']).stem]) for r in records)
result['fold'] = args.fold
result['observations'] = len(records)
print(json.dumps(result, indent=2))
