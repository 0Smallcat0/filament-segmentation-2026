"""Validate CSV IDs, image coverage, COCO RLEs and pairwise non-overlap."""
import argparse, csv
from collections import defaultdict
from pathlib import Path
from pycocotools import mask as mask_utils
from src.postprocess import assert_disjoint

parser = argparse.ArgumentParser()
parser.add_argument('csv'); parser.add_argument('--images', help='test image directory, for exact coverage check')
args = parser.parse_args()
groups = defaultdict(list); identifiers = set()
with open(args.csv, newline='', encoding='utf8') as f:
    reader = csv.DictReader(f)
    if reader.fieldnames != ['filament_id', 'segmentation_rle']: raise ValueError('invalid CSV columns')
    for row in reader:
        identifier = row['filament_id']; stem, number = identifier.rsplit('_', 1)
        if not stem or not number.isdigit() or int(number) < 1 or identifier in identifiers: raise ValueError('invalid or duplicate ID')
        identifiers.add(identifier)
        counts = row['segmentation_rle']
        if not counts: raise ValueError('empty RLE')
        rle = {'size': [2048, 2048], 'counts': counts.encode('ascii')}
        if int(mask_utils.area(rle)) == 0: raise ValueError('empty mask')
        groups[stem].append(rle)
if not identifiers: raise ValueError('empty submission')
for rles in groups.values(): assert_disjoint(rles)
if args.images:
    images = {p.stem for p in Path(args.images).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')}
    if set(groups) != images: raise ValueError(f'coverage differs: missing={len(images-set(groups))}, extra={len(set(groups)-images)}')
print(f'OK: {len(groups)} images, {len(identifiers)} masks, no pairwise overlaps')
