"""Load only competition train COCO labels and group records by observation."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
from pycocotools import mask as mask_utils


def annotation_rle(annotation, height, width):
    """Convert a COCO polygon or RLE annotation into compressed COCO RLE."""
    seg = annotation['segmentation']
    if isinstance(seg, dict):
        result = mask_utils.frPyObjects(seg, height, width) if isinstance(seg['counts'], list) else dict(seg)
        if isinstance(result['counts'], str): result['counts'] = result['counts'].encode('ascii')
        return result
    if not seg:
        return mask_utils.encode(np.asfortranarray(np.zeros((height, width), dtype=np.uint8)))
    return mask_utils.merge(mask_utils.frPyObjects(seg, height, width))


def prepare_train(path, folds=5, seed=20261109):
    """Select semantic-IoU medoid annotator and split physical filenames into folds.

    Ties are resolved by sorted record id; the selected record's instances remain
    distinct. Other annotators of an observation never enter another fold.
    """
    if folds < 2: raise ValueError('folds must be at least 2')
    with Path(path).open(encoding='utf8') as f: coco = json.load(f)
    anns = defaultdict(list)
    for ann in coco['annotations']:
        if not ann.get('iscrowd', 0): anns[str(ann['image_id'])].append(ann)
    groups = defaultdict(list)
    for im in coco['images']: groups[im['file_name']].append(im)
    names = sorted(groups)
    if len(names) < folds: raise ValueError('fewer observations than folds')
    order = np.random.default_rng(seed).permutation(len(names))
    fold_by_name = {names[int(index)]: int(pos % folds) for pos, index in enumerate(order)}
    records = []
    for name in names:
        images = sorted(groups[name], key=lambda x: str(x['id']))
        unions = []
        all_rles = []
        for im in images:
            h, w = int(im['height']), int(im['width'])
            rles = [annotation_rle(a, h, w) for a in anns[str(im['id'])]]
            rles = [rle for rle in rles if mask_utils.area(rle) > 0]
            all_rles.append(rles)
            unions.append(mask_utils.merge(rles) if rles else mask_utils.encode(np.asfortranarray(np.zeros((h,w), dtype=np.uint8))))
        if len(images) > 1:
            pair = np.asarray(mask_utils.iou(unions, unions, [0]*len(unions)))
            # Empty/empty annotations agree perfectly, unlike pycocotools' 0/0.
            empty = np.array([not r or sum(mask_utils.area(x) for x in r) == 0 for r in all_rles])
            pair[np.outer(empty, empty)] = 1.0
            agreement = (pair.sum(axis=1) - np.diag(pair)) / (len(images)-1)
            chosen = int(np.argmax(agreement))
        else: chosen = 0
        records.append({'filename': name, 'image_id': str(images[chosen]['id']),
                        'annotators': len(images), 'fold': fold_by_name[name], 'rles': all_rles[chosen]})
    return records
