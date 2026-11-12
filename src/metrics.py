"""Image-level overlap and dataset-level PQ, including the official all-hit variant."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from pycocotools import mask as mask_utils


def overlap_matrix(gt, pred):
    """Pairwise IoU of COCO RLEs; does not decode a stack of full-size masks."""
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=float)
    return np.asarray(mask_utils.iou(gt, pred, [0] * len(pred)), dtype=float)


def score_matrices(matrices, threshold=0.5):
    """Aggregate official and one-to-one PQ. Diagnostic degrees use IoU > 0.

    Maximum total qualifying IoU maximizes PQ because its denominator is fixed
    at (n_gt + n_pred) / 2 under one-to-one matching.
    Official counts each qualifying pair and only counts unmatched rows/columns
    as FN/FP, matching the organizer self-evaluation notebook.
    """
    strict_sum = official_sum = 0.0
    strict_tp = strict_fp = strict_fn = official_tp = official_fp = official_fn = 0
    one_to_many = many_to_one = 0
    matched_iou = []
    for matrix in matrices:
        a = np.asarray(matrix, dtype=float)
        if a.ndim != 2 or not np.isfinite(a).all() or ((a < 0) | (a > 1)).any():
            raise ValueError('Expected finite two-dimensional IoU matrix in [0,1]')
        ng, npred = a.shape
        hit = a > threshold
        one_to_many += int(((a > 0).sum(axis=1) > 1).sum())
        many_to_one += int(((a > 0).sum(axis=0) > 1).sum())
        official_tp += int(hit.sum())
        official_sum += float(a[hit].sum())
        official_fn += int((~hit.any(axis=1)).sum())
        official_fp += int((~hit.any(axis=0)).sum())
        if ng and npred:
            # Maximize the sum of qualifying IoUs, not just match cardinality.
            cost = np.where(hit, -a, 0.0)
            rows, cols = linear_sum_assignment(cost)
            matched = a[rows, cols][hit[rows, cols]]
        else:
            matched = np.empty(0)
        n = len(matched)
        strict_tp += n
        strict_fn += ng - n
        strict_fp += npred - n
        strict_sum += float(matched.sum())
        matched_iou.extend(matched.tolist())
    def pack(total, tp, fp, fn):
        denominator = tp + 0.5 * (fp + fn)
        return {'pq': total / denominator if denominator else 0.0,
                'tp': tp, 'fp': fp, 'fn': fn, 'sum_iou': total}
    return {'one_to_one': pack(strict_sum, strict_tp, strict_fp, strict_fn),
            'official': pack(official_sum, official_tp, official_fp, official_fn),
            'one_gt_to_many_predictions_iou_gt_0': one_to_many,
            'many_gt_to_one_prediction_iou_gt_0': many_to_one,
            'matched_iou': matched_iou}
