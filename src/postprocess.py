"""Convert semantic probabilities into disjoint connected components."""
import cv2
import numpy as np
from pycocotools import mask as mask_utils


def instances(probability, threshold=0.5, min_area=100, close_radius=0):
    """Threshold, optionally close gaps, and return disjoint components as RLEs."""
    mask = (probability > threshold).astype(np.uint8)
    if close_radius:
        size = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = []
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            result.append(mask_utils.encode(np.asfortranarray((labels == label).astype(np.uint8))))
    return result


def assert_disjoint(rles):
    """Raise if any two instance masks share at least one pixel."""
    if len(rles) < 2: return
    iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)))
    if np.any(np.triu(iou, 1) > 0): raise ValueError('overlapping instances')
