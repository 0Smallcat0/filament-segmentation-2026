"""YOLO polygon conversion and deterministic non-overlapping predictions."""
import cv2
import numpy as np
from pycocotools import mask as mask_utils


def polygon_from_rle(rle):
    """Approximate one instance with its largest external contour for YOLO."""
    mask = mask_utils.decode(rle).astype(np.uint8)
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 3:
        return None
    points = contour.astype(np.float64)
    points[:, 0] /= width
    points[:, 1] /= height
    return '0 ' + ' '.join(f'{v:.6f}' for v in points.ravel())


def disjoint(raw, confidence, min_area, close_radius):
    """Assign contested pixels to highest confidence; ties retain model order."""
    chosen = sorted(enumerate(raw), key=lambda x: (-x[1]['score'], x[0]))
    occupied = None
    output = []
    for _, item in chosen:
        if item['score'] < confidence:
            continue
        rle = {'size': item['size'], 'counts': item['counts'].encode('ascii')}
        mask = mask_utils.decode(rle).astype(np.uint8)
        if occupied is None:
            occupied = np.zeros(mask.shape, dtype=bool)
        if mask.shape != occupied.shape:
            raise ValueError('inconsistent mask dimensions')
        if close_radius:
            width = close_radius * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        clean = mask.astype(bool) & ~occupied
        if int(clean.sum()) < min_area:
            continue
        occupied |= clean
        output.append(mask_utils.encode(np.asfortranarray(clean.astype(np.uint8))))
    return output
