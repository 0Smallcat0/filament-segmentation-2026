import unittest

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from yolo.common import disjoint, polygon_from_rle


def encode(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))


def raw(mask, score):
    rle = encode(mask)
    return {'size': rle['size'], 'counts': rle['counts'].decode('ascii'), 'score': score}


class SegmentationTest(unittest.TestCase):
    def test_polygon_coordinates_and_degenerate(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[3:10, 5:15] = 1
        fields = polygon_from_rle(encode(mask)).split()
        self.assertEqual(fields[0], '0')
        self.assertGreaterEqual(len(fields), 7)
        self.assertTrue(all(0 <= float(x) <= 1 for x in fields[1:]))
        self.assertIsNone(polygon_from_rle(encode(np.zeros_like(mask))))

    def test_score_priority_disjoint_and_area(self):
        a = np.zeros((10, 10), dtype=np.uint8); a[1:7, 1:7] = 1
        b = np.zeros_like(a); b[4:9, 4:9] = 1
        output = disjoint([raw(a, .2), raw(b, .9)], confidence=.1, min_area=1, close_radius=0)
        self.assertEqual(len(output), 2)
        first, second = (mask_utils.decode(x).astype(bool) for x in output)
        self.assertTrue(np.array_equal(first, b.astype(bool)))
        self.assertFalse(np.any(first & second))
        self.assertTrue(np.array_equal(first | second, (a | b).astype(bool)))
        self.assertEqual(len(disjoint([raw(a, .2)], confidence=.3, min_area=1, close_radius=0)), 0)
        self.assertEqual(len(disjoint([raw(a, .2)], confidence=.1, min_area=37, close_radius=0)), 0)

    def test_largest_external_polygon_loses_holes_and_other_components(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[4:22, 4:22] = 1
        mask[10:15, 10:15] = 0  # Hole inside the largest component.
        mask[30:35, 30:35] = 1  # A disconnected piece of the same instance.
        fields = polygon_from_rle(encode(mask)).split()[1:]
        points = np.asarray(fields, dtype=float).reshape(-1, 2)
        points[:, 0] *= mask.shape[1]; points[:, 1] *= mask.shape[0]
        roundtrip = np.zeros_like(mask)
        cv2.fillPoly(roundtrip, [np.rint(points).astype(np.int32)], 1)
        self.assertEqual(mask[12, 12], 0)
        self.assertEqual(roundtrip[12, 12], 1)
        self.assertEqual(mask[32, 32], 1)
        self.assertEqual(roundtrip[32, 32], 0)


if __name__ == '__main__': unittest.main()
