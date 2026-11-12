import unittest

import numpy as np
from pycocotools import mask as mask_utils

from baseline.ensemble_r3 import cluster_masks, consensus, encode
from src.postprocess import assert_disjoint


class EnsembleTests(unittest.TestCase):
    def test_distinct_folds_required(self):
        square = np.zeros((24, 24), dtype=np.uint8)
        square[2:12, 2:12] = 1
        rle = encode(square)
        self.assertEqual(len(cluster_masks([[rle, rle], [rle], [], [], []])), 2)
        self.assertEqual(len(consensus([[rle], [rle], [], [], []], 2, 1)[0]), 1)
        self.assertEqual(consensus([[rle], [rle], [], [], []], 3, 1), ([], False))

    def test_overlap_is_assigned_once(self):
        a = np.zeros((24, 24), dtype=np.uint8)
        b = np.zeros_like(a)
        a[2:12, 2:12] = 1
        b[8:18, 8:18] = 1
        output, ok = consensus([[encode(a), encode(b)], [encode(a), encode(b)]], 2, 1)
        self.assertTrue(ok)
        self.assertEqual(len(output), 2)
        assert_disjoint(output)
        self.assertEqual(sum(int(mask_utils.area(x)) for x in output), int((a | b).sum()))


if __name__ == '__main__':
    unittest.main()
