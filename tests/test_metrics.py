import unittest
import numpy as np
from src.metrics import score_matrices

class MetricTests(unittest.TestCase):
    def test_official_duplicates_and_strict(self):
        result = score_matrices([np.array([[.9, .8]])])
        self.assertEqual(result['official']['tp'], 2)
        self.assertAlmostEqual(result['official']['pq'], .85)
        self.assertEqual(result['one_to_one']['tp'], 1)
        self.assertEqual(result['one_to_one']['fp'], 1)
        self.assertAlmostEqual(result['one_to_one']['pq'], .6)
        self.assertEqual(result['one_gt_to_many_predictions_iou_gt_0'], 1)
    def test_threshold_and_empty(self):
        result = score_matrices([np.array([[.5]]), np.zeros((2,0)), np.zeros((0,1))])
        self.assertEqual((result['one_to_one']['tp'], result['one_to_one']['fp'], result['one_to_one']['fn']), (0,2,3))
    def test_maximum_total_iou(self):
        result = score_matrices([np.array([[1., .55, .3], [.55, 1., .5454545], [.55, .1, 0.]])])
        self.assertEqual(result['one_to_one']['tp'], 2)
        self.assertAlmostEqual(result['one_to_one']['pq'], 2 / 3)

if __name__ == '__main__': unittest.main()
