import unittest

from baseline.grid_report import percentile, summarize


class GridReportTests(unittest.TestCase):
    def test_bootstrap_uses_images_and_is_reproducible(self):
        config = {'detector_score': 0.8}
        docs = []
        for fold, count in enumerate((142, 142, 141, 141, 141)):
            observations = []
            for index in range(count):
                quality = (index + fold) % 2
                observations.append({'filename': f'{fold}-{index}', 'score': {'official': {
                    'tp': 1, 'fp': 0, 'fn': 0, 'sum_iou': quality}}})
            docs.append({'fold': fold, 'configs': [config], 'observations': {'0': observations}})
        first = summarize(docs, config, seed=99, samples=1000)
        self.assertEqual(first, summarize(docs, config, seed=99, samples=1000))
        self.assertAlmostEqual(first['pooled_official_pq'],
                               sum((i + f) % 2 for f, n in enumerate((142, 142, 141, 141, 141)) for i in range(n)) / 707)
        lower, upper = first['bootstrap']['official_pq_ci95']
        self.assertLess(lower, first['pooled_official_pq'])
        self.assertGreater(upper, first['pooled_official_pq'])
        self.assertGreater(first['fold_sample_std'], 0)
        docs[1]['observations']['0'][0]['filename'] = '0-0'
        with self.assertRaises(ValueError):
            summarize(docs, config, samples=1)

    def test_percentile_linear_interpolation(self):
        self.assertEqual(percentile([0, 10], 0.025), 0.25)


if __name__ == '__main__':
    unittest.main()
