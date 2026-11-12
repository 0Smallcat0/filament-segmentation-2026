"""Exercise grouped held-out map loading and the OOF tuning CLI."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from src.data import prepare_train
from src.provenance import sha256_file
from tune_oof import cross_fit_settings


class OofTuningTest(unittest.TestCase):
    def test_cross_fit_excludes_scored_fold_and_aggregates_counts(self):
        settings = [(0.3, 1, 0), (0.6, 1, 0)]
        first = [np.eye(2)] + [np.zeros((1, 1)) for _ in range(4)]
        second = [np.zeros((2, 2))] + [np.ones((1, 1)) for _ in range(4)]
        result = cross_fit_settings(settings, [first, second], list(range(5)))
        self.assertEqual(result['folds'][0]['threshold'], 0.6)
        self.assertEqual(result['folds'][0]['selection_folds'], [1, 2, 3, 4])
        self.assertAlmostEqual(result['score']['one_to_one']['pq'], 4 / 6)
        self.assertEqual(result['score']['one_to_one']['tp'], 4)
        self.assertEqual(result['score']['one_to_one']['fp'], 2)
        second[0] = np.eye(2)
        changed = cross_fit_settings(settings, [first, second], list(range(5)))
        self.assertEqual(changed['folds'][0]['threshold'], result['folds'][0]['threshold'])
        self.assertEqual(changed['folds'][0]['selection_score'], result['folds'][0]['selection_score'])
        self.assertAlmostEqual(changed['folds'][0]['held_out_score']['one_to_one']['pq'], 1)

    def test_all_folds_and_missing_map(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            coco = {'images': [], 'annotations': []}
            for i in range(5):
                coco['images'].append({'id': i, 'file_name': f'image{i}.jpeg', 'height': 8, 'width': 8})
                coco['annotations'].append({'id': i, 'image_id': i, 'segmentation': [[1, 1, 6, 1, 6, 6, 1, 6]]})
            labels = root / 'labels.json'
            labels.write_text(json.dumps(coco), encoding='utf8')
            records = prepare_train(labels)
            for record in records:
                directory = root / f"probabilities_fold{record['fold']}"
                directory.mkdir(exist_ok=True)
                probability = np.zeros((8, 8), dtype=np.uint16)
                probability[1:6, 1:6] = 65535
                self.assertTrue(cv2.imwrite(str(directory / (Path(record['filename']).stem + '.png')), probability))
            for fold in range(5):
                val = [r for r in records if r['fold'] == fold]
                checkpoint = root / f'fold{fold}.pt'
                checkpoint.write_bytes(f'checkpoint{fold}'.encode())
                hashes = {Path(r['filename']).stem: sha256_file(root / f'probabilities_fold{fold}' / (Path(r['filename']).stem + '.png')) for r in val}
                manifest = {'schema': 1, 'fold': fold, 'folds': 5, 'seed': 20261109,
                    'config': {'size': 8}, 'source_sha256': 'source', 'train_json_sha256': sha256_file(labels),
                    'validation_filenames': sorted(r['filename'] for r in val),
                    'checkpoint_sha256': sha256_file(checkpoint), 'probability_sha256': hashes,
                    'probability_format': 'uint16 png, divide by 65535', 'environment': {}}
                (root / f'manifest_fold{fold}.json').write_text(json.dumps(manifest), encoding='utf8')
            out = root / 'scores.json'
            cmd = [sys.executable, 'tune_oof.py', '--train-json', str(labels), '--runs', str(root),
                   '--output', str(out), '--thresholds', '0.5', '--min-areas', '1', '--close-radii', '0']
            subprocess.run(cmd, check=True, capture_output=True, text=True, env=os.environ.copy())
            result = json.loads(out.read_text(encoding='utf8'))
            self.assertEqual(result['observations'], 5)
            self.assertAlmostEqual(result['results'][0]['score']['one_to_one']['pq'], 1.0)
            self.assertAlmostEqual(result['cross_fitted']['score']['one_to_one']['pq'], 1.0)
            self.assertIn('optimistically biased', result['selection_bias'])
            self.assertEqual(len(result['fold_artifacts']), 5)
            self.assertEqual(result['checkpoint_sha256'], [sha256_file(root / f'fold{fold}.pt') for fold in range(5)])
            for artifact in result['fold_artifacts']:
                fold = artifact['fold']
                self.assertEqual(artifact['checkpoint'], f'fold{fold}.pt')
                self.assertEqual(artifact['manifest'], f'manifest_fold{fold}.json')
                self.assertEqual(artifact['checkpoint_sha256'], sha256_file(root / artifact['checkpoint']))
                self.assertEqual(artifact['manifest_sha256'], sha256_file(root / artifact['manifest']))
            checkpoint = root / 'fold0.pt'
            original = checkpoint.read_bytes()
            checkpoint.write_bytes(b'changed checkpoint')
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            checkpoint.write_bytes(original)
            manifest_path = root / 'manifest_fold0.json'
            manifest = json.loads(manifest_path.read_text(encoding='utf8'))
            manifest['source_sha256'] = 'different source'
            manifest_path.write_text(json.dumps(manifest), encoding='utf8')
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            manifest['source_sha256'] = 'source'
            manifest_path.write_text(json.dumps(manifest), encoding='utf8')
            next(root.glob('probabilities_fold*/*.png')).unlink()
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)


if __name__ == '__main__':
    unittest.main()
