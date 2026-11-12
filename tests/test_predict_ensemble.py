"""CPU checks for fold identity and deterministic, non-overlapping output."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from predict_ensemble import image_paths, infer_images, load_configuration
from src.provenance import sha256_file


class EnsembleInferenceTest(unittest.TestCase):
    def test_fold_identity_and_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tuning = root / 'tuning.json'
            tuned_hashes = []
            artifacts = []
            for fold in range(5):
                checkpoint = root / f'fold{fold}.pt'
                checkpoint.write_bytes(str(fold).encode())
                tuned_hashes.append(sha256_file(checkpoint))
                manifest = {'schema': 1, 'fold': fold, 'folds': 5, 'seed': 42,
                            'config': {'tta': False, 'size': 1024}, 'source_sha256': 'src',
                            'train_json_sha256': 'labels', 'probability_format': 'uint16 png, divide by 65535',
                            'checkpoint_sha256': sha256_file(checkpoint)}
                (root / f'manifest_fold{fold}.json').write_text(json.dumps(manifest), encoding='utf8')
                artifacts.append({'fold': fold, 'checkpoint': checkpoint.name, 'checkpoint_sha256': tuned_hashes[-1],
                                  'manifest': f'manifest_fold{fold}.json', 'manifest_sha256': sha256_file(root / f'manifest_fold{fold}.json')})
            tuned = {'seed': 42, 'checkpoint_sha256': tuned_hashes,
                     'fold_artifacts': artifacts,
                     'final_inference_setting': {'threshold': 0.5, 'min_area': 2, 'close_radius': 0}}
            tuning.write_text(json.dumps(tuned), encoding='utf8')
            settings = load_configuration(root, tuning)
            self.assertEqual((settings['tta'], settings['threshold'], settings['seed']), (False, 0.5, 42))
            tuned['checkpoint_sha256'][2] = 'a' * 64
            tuning.write_text(json.dumps(tuned), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'tuning manifest hash mismatch'):
                load_configuration(root, tuning)
            tuned['checkpoint_sha256'][2] = sha256_file(root / 'fold2.pt')
            tuning.write_text(json.dumps(tuned), encoding='utf8')
            manifest_path = root / 'manifest_fold1.json'
            original = manifest_path.read_text(encoding='utf8')
            modified = json.loads(original)
            modified['config']['tta'] = True
            manifest_path.write_text(json.dumps(modified), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'tuning manifest hash mismatch'):
                load_configuration(root, tuning)
            manifest_path.write_text(original, encoding='utf8')
            (root / 'fold4.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint hash mismatch'):
                load_configuration(root, tuning)

    def test_stable_disjoint_masks_and_empty_image(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            images = root / 'test_images'
            images.mkdir()
            for name in ('z.jpeg', 'a.jpeg'): self.assertTrue(cv2.imwrite(str(images / name), np.zeros((2048, 2048), np.uint8)))
            inventory = root / 'inventory.json'
            inventory.write_text(json.dumps({'files': ['z.jpeg', 'a.jpeg']}), encoding='utf8')
            paths, hashes = image_paths(images, inventory, expected_count=2)
            self.assertEqual([p.name for p in paths], ['a.jpeg', 'z.jpeg'])
            def maps(path, empty=False):
                p = np.zeros((2048, 2048), dtype=np.float32)
                if path.stem == 'a' or not empty:
                    p[0:2, 0:2] = 0.9
                    p[5:7, 5:7] = 0.9
                return [p] * 5
            settings = {'threshold': 0.5, 'min_area': 1, 'close_radius': 0}
            output, report_path = root / 'submission.csv', root / 'report.json'
            with self.assertRaisesRegex(ValueError, 'empty predictions=.*z.jpeg'):
                infer_images(paths, hashes, lambda path: maps(path, empty=True), settings, output, report_path, images, {})
            self.assertFalse(output.exists())
            self.assertFalse(report_path.exists())
            def broken(path):
                if path.stem == 'z': raise RuntimeError('inference failure')
                return maps(path)
            with self.assertRaisesRegex(RuntimeError, 'inference failure'):
                infer_images(paths, hashes, broken, settings, output, report_path, images, {})
            self.assertFalse(output.exists())
            self.assertFalse(report_path.exists())
            report = infer_images(paths, hashes, maps, settings, output, report_path, images, {})
            first = output.read_bytes()
            with self.assertRaises(FileExistsError): infer_images(paths, hashes, maps, settings, output, report_path, images, {})
            second = root / 'second.csv'
            infer_images(paths, hashes, maps, settings, second, root / 'second.json', images, {})
            self.assertEqual(first, second.read_bytes())
            self.assertEqual([(x['stem'], x['instances']) for x in report['images']], [('a', 2), ('z', 2)])
            with output.open(newline='', encoding='utf8') as stream: rows = list(csv.DictReader(stream))
            self.assertEqual([r['filament_id'] for r in rows], ['a_1', 'a_2', 'z_1', 'z_2'])
            masks = [mask_utils.decode({'size': [2048, 2048], 'counts': r['segmentation_rle'].encode('ascii')}) for r in rows[:2]]
            self.assertEqual(int((masks[0] & masks[1]).sum()), 0)
            with self.assertRaises(ValueError): image_paths(images, inventory, expected_count=180)
            inventory.write_text(json.dumps({'files': ['wrong.jpeg', 'a.jpeg']}), encoding='utf8')
            with self.assertRaisesRegex(ValueError, 'inventory'): image_paths(images, inventory, expected_count=2)
            inventory.write_text(json.dumps({'files': ['z.jpeg', 'a.jpeg']}), encoding='utf8')
            cv2.imwrite(str(images / 'z.jpeg'), np.zeros((8, 8), np.uint8))
            with self.assertRaisesRegex(ValueError, 'dimensions'): image_paths(images, inventory, expected_count=2)


if __name__ == '__main__':
    unittest.main()
