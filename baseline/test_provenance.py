"""CPU-only checks for provenance identity and fold invariance."""
import tempfile
import unittest
from pathlib import Path

from baseline.provenance import dataset_manifest, file_sha256, object_sha256


class ProvenanceTest(unittest.TestCase):
    def test_dataset_hash_tracks_bytes_assignment_and_not_input_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'labels').write_bytes(b'labels')
            (root / 'a.jpg').write_bytes(b'a')
            (root / 'b.jpg').write_bytes(b'b')
            rows = [{'filename': 'a.jpg', 'image_id': 'aa', 'fold': 0},
                    {'filename': 'b.jpg', 'image_id': 'bb', 'fold': 1}]
            digest = dataset_manifest(root / 'labels', root, rows)['sha256']
            self.assertEqual(digest, dataset_manifest(root / 'labels', root, rows[::-1])['sha256'])
            rows[0]['fold'] = 1
            self.assertNotEqual(digest, dataset_manifest(root / 'labels', root, rows)['sha256'])
            rows[0]['fold'] = 0
            (root / 'a.jpg').write_bytes(b'x')
            self.assertNotEqual(digest, dataset_manifest(root / 'labels', root, rows)['sha256'])
            (root / 'labels').unlink()
            with self.assertRaises(FileNotFoundError):
                file_sha256(root / 'labels')
            self.assertEqual(object_sha256({'a': 1, 'b': 2}), object_sha256({'b': 2, 'a': 1}))


if __name__ == '__main__':
    unittest.main()
