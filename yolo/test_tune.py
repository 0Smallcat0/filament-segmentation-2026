"""CPU-only provenance and selection smoke test with synthetic empty masks."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.provenance import sha256_file
from yolo import tune


class TuningProvenanceTest(unittest.TestCase):
    def test_manifest_binding_and_tampered_weight(self):
        tmp = Path(__file__).resolve().parents[2] / 'tmp'
        tmp.mkdir(exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix='yolo-test-', dir=tmp))
        annotation = root / 'train.json'; annotation.write_text('{}', encoding='utf8')
        records = [{'filename': f'image{i}.jpg', 'fold': i, 'rles': []} for i in range(5)]
        for fold in range(5):
            directory = root / f'fold{fold}'; directory.mkdir()
            checkpoint = directory / f'fold{fold}.pt'; checkpoint.write_text(f'weights{fold}', encoding='utf8')
            raw = directory / f'raw_fold{fold}.json'; raw.write_text(json.dumps({f'image{fold}': []}), encoding='utf8')
            env = directory / 'environment.txt'; env.write_text('test==1', encoding='utf8')
            options = {'data': f'/fold{fold}/data.yaml', 'project': f'/fold{fold}/runs', 'amp': False}
            opts = directory / 'resolved_training_options.json'; opts.write_text(json.dumps(options), encoding='utf8')
            train = sorted(r['filename'] for r in records if r['fold'] != fold)
            held = [records[fold]['filename']]
            manifest = {'fold': fold, 'seed': 20261109, 'epochs': 30, 'size': 1024, 'batch': 2,
                        'train_json_sha256': sha256_file(annotation), 'training_filenames': train,
                        'held_out_filenames': held, 'training_image_sha256': {name: name for name in train},
                        'checkpoint_sha256': sha256_file(checkpoint), 'raw_sha256': sha256_file(raw),
                        'environment_sha256': sha256_file(env), 'resolved_training_options_sha256': sha256_file(opts),
                        'resolved_training_options': options, 'pretrained_sha256': 'same', 'source_sha256': {},
                        'python': 'test', 'ultralytics': 'test', 'torch': 'test', 'cuda': 'test',
                        'cudnn': 1, 'gpu': 'test', 'proposal_conf': .01, 'proposal_iou': .7, 'proposal_max_det': 300}
            (directory / f'manifest_fold{fold}.json').write_text(json.dumps(manifest), encoding='utf8')
        output = root / 'tuning.json'
        argv = ['tune', '--train-json', str(annotation), '--runs', str(root), '--output', str(output)]
        with patch.object(sys, 'argv', argv), patch.object(tune, 'prepare_train', return_value=records), contextlib.redirect_stdout(io.StringIO()):
            tune.main()
        result = json.loads(output.read_text(encoding='utf8'))
        self.assertEqual(result['fold_artifacts'][0]['manifest_sha256'], sha256_file(root / 'fold0' / 'manifest_fold0.json'))
        self.assertEqual(result['selection_metric'], 'strict one-to-one pooled PQ at IoU > 0.5')
        (root / 'fold0' / 'fold0.pt').write_text('changed', encoding='utf8')
        with patch.object(sys, 'argv', argv), patch.object(tune, 'prepare_train', return_value=records):
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                tune.main()


if __name__ == '__main__': unittest.main()
