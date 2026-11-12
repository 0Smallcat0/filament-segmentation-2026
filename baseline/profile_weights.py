"""Verify a full-train result and record its losslessly split checkpoint hashes."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from baseline.provenance import object_sha256


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    manifest = json.loads((args.source / 'provenance_fold0.json').read_text(encoding='utf8'))
    if (manifest['status'] != 'complete' or manifest['settings'].get('training_scope')
            != 'all_competition_train_observations' or manifest['held_out_filenames']):
        raise ValueError('Not a completed full-train model')
    observations = manifest['dataset']['observations']
    names = [r['filename'] for r in observations]
    if (manifest['source_dataset'] != 'Kaggle competition filament-segmentation-2026 train'
            or manifest['dataset']['annotation_sha256'] != '5da9e92b5a1a1947fd5d57adb6688269625c48ec1ef884daf2a01618c9ed54a1'
            or len(names) != 707 or len(set(names)) != 707
            or len(manifest['train_filenames']) != 707 or set(manifest['train_filenames']) != set(names)):
        raise ValueError('Unexpected training dataset or observation list')
    dataset = manifest['dataset']
    if object_sha256({k: v for k, v in dataset.items() if k != 'sha256'}) != dataset['sha256']:
        raise ValueError('Dataset manifest hash mismatch')
    if object_sha256(manifest['source_code']['files']) != manifest['source_code']['sha256']:
        raise ValueError('Source manifest hash mismatch')
    for field in ('settings', 'environment'):
        if object_sha256(manifest[field]) != manifest[field + '_sha256']:
            raise ValueError(f'{field} hash mismatch')
    detector, refiner = args.source / 'final_detector.pt', args.source / 'final_refiner.pt'
    train_manifest = args.source / 'training_manifest.csv'
    if any(sha(path) != manifest['outputs_sha256'][path.name] for path in (detector, refiner, train_manifest)):
        raise ValueError('Full-train checkpoint hash mismatch')
    with train_manifest.open(newline='', encoding='utf8') as stream:
        rows = list(csv.DictReader(stream))
    expected = {(r['medoid_image_id'], r['filename'], Path(r['filename']).stem) for r in observations}
    actual = {(r['id'], r['file_name'], r['observation_id']) for r in rows}
    if len(rows) != 707 or len(actual) != 707 or actual != expected:
        raise ValueError('Training manifest does not match medoid observations')
    subprocess.run([sys.executable, str(Path(__file__).with_name('split_weights.py')),
                    str(detector), str(args.destination)], check=True)
    shutil.copyfile(refiner, args.destination / 'final_refiner.pt')
    state = {}
    for index in range(2):
        state.update(torch.load(args.destination / f'detector-part{index}.pt', map_location='cpu', weights_only=True))
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        for field in (name, str(tensor.dtype), str(tuple(tensor.shape))):
            digest.update(field.encode() + bytes([0]))
        digest.update(tensor.contiguous().numpy().tobytes())
    profile = {'original_detector_sha256': sha(detector), 'detector_state_sha256': digest.hexdigest(),
               'refiner_sha256': sha(refiner), 'provenance_sha256': sha(args.source / 'provenance_fold0.json')}
    (args.destination / 'checkpoint-profile.json').write_text(json.dumps(profile, indent=2), encoding='utf8')
    print(json.dumps(profile, indent=2))


if __name__ == '__main__':
    main()
