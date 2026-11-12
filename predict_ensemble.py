"""Run five trained folds on official test images; no training or label reads."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

from src.postprocess import assert_disjoint, instances
from src.provenance import environment, sha256_file

OFFICIAL_INVENTORY_SHA256 = 'd436055fc93cc13676f12964ee6553038f026c0a0004267f96fa58cb19bc21ff'


def load_configuration(weights_dir, tuning_path):
    """Validate five checkpoints against their training manifests and OOF settings."""
    weights_dir = Path(weights_dir)
    tuning = json.loads(Path(tuning_path).read_text(encoding='utf8'))
    setting = tuning['final_inference_setting']
    tuned_hashes = tuning.get('checkpoint_sha256')
    if not isinstance(tuned_hashes, list) or len(tuned_hashes) != 5 or not all(isinstance(x, str) and len(x) == 64 for x in tuned_hashes):
        raise ValueError('tuning output lacks five checkpoint hashes')
    artifacts = tuning.get('fold_artifacts')
    if not isinstance(artifacts, list) or len(artifacts) != 5:
        raise ValueError('tuning output lacks five fold artifacts')
    threshold, min_area, radius = setting['threshold'], setting['min_area'], setting['close_radius']
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not 0 <= threshold <= 1:
        raise ValueError('invalid threshold')
    if any(type(v) is not int or v < 0 for v in (min_area, radius)):
        raise ValueError('invalid minimum area or closing radius')
    manifests, shared = [], None
    for fold in range(5):
        path = weights_dir / f'manifest_fold{fold}.json'
        artifact = artifacts[fold]
        if not isinstance(artifact, dict) or artifact.get('fold') != fold or artifact.get('manifest') != path.name or artifact.get('checkpoint') != f'fold{fold}.pt' or artifact.get('checkpoint_sha256') != tuned_hashes[fold] or artifact.get('manifest_sha256') != sha256_file(path):
            raise ValueError(f'tuning manifest hash mismatch: {path}')
        manifest = json.loads(path.read_text(encoding='utf8'))
        if manifest.get('schema') != 1 or manifest.get('fold') != fold or manifest.get('folds') != 5:
            raise ValueError(f'invalid fold manifest: {path}')
        config = manifest.get('config', {})
        if type(config.get('tta')) is not bool or type(config.get('size')) is not int or config['size'] <= 0:
            raise ValueError(f'invalid inference settings: {path}')
        signature = {key: manifest.get(key) for key in ('config', 'seed', 'source_sha256', 'train_json_sha256', 'probability_format')}
        if any(value is None for value in signature.values()):
            raise ValueError(f'incomplete fold manifest: {path}')
        if shared is None: shared = signature
        elif shared != signature: raise ValueError(f'inconsistent fold manifest: {path}')
        checkpoint = weights_dir / f'fold{fold}.pt'
        if sha256_file(checkpoint) != manifest.get('checkpoint_sha256'):
            raise ValueError(f'checkpoint hash mismatch: {checkpoint}')
        if manifest['checkpoint_sha256'] != tuned_hashes[fold]:
            raise ValueError(f'tuning checkpoint hash mismatch: {checkpoint}')
        manifests.append(manifest)
    if tuning.get('seed') != shared['seed']:
        raise ValueError('tuning seed does not match checkpoints')
    return {'threshold': float(threshold), 'min_area': min_area, 'close_radius': radius,
            'tta': shared['config']['tta'], 'size': shared['config']['size'],
            'seed': shared['seed'], 'checkpoint_sha256': [m['checkpoint_sha256'] for m in manifests],
            'tuning_sha256': sha256_file(tuning_path),
            'source_sha256': shared['source_sha256'], 'train_json_sha256': shared['train_json_sha256']}


def image_paths(image_dir, inventory_path, expected_count=180, expected_shape=(2048, 2048)):
    """Require the official inventory, readable native dimensions, and content hashes."""
    paths = sorted((p for p in Path(image_dir).iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')),
                   key=lambda p: p.name)
    if len(paths) != expected_count or len({p.stem for p in paths}) != len(paths):
        raise ValueError(f'expected {expected_count} distinct image stems, found {len(paths)} images')
    inventory = json.loads(Path(inventory_path).read_text(encoding='utf8'))
    names = inventory.get('files')
    if not isinstance(names, list) or len(names) != expected_count or any(not isinstance(n, str) or Path(n).name != n for n in names) or len(set(names)) != expected_count or set(names) != {p.name for p in paths}:
        raise ValueError('test images do not match official inventory')
    hashes = {}
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != expected_shape:
            raise ValueError(f'invalid test image dimensions: {path}')
        hashes[path.name] = sha256_file(path)
    return paths, hashes


def infer_images(images, hashes, predict_models, settings, csv_path, report_path, image_dir, metadata):
    """Write temporary output, run the submission checker, then publish both files."""
    csv_path, report_path = Path(csv_path).resolve(), Path(report_path).resolve()
    if csv_path == report_path or csv_path.parent != report_path.parent:
        raise ValueError('output and report must be distinct files in one directory')
    if csv_path.exists() or report_path.exists():
        raise FileExistsError('choose unused output and report paths to avoid stale artifacts')
    started = time.monotonic()
    processed = []
    with tempfile.TemporaryDirectory(prefix='.filament-', dir=csv_path.parent) as temp:
        tmp_csv, tmp_report = Path(temp) / 'submission.csv', Path(temp) / 'report.json'
        with tmp_csv.open('w', newline='', encoding='utf8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['filament_id', 'segmentation_rle'])
            for image in images:
                if sha256_file(image) != hashes[image.name]: raise ValueError(f'image changed: {image}')
                probabilities = [np.asarray(p, dtype=np.float32) for p in predict_models(image)]
                if len(probabilities) != 5 or any(p.shape != (2048, 2048) or not np.isfinite(p).all() for p in probabilities):
                    raise ValueError(f'invalid five-fold predictions for {image}')
                mean = np.mean(np.stack(probabilities), axis=0, dtype=np.float64)
                quantized = np.rint(np.clip(mean, 0, 1) * 65535).astype(np.uint16)
                prob = quantized.astype(np.float32) / 65535
                rles = instances(prob, threshold=settings['threshold'], min_area=settings['min_area'], close_radius=settings['close_radius'])
                assert_disjoint(rles)
                for index, rle in enumerate(rles, 1):
                    writer.writerow([f'{image.stem}_{index}', rle['counts'].decode('ascii')])
                processed.append({'file_name': image.name, 'stem': image.stem, 'sha256': hashes[image.name], 'instances': len(rles)})
        checker = Path(__file__).resolve().parent / 'check_submission.py'
        result = subprocess.run([sys.executable, str(checker), str(tmp_csv), '--images', str(Path(image_dir).resolve())], capture_output=True, text=True)
        if result.returncode:
            empty = [item['file_name'] for item in processed if not item['instances']]
            raise ValueError(f'submission checker failed; empty predictions={empty}: {result.stderr.strip()}')
        if any(sha256_file(image) != hashes[image.name] for image in images): raise ValueError('test images changed during inference')
        report = {'images_processed': len(processed), 'images': processed, 'settings': settings, **metadata,
                  'elapsed_seconds': time.monotonic() - started, 'checker': result.stdout.strip(),
                  'submission_sha256': sha256_file(tmp_csv),
                  'note': 'Five-fold ensemble averaging is not the same prediction distribution as single-fold OOF tuning; no ensemble OOF score or reproduction is claimed.'}
        tmp_report.write_text(json.dumps(report, indent=2), encoding='utf8')
        try:
            os.replace(tmp_report, report_path)
            os.replace(tmp_csv, csv_path)
        except Exception:
            report_path.unlink(missing_ok=True)
            raise
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--tuning', required=True)
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--report', required=True)
    args = parser.parse_args()
    if sha256_file(args.inventory) != OFFICIAL_INVENTORY_SHA256:
        raise ValueError('official test inventory hash mismatch')
    settings = load_configuration(args.weights, args.tuning)
    images, hashes = image_paths(args.images, args.inventory)
    # Deferred imports allow CPU tests of deterministic file output without torch.
    import torch
    from src.inference import load_model, predict
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = []
    for fold in range(5):
        path = Path(args.weights) / f'fold{fold}.pt'
        meta = torch.load(path, map_location='cpu', weights_only=True)
        if meta.get('fold') != fold or meta.get('size') != settings['size'] or meta.get('seed') != settings['seed']:
            raise ValueError(f'checkpoint metadata mismatch: {path}')
        models.append(load_model(path, device))
    def predict_models(path):
        return [predict(model, path, device, size=settings['size'], tta=settings['tta']) for model in models]
    repo = Path(__file__).resolve().parent
    sources = ['predict_ensemble.py', 'notebooks/pipeline.ipynb', 'src/inference.py', 'src/model.py', 'src/postprocess.py', 'src/provenance.py', 'check_submission.py']
    runtime = environment()
    runtime.update({'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                    'device': device, 'gpu_name': torch.cuda.get_device_name(0) if device == 'cuda' else None})
    metadata = {'inventory_sha256': sha256_file(args.inventory), 'runtime': runtime,
                'inference_source_sha256': {name: sha256_file(repo / name) for name in sources}}
    report = infer_images(images, hashes, predict_models, settings, args.output, args.report, args.images, metadata)
    print(f"Processed {report['images_processed']} images, wrote {args.output}; SHA-256 {report['submission_sha256']}")


if __name__ == '__main__':
    main()
