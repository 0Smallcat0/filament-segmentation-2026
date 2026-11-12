"""Build a private held-out-only kernel for empty-prediction fallback tuning."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--low-proposals', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = ['baseline/__init__.py', 'baseline/handoff.py', 'baseline/fallback_oof.py',
             'src/__init__.py', 'src/data.py', 'src/metrics.py', 'src/postprocess.py']
    sources = {name: (root / name).read_text(encoding='utf8') for name in names}
    slug = 'filament-fallback-low-oof' if args.low_proposals else 'filament-fallback-oof'
    meta = {'id': 'oaonoir/' + slug, 'title': slug,
            'code_file': 'kernel.py', 'language': 'python', 'kernel_type': 'script',
            'is_private': 'true', 'enable_gpu': 'true', 'enable_internet': 'false',
            'dataset_sources': [], 'competition_sources': ['filament-segmentation-2026'],
            'kernel_sources': ['oaonoir/filament-baseline-fold0', 'oaonoir/filament-baseline-fold1']}
    script = '''import hashlib, json, os, subprocess, sys, time
from pathlib import Path
started = time.monotonic()
print('UTC_START', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
root = Path('/kaggle/working/source')
for name, source in SOURCES.items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf8')
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
data = [p for p in Path('/kaggle/input').rglob('train_images') if p.is_dir()]
if len(data) != 1: raise RuntimeError(f'Expected exactly one train_images, got {data}')
images = data[0]
annotation = images.parent / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
env = dict(os.environ, PYTHONPATH=str(root))
for fold in (1, 0):
    manifests = list(Path('/kaggle/input').rglob(f'provenance_fold{fold}.json'))
    if len(manifests) != 1: raise RuntimeError(f'Expected one provenance for fold {fold}')
    weights = manifests[0].parent
    manifest = json.loads(manifests[0].read_text())
    if manifest['status'] != 'complete' or manifest['fold'] != fold: raise RuntimeError('Invalid provenance')
    for name in (f'oof_fold{fold}.csv', 'final_detector.pt', 'final_refiner.pt'):
        if sha(weights / name) != manifest['outputs_sha256'][name]: raise RuntimeError(f'Hash mismatch: {name}')
    subprocess.run([sys.executable, '-m', 'baseline.fallback_oof', *EXTRA, '--fold', str(fold),
                    '--weights', str(weights), '--images', str(images), '--train-json', str(annotation),
                    '--output', f'/kaggle/working/fallback-fold{fold}.json'], env=env, check=True)
print('UTC_END', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'wall_seconds', time.monotonic()-started, flush=True)
'''.replace('SOURCES', repr(sources)).replace('EXTRA', repr(['--low-proposals'] if args.low_proposals else []))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'kernel.py').write_text(script, encoding='utf8')
    (args.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    print(f'Prepared {args.output}; not pushed')


if __name__ == '__main__':
    main()
