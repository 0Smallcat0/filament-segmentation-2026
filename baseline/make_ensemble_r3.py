"""Package the five-fold consensus inference as a private Kaggle kernel."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = ['baseline/__init__.py', 'baseline/handoff.py', 'baseline/grid_pooled.py',
             'baseline/detector_tta_r3.py', 'baseline/ensemble_r3.py', 'baseline/provenance.py',
             'src/__init__.py', 'src/data.py', 'src/metrics.py', 'src/postprocess.py',
             'check_submission.py', 'weights/test-inventory.json']
    sources = {name: (root / name).read_text(encoding='utf8') for name in names}
    profiles = []
    for fold in range(5):
        folder = root.parent / f'out-baseline-fold{fold}'
        manifests = list(folder.rglob(f'provenance_fold{fold}.json'))
        if len(manifests) != 1:
            raise ValueError(f'Expected unique local fold {fold} provenance')
        manifest = json.loads(manifests[0].read_text(encoding='utf8'))
        if manifest['status'] != 'complete' or manifest['fold'] != fold:
            raise ValueError(f'Unexpected fold {fold} manifest')
        profiles.append({'provenance_sha256': sha(manifests[0]), **{
            name: manifest['outputs_sha256'][name] for name in ('final_detector.pt', 'final_refiner.pt')}})
    slug = 'filament-fivefold-ensemble-r3'
    meta = {'id': 'oaonoir/' + slug, 'title': slug, 'code_file': 'kernel.py',
            'language': 'python', 'kernel_type': 'script', 'is_private': 'true',
            'enable_gpu': 'true', 'enable_internet': 'false', 'dataset_sources': [],
            'competition_sources': ['filament-segmentation-2026'],
            'kernel_sources': [f'oaonoir/filament-baseline-fold{i}' for i in range(5)]}
    script = '''import json, os, subprocess, sys, time
from pathlib import Path
started = time.monotonic()
print('UTC_START', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
root = Path('/kaggle/working/source')
for name, source in SOURCES.items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf8')
(root / 'profiles.json').write_text(json.dumps(PROFILES), encoding='utf8')
images = [p for p in Path('/kaggle/input').rglob('test_images') if p.is_dir()]
if len(images) != 1: raise RuntimeError('Expected one test image directory')
env = dict(os.environ, PYTHONPATH=str(root))
subprocess.run([sys.executable, '-m', 'baseline.ensemble_r3', '--images', str(images[0]),
                '--inventory', str(root / 'weights/test-inventory.json'), '--profiles', str(root / 'profiles.json'),
                '--output', '/kaggle/working'], env=env, check=True)
for k in (2, 3):
    subprocess.run([sys.executable, str(root / 'check_submission.py'), f'/kaggle/working/submission-k{k}.csv',
                    '--images', str(images[0])], env=env, check=True)
print('UTC_END', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'wall_seconds', time.monotonic()-started, flush=True)
'''.replace('SOURCES', repr(sources)).replace('PROFILES', repr(profiles))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'kernel.py').write_text(script, encoding='utf8')
    (args.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    print(f'Prepared {slug}; not pushed')


if __name__ == '__main__':
    main()
