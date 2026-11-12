"""Package a private fold-0 detector TTA comparison without pushing it."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = ['baseline/__init__.py', 'baseline/handoff.py', 'baseline/grid_pooled.py',
             'baseline/detector_tta_r3.py', 'baseline/provenance.py', 'src/__init__.py',
             'src/data.py', 'src/metrics.py', 'src/postprocess.py']
    sources = {name: (root / name).read_text(encoding='utf8') for name in names}
    slug = 'filament-detector-tta-fold0-r3'
    meta = {'id': 'oaonoir/' + slug, 'title': slug, 'code_file': 'kernel.py',
            'language': 'python', 'kernel_type': 'script', 'is_private': 'true',
            'enable_gpu': 'true', 'enable_internet': 'false', 'dataset_sources': [],
            'competition_sources': ['filament-segmentation-2026'],
            'kernel_sources': ['oaonoir/filament-baseline-fold0']}
    script = '''import os, subprocess, sys, time
from pathlib import Path
started = time.monotonic()
print('UTC_START', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
root = Path('/kaggle/working/source')
for name, source in SOURCES.items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf8')
data = [p for p in Path('/kaggle/input').rglob('train_images') if p.is_dir()]
if len(data) != 1: raise RuntimeError('Expected exactly one train image directory')
images = data[0]
annotation = images.parent / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
manifests = list(Path('/kaggle/input').rglob('provenance_fold0.json'))
if len(manifests) != 1: raise RuntimeError('Expected exactly one fold manifest')
env = dict(os.environ, PYTHONPATH=str(root))
subprocess.run([sys.executable, '-m', 'baseline.detector_tta_r3', '--weights', str(manifests[0].parent),
                '--images', str(images), '--train-json', str(annotation),
                '--output', '/kaggle/working/detector-tta.json'], env=env, check=True)
print('UTC_END', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'wall_seconds', time.monotonic()-started, flush=True)
'''.replace('SOURCES', repr(sources))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'kernel.py').write_text(script, encoding='utf8')
    (args.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    print(f'Prepared {slug}; not pushed')


if __name__ == '__main__':
    main()
