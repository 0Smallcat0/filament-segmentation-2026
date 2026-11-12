"""Package the fold-1 rescue comparison as a private Kaggle job."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    names = ['baseline/__init__.py', 'baseline/handoff.py', 'baseline/provenance.py', 'baseline/compare_fallback_r3.py',
             'src/__init__.py', 'src/data.py', 'src/metrics.py', 'src/postprocess.py']
    sources = {name: (root / name).read_text(encoding='utf8') for name in names}
    meta = {'id': 'oaonoir/filament-fallback-compare-r3', 'title': 'filament-fallback-compare-r3',
            'code_file': 'kernel.py', 'language': 'python', 'kernel_type': 'script', 'is_private': 'true',
            'enable_gpu': 'true', 'enable_internet': 'false', 'dataset_sources': [],
            'competition_sources': ['filament-segmentation-2026'],
            'kernel_sources': ['oaonoir/filament-baseline-fold0', 'oaonoir/filament-baseline-fold1']}
    script = '''import os, subprocess, sys, time
from pathlib import Path
start = time.monotonic()
print('UTC_START', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
root = Path('/kaggle/working/source')
for name, source in SOURCES.items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf8')
def one(name):
    found = list(Path('/kaggle/input').rglob(name))
    if len(found) != 1: raise ValueError(f'Expected one {name}: {found}')
    return found[0]
images = one('train_images')
annotation = images.parent / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
env = dict(os.environ, PYTHONPATH=str(root))
subprocess.run([sys.executable, '-m', 'baseline.compare_fallback_r3', '--train-json', str(annotation),
                '--images', str(images), '--fold0', str(one('provenance_fold0.json').parent),
                '--fold1', str(one('provenance_fold1.json').parent), '--output', '/kaggle/working/comparison.json'],
               env=env, check=True)
print('UTC_END', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'wall_seconds', time.monotonic()-start, flush=True)
'''.replace('SOURCES', repr(sources))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'kernel.py').write_text(script, encoding='utf8')
    (args.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf8')


if __name__ == '__main__':
    main()
