"""Build a private Kaggle GPU kernel for exactly one baseline fold."""
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', required=True, type=int, choices=range(5))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--full-train', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'filament-[a-z0-9-]+', args.slug):
        raise ValueError('slug must start filament- and contain lowercase ASCII, digits, and hyphens')
    root = Path(__file__).resolve().parents[1]
    names = ['baseline/__init__.py', 'baseline/detector.py', 'baseline/refiner.py',
             'baseline/handoff.py', 'baseline/run_fold.py', 'baseline/provenance.py']
    names += [p.relative_to(root).as_posix() for p in sorted((root / 'src').glob('*.py'))]
    sources = {name: (root / name).read_text(encoding='utf8') for name in names}
    args.output.mkdir(parents=True, exist_ok=True)
    meta = {'id': 'oaonoir/' + args.slug, 'title': args.slug, 'code_file': 'kernel.py',
            'language': 'python', 'kernel_type': 'script', 'is_private': 'true',
            'enable_gpu': 'true', 'enable_internet': 'true', 'dataset_sources': [],
            'competition_sources': ['filament-segmentation-2026'], 'kernel_sources': []}
    (args.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    script = '''import os, subprocess, sys, time
from pathlib import Path
started = time.monotonic()
print('UTC_START', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'segmentation-models-pytorch==0.4.0', 'pycocotools==2.0.10'], check=True)
    subprocess.run(['nvidia-smi'], check=True)
    root = Path('/kaggle/working/filament_source')
    for name, source in SOURCES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf8')
    candidates = [Path('/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026'),
                  Path('/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026')]
    data_root = next((p for p in candidates if (p / 'train/train_images').is_dir()), None)
    if data_root is None: raise FileNotFoundError('Competition train data not mounted')
    os.environ['FILAMENT_DATA_ROOT'] = str(data_root)
    subprocess.run([sys.executable, '-m', 'baseline.run_fold', '--fold', FOLD, *EXTRA], cwd=str(root), check=True)
finally:
    print('UTC_END', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'wall_seconds', time.monotonic()-started, flush=True)
'''
    script = script.replace('SOURCES', repr(sources)).replace('FOLD', repr(str(args.fold))).replace('EXTRA', repr(['--full-train'] if args.full_train else []))
    (args.output / 'kernel.py').write_text(script, encoding='utf8')
    print(f'Prepared private kernel in {args.output}; not pushed')


if __name__ == '__main__':
    main()
