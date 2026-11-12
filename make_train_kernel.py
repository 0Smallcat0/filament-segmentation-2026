"""Package the current source as a private Kaggle script kernel; never push it."""
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--slug', required=True, help='private kernel slug, e.g. filament-fold0')
    parser.add_argument('--fold', required=True, type=int, choices=range(5))
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--tta', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'filament-[a-z0-9-]+', args.slug):
        raise ValueError('slug must start filament- and contain lowercase ASCII, digits, and hyphens')
    root = Path(__file__).resolve().parent
    source_paths = [root / 'train.py', *sorted((root / 'src').glob('*.py'))]
    sources = {p.relative_to(root).as_posix(): p.read_text(encoding='utf8') for p in source_paths}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {'id': 'oaonoir/' + args.slug, 'title': args.slug, 'code_file': 'kernel.py',
                'language': 'python', 'kernel_type': 'script', 'is_private': 'true',
                'enable_gpu': 'true', 'enable_internet': 'true',
                'dataset_sources': [], 'competition_sources': ['filament-segmentation-2026'], 'kernel_sources': []}
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    script = '''import os, subprocess, sys, time
from pathlib import Path
print('UTC_START', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', 'segmentation-models-pytorch==0.4.0', 'pycocotools==2.0.10'], check=True)
subprocess.run(['nvidia-smi'], check=True)
sources = SOURCES
root = Path('/kaggle/working/filament_source')
for name, content in sources.items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf8')
labels = list(Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
if len(labels) != 1: raise FileNotFoundError(f'Expected one competition train JSON; found {labels}')
label = labels[0]
images = label.parent / 'train_images'
if not images.is_dir(): raise FileNotFoundError(f'Expected train images beside {label}: {images}')
cmd = [sys.executable, str(root / 'train.py'), '--train-json', str(label), '--train-images', str(images), '--output', '/kaggle/working/results', '--fold', FOLD, '--epochs', EPOCHS, '--size', SIZE, '--batch-size', BATCH] + TTA
print('command:', cmd, flush=True)
subprocess.run(cmd, cwd=str(root), check=True)
print('UTC_END', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
'''
    for name, value in {'FOLD': repr(str(args.fold)), 'EPOCHS': repr(str(args.epochs)),
                        'SIZE': repr(str(args.size)), 'BATCH': repr(str(args.batch_size)),
                        'TTA': repr(['--tta'] if args.tta else []), 'SOURCES': repr(sources)}.items():
        script = script.replace(name, value)
    (output / 'kernel.py').write_text(script, encoding='utf8')
    print(f'Prepared {output}; inspect then run kaggle kernels push -p {output}')


if __name__ == '__main__':
    main()
