"""Prepare, but never upload, one private Kaggle YOLO fold."""
import argparse
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, choices=range(5), required=True)
    p.add_argument('--slug', required=True)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--size', type=int, default=1024)
    p.add_argument('--batch', type=int, default=2)
    args = p.parse_args()
    if not re.fullmatch(r'filament-[a-z0-9-]+', args.slug): raise ValueError('invalid private slug')
    root = Path(__file__).resolve().parents[1]
    files = [*sorted((root / 'src').glob('*.py')), *sorted((root / 'yolo').glob('*.py'))]
    sources = {p.relative_to(root).as_posix(): p.read_text(encoding='utf8') for p in files}
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {'id': 'oaonoir/' + args.slug, 'title': args.slug, 'code_file': 'kernel.py',
                'language': 'python', 'kernel_type': 'script', 'is_private': True,
                'enable_gpu': True, 'enable_internet': True, 'dataset_sources': [],
                'competition_sources': ['filament-segmentation-2026'], 'kernel_sources': []}
    (args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    script = '''import os, subprocess, sys, time
from pathlib import Path
started = time.monotonic()
print('UTC_START', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
try:
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'ultralytics==8.4.103', 'pycocotools==2.0.11'], check=True)
    subprocess.run(['nvidia-smi'], check=True)
    root = Path('/kaggle/working/filament_source')
    for name, source in __SOURCES__.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding='utf8')
    labels = list(Path('/kaggle/input').rglob('MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
    if len(labels) != 1: raise ValueError(f'Expected exactly one competition train JSON: {labels}')
    images = labels[0].parent / 'train_images'
    if not images.is_dir(): raise FileNotFoundError(images)
    cmd = [sys.executable, '-m', 'yolo.fold', '--train-json', str(labels[0]), '--train-images', str(images), '--output', '/kaggle/working/results', '--fold', __FOLD__, '--epochs', __EPOCHS__, '--size', __SIZE__, '--batch', __BATCH__]
    print('COMMAND', cmd, flush=True)
    subprocess.run(cmd, cwd=str(root), check=True)
finally:
    print('UTC_END', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'wall_seconds', time.monotonic() - started, flush=True)
'''
    for key, value in {'__FOLD__': repr(str(args.fold)), '__EPOCHS__': repr(str(args.epochs)),
                       '__SIZE__': repr(str(args.size)), '__BATCH__': repr(str(args.batch))}.items():
        script = script.replace(key, value)
    script = script.replace('__SOURCES__', repr(sources))
    (args.output / 'kernel.py').write_text(script, encoding='utf8')
    print(f'Prepared private fold {args.fold} in {args.output}; not pushed')


if __name__ == '__main__': main()
