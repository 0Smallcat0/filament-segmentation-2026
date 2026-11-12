"""Build a private kernel to evaluate a fixed OOF threshold grid."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--detector-score', type=float)
    parser.add_argument('--refiner-threshold', type=float)
    args = parser.parse_args()
    if (args.detector_score is None) != (args.refiner_threshold is None):
        raise ValueError('Specify both selected thresholds')
    if (args.fold == 0) != (args.detector_score is not None):
        raise ValueError('Fold 1 selects the grid; fold 0 evaluates the selected setting')
    root = Path(__file__).resolve().parents[1]
    names = ['baseline/__init__.py', 'baseline/handoff.py', 'baseline/tune_handoff.py',
             'src/__init__.py', 'src/data.py', 'src/metrics.py', 'src/postprocess.py']
    sources = {name: (root / name).read_text(encoding='utf8') for name in names}
    slug = f'filament-handoff-tune-fold{args.fold}'
    meta = {'id': 'oaonoir/' + slug, 'title': slug, 'code_file': 'kernel.py',
            'language': 'python', 'kernel_type': 'script', 'is_private': 'true',
            'enable_gpu': 'true', 'enable_internet': 'false', 'dataset_sources': [],
            'competition_sources': ['filament-segmentation-2026'],
            'kernel_sources': [f'oaonoir/filament-baseline-fold{args.fold}']}
    if args.fold == 0:
        meta['kernel_sources'].append('oaonoir/filament-handoff-tune-fold1')
    extra = [] if args.detector_score is None else ['--detector-score', str(args.detector_score),
                                                     '--refiner-threshold', str(args.refiner_threshold)]
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
if len(data) != 1: raise RuntimeError('Expected exactly one train image directory')
images = data[0]
annotation = images.parent / 'MAGFiLO_1.0_Annotations_kaggle2026_train.json'
manifests = list(Path('/kaggle/input').rglob('provenance_foldFOLD.json'))
if len(manifests) != 1: raise RuntimeError('Expected exactly one fold manifest')
manifest = json.loads(manifests[0].read_text())
weights = manifests[0].parent
if manifest['status'] != 'complete' or manifest['fold'] != FOLD: raise RuntimeError('Invalid provenance')
if sha(annotation) != manifest['dataset']['annotation_sha256']: raise RuntimeError('Annotation hash mismatch')
for name in ('final_detector.pt', 'final_refiner.pt'):
    if sha(weights / name) != manifest['outputs_sha256'][name]: raise RuntimeError('Checkpoint hash mismatch')
env = dict(os.environ, PYTHONPATH=str(root))
SELECTION_SETUP
subprocess.run([sys.executable, '-m', 'baseline.tune_handoff', '--fold', str(FOLD), '--weights', str(weights),
                '--images', str(images), '--train-json', str(annotation), '--output', '/kaggle/working/tuning.json', *EXTRA, *selection_args], env=env, check=True)
print('UTC_END', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'wall_seconds', time.monotonic()-started, flush=True)
'''
    selection_setup = "candidates = list(Path('/kaggle/input').rglob('tuning.json'))\nif len(candidates) != 1: raise RuntimeError('Expected one fold-1 selection artifact')\nselection_args = ['--selection', str(candidates[0])]" if args.fold == 0 else 'selection_args = []'
    script = script.replace('SELECTION_SETUP', selection_setup).replace('FOLD', str(args.fold)).replace('EXTRA', repr(extra)).replace('SOURCES', repr(sources))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'kernel.py').write_text(script, encoding='utf8')
    (args.output / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf8')
    print(f'Prepared {slug}; not pushed')


if __name__ == '__main__':
    main()
