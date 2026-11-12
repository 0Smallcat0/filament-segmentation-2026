"""Build a private test-only inference kernel using our fold-0 training output."""
import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--verify-notebook', action='store_true')
    parser.add_argument('--model-profile', type=Path)
    parser.add_argument('--model-kernel', default='oaonoir/filament-baseline-fold0')
    parser.add_argument('--slug', default=None)
    parser.add_argument('--detector-score', type=float, choices=(0.7, 0.8), default=0.7)
    parser.add_argument('--fold0-fallback', action='store_true')
    parser.add_argument('--grid-config', type=Path)
    args = parser.parse_args()
    if args.verify_notebook and args.model_profile:
        raise ValueError('Notebook verification is configured for the submitted fold-0 model')
    if args.fold0_fallback and (not args.model_profile or args.verify_notebook):
        raise ValueError('Fold-0 fallback requires a separate profiled model')
    if args.grid_config and (args.fold0_fallback or args.verify_notebook or not args.slug
                             or args.slug in ('filament-baseline-predict', 'filament-baseline-reproduce')):
        raise ValueError('Grid settings require their own non-notebook, same-model kernel')
    if args.detector_score != 0.7 and (not args.slug or args.slug in ('filament-baseline-predict', 'filament-baseline-reproduce')):
        raise ValueError('A different threshold needs a separate kernel slug')
    root = Path(__file__).resolve().parents[1]
    names = ['baseline/__init__.py', 'baseline/handoff.py', 'baseline/predict.py', 'baseline/grid_pooled.py',
             'baseline/provenance.py', 'baseline/split_weights.py', 'src/__init__.py', 'src/data.py',
             'src/metrics.py', 'src/postprocess.py',
             'check_submission.py', 'weights/test-inventory.json']
    if args.verify_notebook:
        names.append('notebooks/pipeline.ipynb')
    sources = {name: (root / name).read_text(encoding='utf8') for name in names}
    profile = json.loads(args.model_profile.read_text(encoding='utf8')) if args.model_profile else None
    if args.model_profile:
        if not isinstance(profile, dict) or not profile:
            raise ValueError('Model profile must be a nonempty object')
        if not args.slug or args.slug in ('filament-baseline-predict', 'filament-baseline-reproduce') or args.model_kernel == 'oaonoir/filament-baseline-fold0':
            raise ValueError('A custom model needs its own slug and training kernel')
        for key in ('original_detector_sha256', 'detector_state_sha256', 'refiner_sha256', 'provenance_sha256'):
            if not re.fullmatch(r'[0-9a-f]{64}', profile[key]):
                raise ValueError(f'Invalid SHA-256: {key}')
        sources['weights/checkpoint-profile.json'] = json.dumps(profile, indent=2)
    slug = args.slug or ('filament-baseline-reproduce' if args.verify_notebook else 'filament-baseline-predict')
    if not slug.startswith('filament-') or not all(c.islower() or c.isdigit() or c == '-' for c in slug):
        raise ValueError('Invalid private kernel slug')
    metadata = {'id': 'oaonoir/' + slug, 'title': slug,
                'code_file': 'kernel.py', 'language': 'python', 'kernel_type': 'script',
                'is_private': 'true', 'enable_gpu': 'true', 'enable_internet': 'false',
                'dataset_sources': [], 'competition_sources': ['filament-segmentation-2026'],
                'kernel_sources': [args.model_kernel] + (['oaonoir/filament-baseline-fold0'] if args.fold0_fallback else [])}
    script = '''import hashlib, os, shutil, subprocess, sys, time
from pathlib import Path
started = time.monotonic()
print('UTC_START', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), flush=True)
root = Path('/kaggle/working/source')
for name, source in SOURCES.items():
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding='utf8')
def find_one(name):
    found = list(Path('/kaggle/input').rglob(name))
    if len(found) != 1: raise RuntimeError(f'Expected one {name}, got {len(found)}')
    return found[0]
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()
detector, refiner = find_one('final_detector.pt'), find_one('final_refiner.pt')
if sha(detector) != '999532f6bf189d6c60119ea097b2c273c726ab08b201ddbaf783e2e14289d743': raise RuntimeError('Detector hash mismatch')
if sha(refiner) != '95ff4976ae06e7022c1f207290a355fdf5a8758a1d47dd660b9bb1f4e3ecfdc4': raise RuntimeError('Refiner hash mismatch')
weights = Path('/kaggle/working/weights')
subprocess.run([sys.executable, str(root / 'baseline/split_weights.py'), str(detector), str(weights)], check=True)
shutil.copyfile(refiner, weights / 'final_refiner.pt')
images = find_one('test_images')
out = Path('/kaggle/working/submission.csv')
env = dict(os.environ, PYTHONPATH=str(root))
RUN_INFERENCE
subprocess.run([sys.executable, str(root / 'check_submission.py'), str(out), '--images', str(images)], env=env, check=True)
VERIFY_HASH
print('UTC_END', time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'wall_seconds', time.monotonic()-started, flush=True)
'''
    if args.verify_notebook:
        run = '''(root / 'weights' / 'baseline-fold0').mkdir(parents=True)
for path in weights.iterdir(): shutil.copyfile(path, root / 'weights' / 'baseline-fold0' / path.name)
env.update({'FILAMENT_REPO': str(root), 'FILAMENT_TEST_IMAGES': str(images), 'FILAMENT_OUTPUT': str(out), 'FILAMENT_DETECTOR_SCORE': __THRESHOLD_VALUE__})
subprocess.run([sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', '--ExecutePreprocessor.timeout=3600', '--output', '/kaggle/working/executed.ipynb', str(root / 'notebooks/pipeline.ipynb')], env=env, cwd=str(root), check=True)'''
        run = run.replace('__THRESHOLD_VALUE__', repr(str(args.detector_score)))
        expected = ('31f6dd9e0f21b09174bc17ffa06d9e8996c7c046fdabde805d0686b11055a850'
                    if args.detector_score == 0.7 else
                    'e1d77f9dd6d1b37df1dc629e186c3f63abcbeba7eab8e1f72d0ca12ffb6ea7d7')
        verify = f"if sha(out) != '{expected}': raise RuntimeError('Submission differs from reference')\nprint('MATCHED_SHA256', sha(out), flush=True)"
    else:
        run = "subprocess.run([sys.executable, '-m', 'baseline.predict', '--images', str(images), '--weights', str(weights), '--inventory', str(root / 'weights/test-inventory.json'), '--output', str(out)], env=env, check=True)"
        run = run.replace("str(out)], env=env", "str(out), '--detector-score', '" + str(args.detector_score) + "'], env=env")
        if profile:
            run = run.replace("], env=env", ", '--profile', str(root / 'weights/checkpoint-profile.json')], env=env")
        verify = ''
    if profile:
        script = script.replace('999532f6bf189d6c60119ea097b2c273c726ab08b201ddbaf783e2e14289d743', profile['original_detector_sha256'])
        script = script.replace('95ff4976ae06e7022c1f207290a355fdf5a8758a1d47dd660b9bb1f4e3ecfdc4', profile['refiner_sha256'])
        script = script.replace("detector, refiner = find_one('final_detector.pt'), find_one('final_refiner.pt')", "provenance = find_one('provenance_fold0.json')\nif sha(provenance) != '" + profile['provenance_sha256'] + "': raise RuntimeError('Training provenance hash mismatch')\ndetector, refiner = find_one('final_detector.pt'), find_one('final_refiner.pt')")
    if args.fold0_fallback:
        # Both source kernels have identically named files. Identify them by their
        # training manifest hashes, never by directory ordering.
        old = "provenance = find_one('provenance_fold0.json')\nif sha(provenance) != '" + profile['provenance_sha256'] + "': raise RuntimeError('Training provenance hash mismatch')\ndetector, refiner = find_one('final_detector.pt'), find_one('final_refiner.pt')"
        replacement = """manifests = list(Path('/kaggle/input').rglob('provenance_fold0.json'))
matching = [p for p in manifests if sha(p) == PROFILE_HASH]
if len(matching) != 1: raise RuntimeError('Full model provenance mismatch')
primary = matching[0].parent
other = [p for p in manifests if p != matching[0]]
if len(other) != 1: raise RuntimeError('Expected exactly one fold0 provenance')
fallback = other[0].parent
detector, refiner = primary / 'final_detector.pt', primary / 'final_refiner.pt'
if sha(fallback / 'final_detector.pt') != '999532f6bf189d6c60119ea097b2c273c726ab08b201ddbaf783e2e14289d743': raise RuntimeError('Fallback detector hash mismatch')
if sha(fallback / 'final_refiner.pt') != '95ff4976ae06e7022c1f207290a355fdf5a8758a1d47dd660b9bb1f4e3ecfdc4': raise RuntimeError('Fallback refiner hash mismatch')
fallback_weights = Path('/kaggle/working/fallback-weights')
subprocess.run([sys.executable, str(root / 'baseline/split_weights.py'), str(fallback / 'final_detector.pt'), str(fallback_weights)], check=True)
shutil.copyfile(fallback / 'final_refiner.pt', fallback_weights / 'final_refiner.pt')""".replace('PROFILE_HASH', repr(profile['provenance_sha256']))
        if old not in script:
            raise ValueError('Expected primary provenance wrapper missing')
        script = script.replace(old, replacement)
        run = run.replace("], env=env", ", '--fallback-weights', str(fallback_weights)], env=env")
    if args.grid_config:
        config_bytes = args.grid_config.read_bytes()
        json.loads(config_bytes)
        script = script.replace("def find_one(name):", "(root / 'grid-selection.json').write_bytes(bytes.fromhex(" + repr(config_bytes.hex()) + "))\ndef find_one(name):")
        run = run.replace("], env=env", ", '--grid-config', str(root / 'grid-selection.json')], env=env")
    script = script.replace('RUN_INFERENCE', run).replace('VERIFY_HASH', verify).replace('SOURCES', repr(sources))
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'kernel.py').write_text(script, encoding='utf8')
    (args.output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    print(f'Prepared {args.output}; not pushed')


if __name__ == '__main__':
    main()
