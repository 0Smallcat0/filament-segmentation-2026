"""Hash input and output artifacts and record the execution environment."""
import hashlib
import importlib.metadata
import platform
from pathlib import Path


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_hash(root):
    root = Path(root)
    paths = [root / 'train.py', *sorted((root / 'src').glob('*.py'))]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode('utf8'))
        digest.update(b'\0')
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def environment():
    packages = ['numpy', 'scipy', 'opencv-python', 'opencv-python-headless', 'pycocotools', 'torch', 'torchvision', 'segmentation-models-pytorch']
    versions = {}
    for name in packages:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': platform.python_version(), 'platform': platform.platform(), 'packages': versions}
