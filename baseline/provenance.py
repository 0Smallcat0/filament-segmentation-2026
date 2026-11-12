"""Deterministic provenance for independently trained OOF folds."""
import hashlib
import json
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True)
    return hashlib.sha256(payload.encode('utf8')).hexdigest()


def dataset_manifest(annotation, image_root, records):
    """Bind label bytes, observation image bytes, medoids, and fold assignments."""
    image_root = Path(image_root)
    observations = []
    for record in sorted(records, key=lambda r: r['filename']):
        name = record['filename']
        path = image_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        observations.append({'filename': name, 'image_sha256': file_sha256(path),
                             'medoid_image_id': record['image_id'], 'fold': record['fold']})
    data = {'annotation_sha256': file_sha256(annotation), 'observations': observations}
    return {'sha256': object_sha256(data), **data}


def source_manifest(root):
    root = Path(root)
    paths = sorted((root / 'baseline').glob('*.py')) + sorted((root / 'src').glob('*.py'))
    files = {p.relative_to(root).as_posix(): file_sha256(p) for p in paths}
    return {'sha256': object_sha256(files), 'files': files}
