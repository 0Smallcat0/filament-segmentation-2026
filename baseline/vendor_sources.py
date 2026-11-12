"""Vendor bounded portions of public Kaggle notebooks for offline use.

Run once with Kaggle CLI credentials. Does not retrieve datasets or predictions.
"""
import json
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi, ApiGetKernelRequest

HERE = Path(__file__).resolve().parent
api = KaggleApi()
api.authenticate()
client = api.build_kaggle_client()
request = ApiGetKernelRequest()
request.user_name = 'nomannic19'
request.kernel_slug = 'cv-solar-filament-mask-r-cnn-u-net-refiner'
notebook = json.loads(client.kernels.kernels_api_client.get_kernel(request).blob.source)
cells = notebook['cells']
source = lambda i: ''.join(cells[i]['source'])

# The training manifest contains only the four training folds. The source's
# "final" mode then trains on exactly that manifest, never on held-out data.
detector = source(21).split('\n', 1)[1]
detector = detector.replace('Path("/kaggle/input/competitions/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")',
                            'Path(os.environ["FILAMENT_DATA_ROOT"])')
detector = detector.replace('val_records = manifest[manifest["split"] == "val"]',
                            'val_records = manifest.iloc[0:0]')
detector = detector.replace('choices=["validation", "final"], default="validation"',
                            'choices=["final"], default="final"')
(HERE / 'detector.py').write_text(detector, encoding='utf8')

# Keep crop and model definitions only; execute training with our own manifest.
crop_defs = source(25).split('train_rows = manifest', 1)[0]
crop_defs = crop_defs.replace('from tqdm.auto import tqdm',
    'import cv2, numpy as np, pandas as pd, torch\nfrom tqdm.auto import tqdm')
refiner_defs = source(26).split('train_crop_dataset =', 1)[0]
(HERE / 'refiner.py').write_text(crop_defs + '\n' + refiner_defs, encoding='utf8')

handoff = json.loads((HERE.parent.parent / 'kernel-ours' / 'filament-run.ipynb').read_text(encoding='utf8'))
code = ''.join(handoff['cells'][4]['source']).split('\ndef main() -> None:', 1)[0]
(HERE / 'handoff.py').write_text(code + '\n', encoding='utf8')
print('Vendored detector, refiner definitions and handoff inference; no data downloaded.')
