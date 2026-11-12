"""Reproducible model inference and COCO submission writing."""
import csv
from pathlib import Path
import cv2
import numpy as np
import torch
from .model import build_model
from .postprocess import instances, assert_disjoint


def load_model(weight_path, device):
    """Load our state dictionary; pretrained initialization is disabled."""
    model = build_model(pretrained=False).to(device)
    checkpoint = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model


def predict(model, path, device, size=1024, tta=False):
    """Predict at fixed resolution and resize probabilities to native image size."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None: raise FileNotFoundError(path)
    x = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = torch.from_numpy(x)[None, None].to(device)
    with torch.no_grad():
        maps = [torch.sigmoid(model(x))]
        if tta:
            for dims in [(2,), (3,), (2,3)]:
                maps.append(torch.flip(torch.sigmoid(model(torch.flip(x, dims))), dims))
        prob = torch.stack(maps).mean(0)[0,0].cpu().numpy()
    return cv2.resize(prob, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)


def write_submission(rows, path):
    """Write only compressed counts; validate non-overlap for each image."""
    with Path(path).open('w', newline='', encoding='utf8') as f:
        writer = csv.writer(f); writer.writerow(['filament_id','segmentation_rle'])
        for stem, rles in rows:
            assert_disjoint(rles)
            for i, rle in enumerate(rles, 1):
                writer.writerow([f'{stem}_{i}', rle['counts'].decode('ascii')])
