"""Train one observation-grouped fold on Kaggle GPU and save OOF predictions."""
import argparse, json, os, random, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset, DataLoader
from src.data import prepare_train
from src.model import build_model
from src.inference import predict, write_submission
from src.metrics import overlap_matrix, score_matrices
from src.postprocess import instances
from src.provenance import environment, sha256_file, source_hash

class TrainImages(Dataset):
    """Resize train images and medoid semantic masks, with random flips."""
    def __init__(self, records, root, size): self.records, self.root, self.size = records, root, size
    def __len__(self): return len(self.records)
    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(self.root / record['filename']), cv2.IMREAD_GRAYSCALE)
        if image is None: raise FileNotFoundError(record['filename'])
        mask = np.zeros(image.shape, dtype=np.uint8)
        for rle in record['rles']: mask |= mask_utils.decode(rle)
        image = cv2.resize(image, (self.size,self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size,self.size), interpolation=cv2.INTER_NEAREST)
        if random.random() < .5: image, mask = np.fliplr(image), np.fliplr(mask)
        if random.random() < .5: image, mask = np.flipud(image), np.flipud(mask)
        return torch.from_numpy(image.copy()).float()[None] / 255, torch.from_numpy(mask.copy()).float()[None]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-json', required=True); parser.add_argument('--train-images', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--epochs', type=int, default=15); parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=2); parser.add_argument('--seed', type=int, default=20261109)
    parser.add_argument('--tta', action='store_true', help='save held-out probabilities with flip TTA')
    args = parser.parse_args()
    if not os.environ.get('KAGGLE_KERNEL_RUN_TYPE') or not torch.cuda.is_available():
        raise RuntimeError('Training is restricted to a Kaggle GPU kernel')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    started = time.monotonic()
    records = prepare_train(args.train_json, seed=args.seed)
    train = [r for r in records if r['fold'] != args.fold]
    val = [r for r in records if r['fold'] == args.fold]
    if not train or not val: raise ValueError('invalid fold')
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(TrainImages(train, Path(args.train_images), args.size), batch_size=args.batch_size, shuffle=True, num_workers=2)
    model = build_model(pretrained=True).cuda(); optim = torch.optim.AdamW(model.parameters(), lr=0.0003)
    criterion = torch.nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        model.train(); losses = []
        for x, y in loader:
            x, y = x.cuda(), y.cuda(); optim.zero_grad(set_to_none=True); logits = model(x)
            prob = logits.sigmoid(); dice = 1 - (2*(prob*y).sum((2,3))+1)/(prob.sum((2,3))+y.sum((2,3))+1)
            loss = criterion(logits, y) + dice.mean(); loss.backward(); optim.step(); losses.append(loss.item())
        print(f'epoch={epoch+1} loss={np.mean(losses):.5f}', flush=True)
    checkpoint = output / f'fold{args.fold}.pt'
    torch.save({'model': model.cpu().state_dict(), 'size': args.size, 'fold': args.fold, 'seed': args.seed}, checkpoint)
    model.cuda().eval(); rows, matrices = [], []
    probability_dir = output / f'probabilities_fold{args.fold}'
    probability_dir.mkdir(exist_ok=True)
    map_hashes = {}
    for record in val:
        prob = predict(model, Path(args.train_images) / record['filename'], 'cuda', size=args.size, tta=args.tta)
        # Quantize before both scoring and storage so tuning can reproduce these maps.
        quantized = np.rint(np.clip(prob, 0, 1) * 65535).astype(np.uint16)
        probability_path = probability_dir / (Path(record['filename']).stem + '.png')
        if not cv2.imwrite(str(probability_path), quantized): raise RuntimeError(f'cannot write {probability_path}')
        map_hashes[Path(record['filename']).stem] = sha256_file(probability_path)
        prob = quantized.astype(np.float32) / 65535
        rles = instances(prob); rows.append((Path(record['filename']).stem, rles)); matrices.append(overlap_matrix(record['rles'], rles))
    write_submission(rows, output / f'oof_fold{args.fold}.csv')
    (output / f'score_fold{args.fold}.json').write_text(json.dumps(score_matrices(matrices), indent=2))
    execution_env = environment()
    execution_env.update({'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                          'gpu_name': torch.cuda.get_device_name(0)})
    manifest = {'schema': 1, 'fold': args.fold, 'folds': 5, 'seed': args.seed,
                'config': {'size': args.size, 'epochs': args.epochs, 'batch_size': args.batch_size, 'tta': args.tta},
                'source_sha256': source_hash(Path(__file__).resolve().parent),
                'train_json_sha256': sha256_file(args.train_json),
                'validation_filenames': sorted(r['filename'] for r in val),
                'checkpoint_sha256': sha256_file(checkpoint), 'probability_sha256': map_hashes,
                'probability_format': 'uint16 png, divide by 65535',
                'seconds': time.monotonic() - started, 'environment': execution_env}
    # Written last: a missing manifest marks an incomplete or legacy run.
    (output / f'manifest_fold{args.fold}.json').write_text(json.dumps(manifest, indent=2), encoding='utf8')
    print('saved', checkpoint, flush=True)

if __name__ == '__main__':
    if '--baseline-fold' in sys.argv:
        index = sys.argv.index('--baseline-fold')
        sys.argv[index] = '--fold'
        from baseline.run_fold import main as train_baseline
        train_baseline()
    else:
        main()
