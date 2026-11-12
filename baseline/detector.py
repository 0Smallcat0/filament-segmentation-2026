import os, argparse, random, time
from pathlib import Path
import cv2, numpy as np, pandas as pd, torch, torch.distributed as dist
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from tqdm.auto import tqdm

ROOT = Path(os.environ["FILAMENT_DATA_ROOT"])
ANN_PATH, IMAGE_DIR = ROOT / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json", ROOT / "train/train_images"
WORK_DIR = Path("/kaggle/working/solar_filaments_final")
MANIFEST_PATH = WORK_DIR / "training_manifest.csv"
IMAGE_SIZE, SCORE_THRESHOLD, MASK_THRESHOLD, SEED = 1024, 0.60, 0.50, 2026
cv2.setNumThreads(0)

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed); np.random.seed(seed)

def masks_to_boxes(masks):
    boxes = []
    for mask in masks:
        ys, xs = np.where(mask)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

def encode_mask(mask):
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))

def merge_overlaps(masks):
    masks = [np.asarray(mask, dtype=bool) for mask in masks if np.any(mask)]
    if len(masks) < 2:
        return masks
    rles = [encode_mask(mask) for mask in masks]
    adjacency = mask_utils.iou(rles, rles, [0] * len(rles)) > 0
    groups, seen = [], set()
    for start in range(len(masks)):
        if start in seen:
            continue
        stack, group = [start], []
        while stack:
            index = stack.pop()
            if index in seen:
                continue
            seen.add(index); group.append(index)
            stack.extend(np.flatnonzero(adjacency[index]).tolist())
        groups.append(group)
    return [np.logical_or.reduce([masks[index] for index in group]) for group in groups]

def instance_metrics(pred_masks, true_masks):
    p, t = len(pred_masks), len(true_masks)
    if p and t:
        iou = mask_utils.iou([encode_mask(mask) for mask in pred_masks], [encode_mask(mask) for mask in true_masks], [0] * t)
        dice = 2 * iou / (1 + iou + 1e-7)
        rows, cols = linear_sum_assignment(-dice)
        mean_dice = float(dice[rows, cols].sum() / max(p, t))
        tp = int((iou[rows, cols] >= 0.5).sum())
    else:
        mean_dice, tp = float(p == 0 and t == 0), 0

    if p or t:
        shape = pred_masks[0].shape if p else true_masks[0].shape
        pred_union = np.logical_or.reduce(pred_masks) if p else np.zeros(shape, bool)
        true_union = np.logical_or.reduce(true_masks) if t else np.zeros(shape, bool)
        intersection = np.logical_and(pred_union, true_union).sum()
        union_dice = float(2 * intersection / (pred_union.sum() + true_union.sum() + 1e-7))
    else:
        union_dice = 1.0

    precision, recall = tp / p if p else 0.0, tp / t if t else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"mean_dice": mean_dice, "union_dice": union_dice, "f1_50": f1, "count_error": abs(p - t), "predictions": p}

class FilamentDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records, self.augment = records.reset_index(drop=True), augment
        self.coco = COCO(str(ANN_PATH))

    def __len__(self):
        return len(self.records)

    @staticmethod
    def photometric(image):
        valid = image > 0.01

        if np.random.rand() < 0.70:
            image[valid] = image[valid] ** np.random.uniform(0.80, 1.25)

        if np.random.rand() < 0.70:
            mean = image[valid].mean()
            contrast, gain = np.random.uniform(0.85, 1.15), np.random.uniform(0.90, 1.10)
            image[valid] = ((image[valid] - mean) * contrast + mean) * gain

        if np.random.rand() < 0.15:
            blurred = cv2.GaussianBlur(image, (0, 0), np.random.uniform(0.30, 0.80))
            image[valid] = blurred[valid]

        if np.random.rand() < 0.25:
            image[valid] += np.random.normal(0, np.random.uniform(0.003, 0.015), valid.sum())

        return np.clip(image, 0, 1)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        image_id = str(row["id"])
        image = cv2.imread(str(IMAGE_DIR / row["file_name"]), cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32) / 255

        masks = []
        for annotation in self.coco.imgToAnns[image_id]:
            mask = cv2.resize(self.coco.annToMask(annotation), (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST).astype(bool)
            if np.any(mask):
                masks.append(mask)
        masks = np.stack(masks) if masks else np.zeros((0, IMAGE_SIZE, IMAGE_SIZE), bool)

        if self.augment:
            rotations = np.random.randint(4)
            image = np.rot90(image, rotations)
            masks = np.rot90(masks, rotations, axes=(1, 2))

            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1)
                masks = np.flip(masks, axis=2)

            image = self.photometric(image)

        image, masks = np.ascontiguousarray(image), np.ascontiguousarray(masks)
        boxes = masks_to_boxes(masks)
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32), "labels": torch.ones(len(masks), dtype=torch.int64), "masks": torch.as_tensor(masks, dtype=torch.uint8), "image_id": torch.tensor([index]), "area": torch.as_tensor(masks.sum((1, 2)), dtype=torch.float32), "iscrowd": torch.zeros(len(masks), dtype=torch.int64)}
        image = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
        return image, target, str(row["observation_id"])

def collate_fn(batch):
    return tuple(zip(*batch))

def build_model(pretrained=True):
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None)

    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, hidden, 2)

    model.rpn.anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)), aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=28, sampling_ratio=2)
    model.transform.min_size, model.transform.max_size = (IMAGE_SIZE,), IMAGE_SIZE
    return model

@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    values = []

    for images, targets, _ in tqdm(loader, desc="Validation", leave=False):
        output = model([images[0].to(device, non_blocking=True)])[0]
        keep = output["scores"] >= SCORE_THRESHOLD
        pred_masks = (output["masks"][keep, 0] >= MASK_THRESHOLD).byte().cpu().numpy().astype(bool)
        pred_masks = merge_overlaps(pred_masks)
        true_masks = targets[0]["masks"].numpy().astype(bool)
        values.append(instance_metrics(pred_masks, true_masks))

    return pd.DataFrame(values).mean().to_dict()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["final"], default="final")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    seed_everything(SEED + rank)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    manifest = pd.read_csv(MANIFEST_PATH, dtype={"id": str})
    train_records = manifest if args.mode == "final" else manifest[manifest["split"] == "train"]
    val_records = manifest.iloc[0:0]

    train_dataset = FilamentDataset(train_records, augment=True)
    sampler = DistributedSampler(train_dataset, shuffle=True, seed=SEED)
    train_loader = DataLoader(train_dataset, batch_size=1, sampler=sampler, num_workers=2, pin_memory=True, persistent_workers=True, collate_fn=collate_fn, worker_init_fn=seed_worker)

    val_loader = None
    if rank == 0 and args.mode == "validation":
        val_dataset = FilamentDataset(val_records, augment=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True, collate_fn=collate_fn, worker_init_fn=seed_worker)

    model = DDP(build_model(pretrained=True).to(device), device_ids=[local_rank], broadcast_buffers=False)
    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.005, momentum=0.9, weight_decay=1e-4)
    milestones = sorted(set([max(1, round(args.epochs * 0.70)), max(2, round(args.epochs * 0.90))]))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")
    history, best_dice = [], -1.0

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        totals = torch.zeros(2, device=device)
        started = time.perf_counter()

        progress = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}", disable=rank != 0, leave=False)
        for images, targets, _ in progress:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = [{key: value.to(device, non_blocking=True) for key, value in target.items()} for target in targets]
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                losses = model(images, targets)
                loss = sum(losses.values())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            totals[0] += loss.detach()
            totals[1] += 1

        dist.all_reduce(totals)
        train_loss = (totals[0] / totals[1]).item()
        metrics = {}

        if rank == 0 and args.mode == "validation":
            metrics = evaluate(model.module, val_loader, device)

        dist.barrier(device_ids=[local_rank])
        current_lr = optimizer.param_groups[0]["lr"]

        if rank == 0:
            row = {"epoch": epoch, "train_loss": train_loss, **metrics, "lr": current_lr, "minutes": (time.perf_counter() - started) / 60}
            history.append(row)

            if args.mode == "validation" and metrics["mean_dice"] > best_dice:
                best_dice = metrics["mean_dice"]
                torch.save({"epoch": epoch, "model": model.module.state_dict()}, WORK_DIR / "best_detector.pt")

            metric_text = f" | Dice {metrics['mean_dice']:.4f} | Union {metrics['union_dice']:.4f} | F1 {metrics['f1_50']:.4f} | Count {metrics['count_error']:.2f}" if metrics else ""
            print(f"Epoch {epoch:02d}/{args.epochs} | Loss {train_loss:.4f}{metric_text} | LR {current_lr:.6f} | {row['minutes']:.1f} min", flush=True)

        scheduler.step()
        dist.barrier(device_ids=[local_rank])

    if rank == 0:
        checkpoint_name = "last_detector.pt" if args.mode == "validation" else "final_detector.pt"
        torch.save({"epoch": args.epochs, "model": model.module.state_dict()}, WORK_DIR / checkpoint_name)
        pd.DataFrame(history).to_csv(WORK_DIR / f"{args.mode}_detector_history.csv", index=False)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()