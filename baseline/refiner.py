import cv2, numpy as np, pandas as pd, torch
from tqdm.auto import tqdm

def square_bounds(mask, context=1.8, minimum=96):
    height, width = mask.shape
    ys, xs = np.where(mask)
    center_x, center_y = (xs.min() + xs.max() + 1) / 2, (ys.min() + ys.max() + 1) / 2
    side = int(np.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    x1, y1 = x0 + side, y0 + side

    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > width:
        x0 -= x1 - width; x1 = width
    if y1 > height:
        y0 -= y1 - height; y1 = height

    return int(x0), int(y0), int(x1), int(y1)

def build_crop_cache(rows, prefix):
    total = sum(len(coco.imgToAnns[str(record_id)]) for record_id in rows["id"])
    image_path, mask_path = WORK_DIR / f"{prefix}_crop_images.npy", WORK_DIR / f"{prefix}_crop_masks.npy"
    images = np.lib.format.open_memmap(image_path, mode="w+", dtype=np.uint8, shape=(total, 256, 256))
    masks = np.lib.format.open_memmap(mask_path, mode="w+", dtype=np.uint8, shape=(total, 256, 256))
    position = 0

    for row in tqdm(rows.itertuples(index=False), total=len(rows), desc=f"Caching {prefix} crops"):
        image = cv2.imread(str(TRAIN_DIR / row.file_name), cv2.IMREAD_GRAYSCALE)

        for annotation in coco.imgToAnns[str(row.id)]:
            mask = coco.annToMask(annotation).astype(np.uint8)
            x0, y0, x1, y1 = square_bounds(mask)
            images[position] = cv2.resize(image[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_LINEAR)
            masks[position] = cv2.resize(mask[y0:y1, x0:x1], (256, 256), interpolation=cv2.INTER_NEAREST)
            position += 1

    images.flush(); masks.flush()
    return image_path, mask_path, total


import random
from torch import nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision.models import resnet18, ResNet18_Weights

class CropDataset(Dataset):
    def __init__(self, image_path, mask_path, augment=False):
        self.images, self.masks = np.load(image_path, mmap_mode="r"), np.load(mask_path, mmap_mode="r")
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = torch.from_numpy(np.asarray(self.images[index]).copy()).float() / 255
        mask = torch.from_numpy(np.asarray(self.masks[index]).copy()).float().unsqueeze(0)

        if self.augment:
            rotations = random.randrange(4)
            image, mask = torch.rot90(image, rotations, (0, 1)), torch.rot90(mask, rotations, (1, 2))

            if random.random() < 0.5:
                image, mask = torch.flip(image, (1,)), torch.flip(mask, (2,))

            if random.random() < 0.5:
                image = image ** random.uniform(0.85, 1.20)

            if random.random() < 0.5:
                image = torch.clamp(image * random.uniform(0.90, 1.10), 0, 1)

        return image.unsqueeze(0).repeat(3, 1, 1), mask

class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True), nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

class CropUNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        encoder = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
        self.pool, self.layer1, self.layer2, self.layer3, self.layer4 = encoder.maxpool, encoder.layer1, encoder.layer2, encoder.layer3, encoder.layer4
        self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
        self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
        self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
        self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(96, 32)
        self.up0, self.head = nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        image = (image - self.mean) / self.std
        x0 = self.stem(image)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.dec4(torch.cat([self.up4(x4), x3], 1))
        x = self.dec3(torch.cat([self.up3(x), x2], 1))
        x = self.dec2(torch.cat([self.up2(x), x1], 1))
        x = self.dec1(torch.cat([self.up1(x), x0], 1))
        return self.head(self.up0(x))

def soft_dice_loss(logits, targets):
    probabilities = logits.sigmoid()
    intersection = (probabilities * targets).sum((1, 2, 3))
    denominator = probabilities.sum((1, 2, 3)) + targets.sum((1, 2, 3))
    return (1 - (2 * intersection + 1) / (denominator + 1)).mean()

def train_refiner(model, train_loader, epochs, output_path, val_loader=None):
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")
    history, best_dice = [], -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = bce_sum = dice_sum = samples = 0

        for images, masks in tqdm(train_loader, desc=f"Refiner epoch {epoch:02d}/{epochs}", leave=False):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                logits = model(images)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, masks)
                dice_loss = soft_dice_loss(logits, masks)
                loss = 0.5 * bce + 0.5 * dice_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch = len(images)
            loss_sum += loss.item() * batch
            bce_sum += bce.item() * batch
            dice_sum += dice_loss.item() * batch
            samples += batch

        val_dice = np.nan
        if val_loader is not None:
            model.eval()
            dice_total = val_samples = 0

            with torch.inference_mode():
                for images, masks in tqdm(val_loader, desc="Refiner validation", leave=False):
                    masks = masks.to(device, non_blocking=True).bool()
                    predictions = model(images.to(device, non_blocking=True)).sigmoid() >= 0.65
                    intersection = (predictions & masks).sum((1, 2, 3)).float()
                    denominator = predictions.sum((1, 2, 3)) + masks.sum((1, 2, 3))
                    dice_total += ((2 * intersection + 1e-7) / (denominator + 1e-7)).sum().item()
                    val_samples += len(images)

            val_dice = dice_total / val_samples

            if val_dice > best_dice:
                best_dice = val_dice
                torch.save({"epoch": epoch, "model": model.state_dict()}, output_path)

        scheduler.step()
        row = {"epoch": epoch, "loss": loss_sum / samples, "bce": bce_sum / samples, "dice_loss": dice_sum / samples, "val_dice": val_dice}
        history.append(row)
        metric_text = f" | Val Dice {val_dice:.4f}" if val_loader is not None else ""
        print(f"Epoch {epoch:02d}/{epochs} | Loss {row['loss']:.4f} | BCE {row['bce']:.4f} | Dice loss {row['dice_loss']:.4f}{metric_text}")

    if val_loader is None:
        torch.save({"epoch": epochs, "model": model.state_dict()}, output_path)

    return pd.DataFrame(history)

