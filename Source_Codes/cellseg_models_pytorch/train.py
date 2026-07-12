#!/usr/bin/env python3
"""Train CellPose (cellseg_models.pytorch) on OrganoIDNetData-256."""

import os
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
from torch.utils.data import Dataset, DataLoader, Subset
from scipy.io import loadmat
import matplotlib.pyplot as plt
from tqdm import tqdm

import albumentations as A
from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei
from cellseg_models_pytorch.transforms.albu_transforms import MinMaxNormalization
from cellseg_models_pytorch.transforms.functional.cellpose import gen_flow_maps

SEED = 42

DATA_ROOT = Path("/Users/ajinkyakulkarni/Desktop/OrganoIDNetData-256")
OUT_DIR = Path(__file__).parent / "output"

BATCH_SIZE = 32
LR = 1e-3
ENCODER = "efficientnet_b0"
SUB_SIZE = 3000
PATIENCE = 10
MAX_EPOCHS = 30
WARMUP_EPOCHS = 3
ETA_MIN = 1e-6
CLIP_NORM = 1.0

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


class OrganoidDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_paths = sorted(Path(img_dir).glob("*.png"))
        self.label_paths = sorted(Path(label_dir).glob("*.mat"))

        if len(self.img_paths) != len(self.label_paths):
            raise RuntimeError(
                f"Count mismatch: {len(self.img_paths)} images vs {len(self.label_paths)} labels"
            )

        for img_p, label_p in zip(self.img_paths, self.label_paths):
            if img_p.stem != label_p.stem:
                raise RuntimeError(
                    f"Mismatch: image '{img_p.name}' paired with label '{label_p.name}'"
                )

        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(str(self.img_paths[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mat = loadmat(str(self.label_paths[idx]))
        inst = mat["inst"].astype(np.int32)
        type_map = mat["type"].astype(np.int32)

        if self.transform:
            aug = self.transform(image=img, masks=[inst, type_map])
            img = aug["image"]
            inst = aug["masks"][0].astype(np.int32)
            type_map = aug["masks"][1].astype(np.int32)

        flow = gen_flow_maps(inst)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        inst_map = torch.from_numpy(inst).long()
        type_map = torch.from_numpy(type_map).long()
        flow = torch.from_numpy(flow).float()

        return {"image": img, "inst": inst_map, "type": type_map, "inst-cellpose": flow}


def dice_score(logits, target, eps=1e-8):
    pred = logits.argmax(dim=1)
    intersection = (pred & target).sum().float()
    return (2.0 * intersection / (pred.sum().float() + target.sum().float() + eps)).item()


@torch.no_grad()
def save_viz(loader, model, epoch, path):
    model.eval()
    batch = next(iter(loader))
    img = batch["image"][0].cpu()
    gt_inst = batch["inst"][0].cpu().numpy()

    out = model(img.unsqueeze(0).to(DEVICE))
    prob_fg = torch.softmax(out["nuc"].type_map, dim=1)[0, 1].cpu().numpy()

    spec = plt.colormaps["nipy_spectral"]
    spec.set_bad(color="black")

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    axes[0].imshow(img.permute(1, 2, 0).clip(0, 1))
    axes[0].set_title("Input")
    axes[0].axis("off")

    gt_viz = gt_inst.astype(float)
    if gt_viz.max() > 0:
        gt_viz = 0.08 + 0.75 * gt_viz / gt_viz.max()
    gt_viz[gt_inst == 0] = np.nan
    axes[1].imshow(gt_viz, cmap=spec, vmin=0, vmax=1)
    axes[1].set_title("GT instances")
    axes[1].axis("off")

    axes[2].imshow(prob_fg, cmap="Reds", vmin=0, vmax=1)
    axes[2].set_title(f"Pred probability (epoch {epoch})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Device: {DEVICE}")

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=0.5, sigma=10, fill_mask=0, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.GaussNoise(std_range=(0.01, 0.03), p=0.3),
        MinMaxNormalization(always_apply=True),
    ])
    val_transform = A.Compose([
        MinMaxNormalization(always_apply=True),
    ])

    train_ds = OrganoidDataset(
        DATA_ROOT / "Training" / "images",
        DATA_ROOT / "Training" / "labels",
        transform=train_transform,
    )
    val_ds = OrganoidDataset(
        DATA_ROOT / "Validation" / "images",
        DATA_ROOT / "Validation" / "labels",
        transform=val_transform,
    )

    train_ds = Subset(train_ds, range(min(SUB_SIZE, len(train_ds))))
    val_ds = Subset(val_ds, range(min(32, len(val_ds))))

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model = cellpose_nuclei(n_nuc_classes=2, enc_name=ENCODER, enc_pretrain=True)
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Compute class weights from training labels
    n_bg, n_fg = 0, 0
    for p in tqdm(sorted((DATA_ROOT / "Training" / "labels").glob("*.mat")), desc="Computing class weights", leave=False):
        m = loadmat(str(p))
        t = m["type"].ravel()
        n_bg += (t == 0).sum()
        n_fg += (t == 1).sum()
    total = n_bg + n_fg
    class_weights = torch.tensor([total / (2 * n_bg), total / (2 * n_fg)], dtype=torch.float32, device=DEVICE)
    print(f"  Class weights: bg={class_weights[0]:.4f}, fg={class_weights[1]:.4f}")

    best_dice = 0.0
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        # LR schedule: linear warmup → cosine decay
        if epoch <= WARMUP_EPOCHS:
            lr = LR * epoch / WARMUP_EPOCHS
        else:
            progress = (epoch - WARMUP_EPOCHS) / (MAX_EPOCHS - WARMUP_EPOCHS)
            lr = ETA_MIN + (LR - ETA_MIN) * 0.5 * (1 + math.cos(math.pi * progress))
        for g in optimizer.param_groups:
            g["lr"] = lr

        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            img = batch["image"].to(DEVICE)
            gt_flow = batch["inst-cellpose"].to(DEVICE)
            gt_type = batch["type"].to(DEVICE)

            out = model(img)

            loss = F.mse_loss(out["nuc"].aux_map, gt_flow) + \
                   F.cross_entropy(out["nuc"].type_map, gt_type, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(DEVICE)
                gt_flow = batch["inst-cellpose"].to(DEVICE)
                gt_type = batch["type"].to(DEVICE)

                out = model(img)
                val_loss += F.mse_loss(out["nuc"].aux_map, gt_flow).item() + \
                            F.cross_entropy(out["nuc"].type_map, gt_type, weight=class_weights).item()
                val_dice += dice_score(out["nuc"].type_map, gt_type)

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        avg_dice = val_dice / len(val_loader)

        print(f"  train: {avg_train:.4f}  val: {avg_val:.4f}  dice: {avg_dice:.4f}  lr: {lr:.2e}")

        torch.save({"model": model.state_dict(), "epoch": epoch}, OUT_DIR / "last.pt")

        if avg_dice > best_dice:
            best_dice = avg_dice
            no_improve = 0
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "dice": avg_dice},
                OUT_DIR / "best.pt",
            )
            print(f"  -> best.pt (dice={avg_dice:.4f})")
        else:
            no_improve += 1

        if epoch % 5 == 0 or epoch == 1:
            save_viz(val_loader, model, epoch, OUT_DIR / f"viz_epoch_{epoch}.png")

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE})")
            break

    print(f"\nDone. Best Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
