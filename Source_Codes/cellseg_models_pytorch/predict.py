#!/usr/bin/env python3
"""Inference + viz: image, GT instances, predicted instances (no overlay)."""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A
from scipy.io import loadmat

from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei
from cellseg_models_pytorch.postproc.functional.cellpose.cellpose import (
    post_proc_cellpose,
)
from cellseg_models_pytorch.transforms.albu_transforms import MinMaxNormalization

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"


def load_model(ckpt_path, enc_name="efficientnet_b0"):
    model = cellpose_nuclei(n_nuc_classes=2, enc_name=enc_name, enc_pretrain=False)
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model"])
    model = model.to(DEVICE)
    model.eval()
    return model


PREDICT_TRANSFORM = A.Compose([MinMaxNormalization(always_apply=True)])


@torch.no_grad()
def predict(model, img_path):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_norm = PREDICT_TRANSFORM(image=img_rgb)["image"]
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    out = model(tensor)

    type_prob = torch.softmax(out["nuc"].type_map, dim=1)
    fg_prob = type_prob[0, 1].cpu().numpy()
    fg_mask = fg_prob > 0.5

    flow = out["nuc"].aux_map[0].cpu().numpy()

    instances = post_proc_cellpose(fg_mask, flow, min_size=30)

    return img_rgb, instances, fg_prob


def visualize(img_rgb, gt_inst, pred_inst, save_path):
    spec = plt.colormaps["nipy_spectral"]
    spec.set_bad(color="black")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")

    if gt_inst is not None:
        gt_viz = gt_inst.astype(float)
        if gt_viz.max() > 0:
            gt_viz = 0.08 + 0.75 * gt_viz / gt_viz.max()
        gt_viz[gt_inst == 0] = np.nan
        axes[1].imshow(gt_viz, cmap=spec, vmin=0, vmax=1)
        axes[1].set_title(f"GT ({gt_inst.max()} organoids)")
    else:
        axes[1].set_title("No GT")
    axes[1].axis("off")

    pred_viz = pred_inst.astype(float)
    if pred_viz.max() > 0:
        pred_viz = 0.08 + 0.75 * pred_viz / pred_viz.max()
    pred_viz[pred_inst == 0] = np.nan
    axes[2].imshow(pred_viz, cmap=spec, vmin=0, vmax=1)
    axes[2].set_title(f"Pred ({pred_inst.max()} organoids)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    ckpt = Path(__file__).parent / "output" / "best.pt"
    test_img_dir = Path(
        "/Users/ajinkyakulkarni/Desktop/OrganoIDNetData-256/Test/images"
    )
    test_label_dir = Path(
        "/Users/ajinkyakulkarni/Desktop/OrganoIDNetData-256/Test/labels"
    )
    out_dir = Path(__file__).parent / "output"
    os.makedirs(out_dir, exist_ok=True)

    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        print("Run train.py first.")
        return

    model = load_model(ckpt)
    print(f"Loaded model from {ckpt}")

    img_paths = sorted(test_img_dir.glob("*.png"))
    if not img_paths:
        print("No test images found.")
        return

    for img_path in img_paths[:10]:
        label_path = test_label_dir / img_path.with_suffix(".mat").name

        img_rgb, pred_inst, _ = predict(model, img_path)

        gt_inst = None
        if label_path.exists():
            gt_inst = loadmat(str(label_path))["inst"]

        save_path = out_dir / f"pred_{img_path.stem}.png"
        visualize(img_rgb, gt_inst, pred_inst, save_path)


if __name__ == "__main__":
    main()
