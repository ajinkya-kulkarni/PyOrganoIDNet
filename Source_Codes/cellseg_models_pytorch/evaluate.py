#!/usr/bin/env python3
"""Evaluate best.pt on all test patches, print metrics, save CSV, generate 5 viz plots."""

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import albumentations as A
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei
from cellseg_models_pytorch.postproc.functional.cellpose.cellpose import (
    post_proc_cellpose,
)
from cellseg_models_pytorch.transforms.albu_transforms import MinMaxNormalization

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT = Path(__file__).parent.parent.parent / "models" / "best.pt"
DATA_ROOT = Path("/Users/ajinkyakulkarni/Desktop/OrganoIDNetData-256")
OUT = Path(__file__).parent / "output"
TRANSFORM = A.Compose([MinMaxNormalization(always_apply=True)])
IOU_THRESH = 0.5
N_VIZ = 5


def load_model():
    model = cellpose_nuclei(n_nuc_classes=2, enc_name="efficientnet_b0", enc_pretrain=False)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    model = model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def predict(model, img_rgb):
    img_norm = TRANSFORM(image=img_rgb)["image"]
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    out = model(tensor)
    type_prob = torch.softmax(out["nuc"].type_map, dim=1)
    fg_prob = type_prob[0, 1].cpu().numpy()
    flow = out["nuc"].aux_map[0].cpu().numpy()
    instances = post_proc_cellpose(fg_prob > 0.5, flow, min_size=30)
    return instances, fg_prob


def dice_iou(pred_mask, gt_mask):
    inter = (pred_mask & gt_mask).sum()
    dice = 2 * inter / (pred_mask.sum() + gt_mask.sum() + 1e-8)
    iou = inter / ((pred_mask | gt_mask).sum() + 1e-8)
    return dice.item(), iou.item()


def match_instances(pred_inst, gt_inst):
    pred_ids = np.unique(pred_inst)
    pred_ids = pred_ids[pred_ids > 0]
    gt_ids = np.unique(gt_inst)
    gt_ids = gt_ids[gt_ids > 0]
    if len(pred_ids) == 0 and len(gt_ids) == 0:
        return 0, 0, 0, []
    if len(pred_ids) == 0:
        return 0, 0, len(gt_ids), []
    if len(gt_ids) == 0:
        return 0, len(pred_ids), 0, []

    cost = np.zeros((len(gt_ids), len(pred_ids)))
    for i, gid in enumerate(gt_ids):
        gmask = gt_inst == gid
        for j, pid in enumerate(pred_ids):
            pmask = pred_inst == pid
            inter = (gmask & pmask).sum()
            union = (gmask | pmask).sum()
            cost[i, j] = 1 - inter / (union + 1e-8)

    gt_idx, pred_idx = linear_sum_assignment(cost)
    tp = 0
    pairs = []
    for i, j in zip(gt_idx, pred_idx):
        iou = 1 - cost[i, j]
        if iou >= IOU_THRESH:
            tp += 1
            pairs.append((gt_ids[i], pred_ids[j], iou))
    fp = len(pred_ids) - tp
    fn = len(gt_ids) - tp
    return tp, fp, fn, pairs


def visualize(img_rgb, gt_inst, pred_inst, fg_prob, save_path, title=""):
    spec = plt.colormaps["nipy_spectral"]
    spec.set_bad(color="black")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")

    if gt_inst is not None:
        gt_n = len(np.unique(gt_inst)) - 1
        gt_viz = gt_inst.astype(float)
        if gt_viz.max() > 0:
            gt_viz = 0.08 + 0.75 * gt_viz / gt_viz.max()
        gt_viz[gt_inst == 0] = np.nan
        axes[1].imshow(gt_viz, cmap=spec, vmin=0, vmax=1)
        axes[1].set_title(f"GT ({gt_n} organoids)")
    else:
        axes[1].set_title("No GT")
    axes[1].axis("off")

    pred_n = len(np.unique(pred_inst)) - 1
    pred_viz = pred_inst.astype(float)
    if pred_viz.max() > 0:
        pred_viz = 0.08 + 0.75 * pred_viz / pred_viz.max()
    pred_viz[pred_inst == 0] = np.nan
    axes[2].imshow(pred_viz, cmap=spec, vmin=0, vmax=1)
    axes[2].set_title(f"Pred ({pred_n} organoids)")
    axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path.name}")


def print_section(name, n, dice_arr, iou_arr, ce_arr, gt_arr, pred_arr, tp, fp, fn):
    if n == 0:
        return
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    f = 2 * p * r / (p + r + 1e-8)
    print(f"\n{'='*55}")
    print(f"  {name} ({n} patches)")
    print(f"{'='*55}")
    print("  Segmentation:")
    print(f"    Dice:  {dice_arr.mean():.4f} ± {dice_arr.std():.4f}  (median {np.median(dice_arr):.4f})")
    print(f"    IoU:   {iou_arr.mean():.4f} ± {iou_arr.std():.4f}  (median {np.median(iou_arr):.4f})")
    print(f"    Dice < 0.1: {(dice_arr < 0.1).sum()} / {n}")
    print("  Detection:")
    print(f"    GT count:   {gt_arr.mean():.1f} ± {gt_arr.std():.1f}")
    print(f"    Pred count: {pred_arr.mean():.1f} ± {pred_arr.std():.1f}")
    print(f"    Bias:       {pred_arr.mean() - gt_arr.mean():.1f}")
    print(f"    |Error|:    {ce_arr.mean():.2f} ± {ce_arr.std():.2f}  (median {np.median(ce_arr):.1f})")
    print(f"  Instance matching @ IoU ≥ {IOU_THRESH}:")
    print(f"    Precision:  {p:.4f}")
    print(f"    Recall:     {r:.4f}")
    print(f"    F1:         {f:.4f}")


def main():
    OUT.mkdir(exist_ok=True)

    if not CKPT.exists():
        print(f"Checkpoint not found: {CKPT}")
        print("Run train.py first.")
        return

    model = load_model()
    print(f"Loaded: {CKPT}")

    img_dir = DATA_ROOT / "Test" / "images"
    label_dir = DATA_ROOT / "Test" / "labels"
    img_paths = sorted(img_dir.glob("*.png"))

    rows = []
    dice_list, iou_list = [], []
    ce_list, gt_n_list, pred_n_list = [], [], []
    tp_total, fp_total, fn_total = 0, 0, 0
    tp_sp, fp_sp, fn_sp = {"Human": 0, "Mouse": 0}, {"Human": 0, "Mouse": 0}, {"Human": 0, "Mouse": 0}

    random.seed(42)
    viz_paths = set(random.sample(img_paths, min(N_VIZ, len(img_paths))))

    for img_path in tqdm(img_paths, desc="Evaluating"):
        stem = img_path.stem
        species = "Human" if "Human" in stem else "Mouse"

        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mat = loadmat(str(label_dir / img_path.with_suffix(".mat").name))
        gt_inst = mat["inst"]

        pred_inst, fg_prob = predict(model, img_rgb)

        if img_path in viz_paths:
            save_path = OUT / f"pred_{stem}.png"
            visualize(img_rgb, gt_inst, pred_inst, fg_prob, save_path, title=stem)

        gt_bin = gt_inst > 0
        pred_bin = pred_inst > 0
        d, i = dice_iou(pred_bin, gt_bin)
        dice_list.append(d)
        iou_list.append(i)

        gt_n = len(np.unique(gt_inst)) - 1
        pred_n = len(np.unique(pred_inst)) - 1
        ce_list.append(abs(pred_n - gt_n))
        gt_n_list.append(gt_n)
        pred_n_list.append(pred_n)

        tp, fp, fn, _ = match_instances(pred_inst, gt_inst)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        tp_sp[species] += tp
        fp_sp[species] += fp
        fn_sp[species] += fn

        rows.append({
            "patch": stem, "species": species,
            "dice": round(d, 4), "iou": round(i, 4),
            "gt_count": gt_n, "pred_count": pred_n, "count_error": abs(pred_n - gt_n),
            "tp": tp, "fp": fp, "fn": fn,
        })

    csv_path = OUT / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    dice_arr = np.array(dice_list)
    iou_arr = np.array(iou_list)
    ce_arr = np.array(ce_list)
    gt_arr = np.array(gt_n_list)
    pred_arr = np.array(pred_n_list)

    print_section("Overall", len(rows), dice_arr, iou_arr, ce_arr, gt_arr, pred_arr,
                  tp_total, fp_total, fn_total)
    for sp in ["Human", "Mouse"]:
        mask = np.array([r["species"] == sp for r in rows])
        print_section(sp, mask.sum(),
                      dice_arr[mask], iou_arr[mask], ce_arr[mask],
                      gt_arr[mask], pred_arr[mask],
                      tp_sp[sp], fp_sp[sp], fn_sp[sp])

    print(f"\nMetrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
