#!/usr/bin/env python3
"""Stratified image-level split of 190 original images, then patch to 256x256.

Overwrites OrganoIDNetData-256 with a properly stratified dataset.

Sources:
  - 180 annotations from Annotations.zip
  - 10 test images from Dataset/Test/

Split: stratified by species+donor, systematic by timepoint (70/15/15).
"""

import os

import zipfile
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.io import savemat
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

ANNOTATIONS_ZIP = "/Users/ajinkyakulkarni/Desktop/OrganoIDNetData/Raw_Data/Annotations.zip"
TEST_IMG_DIR = "/Users/ajinkyakulkarni/Desktop/OrganoIDNetData/Dataset/Test/Images"
TEST_MASK_DIR = "/Users/ajinkyakulkarni/Desktop/OrganoIDNetData/Dataset/Test/Masks"
OUTPUT_DIR = "/Users/ajinkyakulkarni/Desktop/OrganoIDNetData-256"

PATCH_SIZE = 256
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42


def parse_image_name(name):
    name = name.replace("Test_", "")
    parts = name.split("_")
    return parts[0], parts[1], parts[2]


def patch_starts(dim_size):
    if dim_size <= PATCH_SIZE:
        return [0]
    starts = list(range(0, dim_size - PATCH_SIZE + 1, PATCH_SIZE))
    if starts[-1] + PATCH_SIZE < dim_size:
        starts.append(dim_size - PATCH_SIZE)
    return starts


def clean_mask(mask):
    mask = mask.copy()
    mask[mask < 0] = 0
    mask = relabel_sequential(mask)[0]
    return mask.astype(np.int32)


def split_image_list(images):
    from collections import defaultdict

    groups = defaultdict(list)
    for img_name, img_path, mask_path in images:
        s, d, t = parse_image_name(img_name)
        time_val = int(t.rstrip("h"))
        groups[f"{s}_{d}"].append((time_val, img_name, img_path, mask_path))

    rng = np.random.default_rng(SEED)
    train, val, test = [], [], []

    for key in sorted(groups):
        entries = sorted(groups[key], key=lambda x: x[0])
        n = len(entries)
        n_test = max(1, round(n * TEST_RATIO))
        n_val = max(1, round(n * VAL_RATIO))
        n_train = n - n_val - n_test

        if n_train <= 0:
            n_val = max(1, int(n * VAL_RATIO))
            n_test = max(1, int(n * TEST_RATIO))
            n_train = n - n_val - n_test
            if n_train <= 0:
                n_val = 1
                n_test = 1
                n_train = n - 2

        indices = np.array(list(range(n)))
        rng.shuffle(indices)

        test_idx = set(indices[:n_test].tolist())
        val_idx = set(indices[n_test : n_test + n_val].tolist())

        for i in range(n):
            if i in test_idx:
                test.append(entries[i])
            elif i in val_idx:
                val.append(entries[i])
            else:
                train.append(entries[i])

    return train, val, test


def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing existing {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)

    print("Extracting Annotations.zip...")
    with zipfile.ZipFile(ANNOTATIONS_ZIP) as z:
        z.extractall(OUTPUT_DIR)
    annot_dir = os.path.join(OUTPUT_DIR, "Annotations")

    images = []

    for f in sorted(os.listdir(os.path.join(annot_dir, "Images"))):
        stem = Path(f).stem
        img_path = os.path.join(annot_dir, "Images", f)
        mask_path = os.path.join(annot_dir, "Masks", f.replace(".tif", "_mask.tif"))
        if os.path.exists(mask_path):
            images.append((stem, img_path, mask_path))

    for f in sorted(os.listdir(TEST_IMG_DIR)):
        stem = Path(f).stem
        mask_name = f.replace(".tif", "_mask.tif")
        img_path = os.path.join(TEST_IMG_DIR, f)
        mask_path = os.path.join(TEST_MASK_DIR, mask_name)
        if os.path.exists(mask_path):
            images.append((stem, img_path, mask_path))

    print(f"Total images collected: {len(images)}")

    train_imgs, val_imgs, test_imgs = split_image_list(images)
    print(f"Split: Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}")

    for split_name, split_imgs in [
        ("Training", train_imgs),
        ("Validation", val_imgs),
        ("Test", test_imgs),
    ]:
        out_img_dir = os.path.join(OUTPUT_DIR, split_name, "images")
        out_label_dir = os.path.join(OUTPUT_DIR, split_name, "labels")
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_label_dir, exist_ok=True)

        total_patches = 0
        skipped_global = 0

        for _, stem, img_path, mask_path in tqdm(
            split_imgs, desc=f"Patching {split_name}"
        ):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            H, W = mask.shape

            x_starts = patch_starts(W)
            y_starts = patch_starts(H)

            n_skipped = 0

            for y in y_starts:
                for x in x_starts:
                    img_patch = img[y : y + PATCH_SIZE, x : x + PATCH_SIZE]
                    mask_patch = mask[y : y + PATCH_SIZE, x : x + PATCH_SIZE]

                    if img_patch.shape[0] != PATCH_SIZE or img_patch.shape[1] != PATCH_SIZE:
                        continue

                    mask_patch = clean_mask(mask_patch)

                    if mask_patch.max() == 0:
                        n_skipped += 1
                        continue

                    type_map = (mask_patch > 0).astype(np.int32)

                    patch_name = f"{stem}_x{x}_y{y}"

                    cv2.imwrite(
                        os.path.join(out_img_dir, f"{patch_name}.png"),
                        cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR),
                    )

                    savemat(
                        os.path.join(out_label_dir, f"{patch_name}.mat"),
                        {"inst": mask_patch, "type": type_map},
                    )

                    total_patches += 1

            skipped_global += n_skipped

        print(
            f"  {split_name}: {len(os.listdir(out_img_dir))} patches, {skipped_global} skipped (empty)"
        )

    shutil.rmtree(annot_dir)
    print(f"\nDone. Dataset saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
