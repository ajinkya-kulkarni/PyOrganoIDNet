import numpy as np
from skimage import measure
from skimage.segmentation import relabel_sequential


PATCH_SIZE = 256
OVERLAP = 64
MAX_DIM = 2000


def _patch_starts(dim_size, patch_size, overlap):
    stride = patch_size - overlap
    if dim_size <= patch_size:
        return [0]
    starts = list(range(0, dim_size - patch_size + 1, stride))
    if starts[-1] + patch_size < dim_size:
        starts.append(dim_size - patch_size)
    return starts


def _remove_border_instances(instances, tile_x, tile_y, image_w, image_h):
    for prop in measure.regionprops(instances):
        r = prop.slice
        label = prop.label
        remove = False
        if tile_y != 0 and r[0].start == 0:
            remove = True
        elif tile_y + r[0].stop != image_h and r[0].stop == instances.shape[0]:
            remove = True
        elif tile_x != 0 and r[1].start == 0:
            remove = True
        elif tile_x + r[1].stop != image_w and r[1].stop == instances.shape[1]:
            remove = True
        if remove:
            instances[instances == label] = 0


def _instance_priority(region):
    cy, cx = region.centroid_local
    tile_center = PATCH_SIZE / 2.0
    max_dist = tile_center * 2**0.5
    dist = ((cx - tile_center) ** 2 + (cy - tile_center) ** 2) ** 0.5
    return 1.0 - dist / max_dist


def _collect_regions(cleaned_instances, tile_x, tile_y, candidates):
    for region in measure.regionprops(cleaned_instances):
        if region.area == 0:
            continue
        r = region.slice
        label = region.label
        local_mask = (cleaned_instances[r[0], r[1]] == label).astype(bool)
        priority = _instance_priority(region)
        candidates.append(
            {
                "local_mask": local_mask,
                "y_start": tile_y + r[0].start,
                "x_start": tile_x + r[1].start,
                "priority": priority,
                "area": region.area,
            }
        )


def predict_large_image(
    predict_patch_fn,
    img_rgb,
    patch_size=PATCH_SIZE,
    overlap=OVERLAP,
    iou_thresh=0.5,
    max_dim=MAX_DIM,
    progress_callback=None,
):
    H, W = img_rgb.shape[:2]

    if H < patch_size or W < patch_size:
        raise ValueError(
            f"Image must be at least {patch_size}x{patch_size} pixels, got {W}x{H}"
        )
    if H > max_dim or W > max_dim:
        raise ValueError(f"Image too large (max {max_dim}px per side), got {W}x{H}")

    x_starts = _patch_starts(W, patch_size, overlap)
    y_starts = _patch_starts(H, patch_size, overlap)

    total_tiles = len(x_starts) * len(y_starts)
    tiles_done = 0

    candidates = []

    for y in y_starts:
        for x in x_starts:
            tile = img_rgb[y : y + patch_size, x : x + patch_size]
            instances = predict_patch_fn(tile)

            _remove_border_instances(instances, x, y, W, H)

            _collect_regions(instances, x, y, candidates)

            tiles_done += 1
            if progress_callback is not None:
                progress_callback(tiles_done, total_tiles)

    candidates.sort(key=lambda c: c["priority"], reverse=True)

    canvas = np.zeros((H, W), dtype=np.int32)
    next_id = 1

    for candidate in candidates:
        y0, x0 = candidate["y_start"], candidate["x_start"]
        local_mask = candidate["local_mask"]
        h, w = local_mask.shape
        slice_ = (slice(y0, y0 + h), slice(x0, x0 + w))
        overlap_pixels = (canvas[slice_] > 0) & local_mask
        ratio = (
            overlap_pixels.sum() / candidate["area"] if candidate["area"] > 0 else 1.0
        )

        if ratio <= iou_thresh:
            canvas[slice_][local_mask] = next_id
            next_id += 1

    return relabel_sequential(canvas)[0]
