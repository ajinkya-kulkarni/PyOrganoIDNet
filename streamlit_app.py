import io

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei
from cellseg_models_pytorch.postproc.functional.cellpose.cellpose import (
    post_proc_cellpose,
)
from cellseg_models_pytorch.transforms.albu_transforms import MinMaxNormalization
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries

CKPT = "models/best.pt"
DEVICE = "cpu"
TRANSFORM = A.Compose([MinMaxNormalization(always_apply=True)])
INTENSITY_THRESHOLD = 50


@st.cache_resource
def load_model():
    model = cellpose_nuclei(n_nuc_classes=2, enc_name="efficientnet_b0", enc_pretrain=False)
    model.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])
    model.to(DEVICE)
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


def classify_organoids(instances, gray_img, threshold=50):
    live, dead = [], []
    for inst_id in np.unique(instances):
        if inst_id == 0:
            continue
        mask = instances == inst_id
        if gray_img[mask].mean() >= threshold:
            live.append(inst_id)
        else:
            dead.append(inst_id)
    return live, dead


def render_instance_mask(inst):
    spec = plt.colormaps["nipy_spectral"]
    spec.set_bad(color="black")
    viz = inst.astype(float)
    if viz.max() > 0:
        viz = 0.08 + 0.75 * viz / viz.max()
    viz[inst == 0] = np.nan
    rgba = spec(np.ma.masked_invalid(viz))
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def draw_classified_outlines(img, instances, live_ids, dead_ids):
    overlay = img.copy()
    if len(np.unique(instances)) <= 1:
        return overlay
    all_bounds = find_boundaries(instances, mode="outer")
    if live_ids:
        live_bounds = find_boundaries(np.isin(instances, live_ids).astype(int), mode="outer")
        overlay[live_bounds & all_bounds] = [0, 255, 0]
    if dead_ids:
        dead_bounds = find_boundaries(np.isin(instances, dead_ids).astype(int), mode="outer")
        overlay[dead_bounds & all_bounds] = [255, 0, 0]
    return overlay


def compute_stats(instances, img):
    gray = np.mean(img, axis=2)
    props = regionprops_table(
        instances,
        intensity_image=gray,
        properties=("label", "area", "eccentricity", "mean_intensity"),
    )
    df = pd.DataFrame(props)
    if len(df) == 0:
        return df
    df["Status"] = np.where(df["mean_intensity"] >= INTENSITY_THRESHOLD, "Live", "Dead")
    areas = df["area"].values
    if len(areas) > 1:
        p20, p40, p60, p80 = np.percentile(areas, [20, 40, 60, 80])
    else:
        p20 = p40 = p60 = p80 = areas[0]
    def size_cat(a):
        if a <= p20:
            return "Tiny"
        if a <= p40:
            return "Small"
        if a <= p60:
            return "Medium"
        if a <= p80:
            return "Large"
        return "Huge"
    df["Size"] = df["area"].apply(size_cat)
    return df


st.set_page_config(page_title="OrganoIDNet", layout="wide")
st.title("OrganoIDNet")

model = load_model()

uploaded = st.file_uploader("Upload a 256\u00d7256 organoid image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = np.array(Image.open(io.BytesIO(uploaded.read())).convert("RGB"))
    if img.shape[:2] != (256, 256):
        st.warning(f"Expected 256\u00d7256, got {img.shape[1]}\u00d7{img.shape[0]}. Resizing.")
        img = np.array(Image.fromarray(img).resize((256, 256), Image.LANCZOS))

    instances, fg_prob = predict(model, img)
    gray = np.mean(img, axis=2)
    live_ids, dead_ids = classify_organoids(instances, gray, INTENSITY_THRESHOLD)
    stats_df = compute_stats(instances, img)
    total = len(stats_df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.image(img, caption="Input", width="stretch")
    with col2:
        st.image(render_instance_mask(instances), caption="Instance mask", width="stretch")
    with col3:
        st.image(draw_classified_outlines(img, instances, live_ids, dead_ids),
                 caption="Overlay", width="stretch")
    with col4:
        st.image(fg_prob, caption="Foreground prob.", clamp=True, width="stretch")

    n_live = len(live_ids)
    n_dead = len(dead_ids)
    viability = n_live / total * 100 if total > 0 else 0
    mean_area = stats_df["area"].mean() if total > 0 else 0
    mean_ecc = stats_df["eccentricity"].mean() if total > 0 else 0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total", total)
    m2.metric("Live", n_live)
    m3.metric("Dead", n_dead)
    m4.metric("Viability", f"{viability:.1f}%")
    m5.metric("Mean area", f"{mean_area:.0f} px\u00b2")
    m6.metric("Mean eccentricity", f"{mean_ecc:.2f}")

    if total > 0:
        st.subheader("Size distribution")
        order = ["Tiny", "Small", "Medium", "Large", "Huge"]
        rows = []
        for sz in order:
            sub = stats_df[stats_df["Size"] == sz]
            if len(sub):
                rows.append({
                    "Size": sz,
                    "Total": len(sub),
                    "Live": int((sub["Status"] == "Live").sum()),
                    "Dead": int((sub["Status"] == "Dead").sum()),
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.subheader("Per-organoid details")
        display = stats_df[["label", "area", "eccentricity", "mean_intensity", "Size", "Status"]]
        display.columns = ["ID", "Area (px\u00b2)", "Eccentricity", "Mean intensity", "Size", "Status"]
        st.dataframe(display, use_container_width=True, hide_index=True)
