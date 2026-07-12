import io

import albumentations as A
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
    for inst_id in live_ids:
        mask = instances == inst_id
        bounds = find_boundaries(mask, mode="outer")
        overlay[bounds] = [0, 255, 0]
    for inst_id in dead_ids:
        mask = instances == inst_id
        bounds = find_boundaries(mask, mode="outer")
        overlay[bounds] = [255, 0, 0]
    return overlay


def compute_stats(instances, img):
    gray = np.mean(img, axis=2)
    props = regionprops_table(
        instances,
        intensity_image=gray,
        properties=("label", "area", "perimeter", "eccentricity", "mean_intensity"),
    )
    df = pd.DataFrame(props)
    if len(df) == 0:
        return df
    df["jaggedness"] = df["perimeter"] / df["area"]
    df["compactness"] = df["area"] / df["perimeter"]
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


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "Liberation Sans", "DejaVu Sans", "sans-serif"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
})
sns.set_theme(style="ticks", rc={
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linestyle": "-",
    "grid.alpha": 0.5,
})
COLORS = {"Live": "#27ae60", "Dead": "#e74c3c"}


def plot_morphology(df):
    classes = [c for c in ("Live", "Dead") if c in df["Status"].values]
    n_rows = len(classes)

    fig = plt.figure(figsize=(12, n_rows * 2.5))
    gs = gridspec.GridSpec(n_rows, 4, figure=fig, hspace=0.4, wspace=0.35)

    cols = ["area", "eccentricity", "jaggedness", "compactness"]
    titles = {
        "area": "Area",
        "eccentricity": "Eccentricity",
        "jaggedness": "Jaggedness",
        "compactness": "Compactness",
    }
    xlabels = {
        "area": r"$\textit{Area}$ (px\u00b2)",
        "eccentricity": r"$\textit{Eccentricity}$ (a.u.)",
        "jaggedness": r"$\textit{Jaggedness}$ (px\u207b\u00b9)",
        "compactness": r"$\textit{Compactness}$ (px)",
    }

    for row_idx, status in enumerate(classes):
        sub = df[df["Status"] == status]
        n = len(sub)
        for col_idx, col in enumerate(cols):
            ax = fig.add_subplot(gs[row_idx, col_idx])
            if n:
                bins = min(30, max(8, n // 5))
                sns.histplot(sub[col], bins=bins, stat="density",
                             alpha=0.3, color=COLORS[status],
                             edgecolor=COLORS[status], linewidth=0.4, ax=ax)
                if n >= 2:
                    sns.kdeplot(sub[col], color=COLORS[status], linewidth=1.8,
                                bw_adjust=0.5, ax=ax)
                mean_val = sub[col].mean()
                ax.axvline(mean_val, color=COLORS[status], linestyle="--",
                           linewidth=1.0, alpha=0.6, label=f"{status} mean")
            ax.set_title(f"{status} \u2013 {titles[col]}", fontsize=12, pad=6)
            ax.set_xlabel(xlabels[col], fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.tick_params(labelsize=8)
            if n >= 2:
                leg = ax.legend(fontsize=8, framealpha=0.9, edgecolor="#b0b0b0")
                leg.get_frame().set_linewidth(0.5)
            sns.despine(ax=ax, top=True, right=True)

    fig.suptitle("Organoid Morphology Analysis", fontsize=15, y=1.02)
    return fig


st.set_page_config(page_title="OrganoIDNet", layout="wide")
st.title("OrganoIDNet")

model = load_model()

uploaded = st.file_uploader("Upload a 256\u00d7256 organoid image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = np.array(Image.open(io.BytesIO(uploaded.read())).convert("RGB"))
    if img.shape[:2] != (256, 256):
        st.warning(f"Expected 256\u00d7256, got {img.shape[1]}\u00d7{img.shape[0]}. Resizing.")
        img = np.array(Image.fromarray(img).resize((256, 256), Image.Resampling.LANCZOS))

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
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.subheader("Morphology distributions")
        st.pyplot(plot_morphology(stats_df))

        st.subheader("Per-organoid details")
        display = stats_df[["label", "area", "eccentricity", "mean_intensity", "Size", "Status"]]
        display.columns = ["ID", "Area (px\u00b2)", "Eccentricity", "Mean intensity", "Size", "Status"]
        st.dataframe(display, width="stretch", hide_index=True)
