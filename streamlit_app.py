import io

import albumentations as A
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
    return instances


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
        ys, xs = np.where(instances == inst_id)
        if len(ys) == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        overlay[y1:y2 + 1, x1] = [0, 255, 0]
        overlay[y1:y2 + 1, x2] = [0, 255, 0]
        overlay[y1, x1:x2 + 1] = [0, 255, 0]
        overlay[y2, x1:x2 + 1] = [0, 255, 0]
    for inst_id in dead_ids:
        ys, xs = np.where(instances == inst_id)
        if len(ys) == 0:
            continue
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        overlay[y1:y2 + 1, x1] = [255, 0, 0]
        overlay[y1:y2 + 1, x2] = [255, 0, 0]
        overlay[y1, x1:x2 + 1] = [255, 0, 0]
        overlay[y2, x1:x2 + 1] = [255, 0, 0]
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
    areas = np.asarray(df["area"], dtype=float)
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
    cols = ["area", "eccentricity", "jaggedness", "compactness"]
    titles = {
        "area": "Area",
        "eccentricity": "Eccentricity",
        "jaggedness": "Jaggedness",
        "compactness": "Compactness",
    }
    xlabels = {
        "area": "Area (px\u00b2)",
        "eccentricity": "Eccentricity (a.u.)",
        "jaggedness": "Jaggedness (px\u207b\u00b9)",
        "compactness": "Compactness (px)",
    }

    figs = {}
    for status in ("Live", "Dead"):
        sub = df[df["Status"] == status]
        n = len(sub)
        if n == 0:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))
        for ax, col in zip(axes, cols):
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
                leg = ax.legend(fontsize=8, framealpha=0.9, edgecolor="#b0b0b0")
                leg.get_frame().set_linewidth(0.5)
            ax.set_title(titles[col], fontsize=12, pad=6)
            ax.set_xlabel(xlabels[col], fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.tick_params(labelsize=8)
            sns.despine(ax=ax, top=True, right=True)
        fig.suptitle(f"{status} organoids", fontsize=14, y=1.02)
        figs[status] = fig
    return figs


st.set_page_config(page_title="OrganoIDNet", layout="wide")
st.title("OrganoIDNet")

model = load_model()


def load_image(f):
    img = np.array(Image.open(io.BytesIO(f.read())).convert("RGB"))
    if img.shape[:2] != (256, 256):
        img = np.array(Image.fromarray(img).resize((256, 256), Image.Resampling.LANCZOS))
    return img


uploaded_files = st.file_uploader(
    "Upload 256\u00d7256 organoid images (up to 20)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.stop()

n_files = len(uploaded_files)
if n_files > 20:
    st.error("Maximum 20 images allowed.")
    st.stop()

# ── Single image: full detail view ──────────────────────────────────────
if n_files == 1:
    f = uploaded_files[0]
    img = load_image(f)
    instances = predict(model, img)
    gray = np.mean(img, axis=2)
    live_ids, dead_ids = classify_organoids(instances, gray, INTENSITY_THRESHOLD)
    stats_df = compute_stats(instances, img)
    total = len(stats_df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img, caption="Input", width="stretch")
    with col2:
        st.image(render_instance_mask(instances), caption="Instance mask", width="stretch")
    with col3:
        st.image(draw_classified_outlines(img, instances, live_ids, dead_ids),
                 caption="Overlay", width="stretch")

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
        figs = plot_morphology(stats_df)
        for fig in figs.values():
            st.pyplot(fig)

        st.subheader("Per-organoid details")
        display = stats_df[["label", "area", "eccentricity", "mean_intensity", "Size", "Status"]]
        display.columns = ["ID", "Area (px\u00b2)", "Eccentricity", "Mean intensity", "Size", "Status"]
        st.dataframe(display, width="stretch", hide_index=True)

# ── Multiple images: aggregate view ─────────────────────────────────────
else:
    # Cache results per batch of file names
    file_key = tuple(f.name for f in uploaded_files)
    if "batch_key" not in st.session_state or st.session_state["batch_key"] != file_key:
        st.session_state["batch_key"] = file_key
        st.session_state["batch_results"] = []
        progress = st.progress(0, text="Segmenting organoids...")
        for i, f in enumerate(uploaded_files):
            img = load_image(f)
            instances = predict(model, img)
            stats_df = compute_stats(instances, img)
            st.session_state["batch_results"].append((f.name, stats_df))
            progress.progress((i + 1) / n_files, text=f"Segmented {i + 1}/{n_files}")
        progress.empty()

    results = st.session_state["batch_results"]
    all_dfs = [df for _, df in results if len(df) > 0]

    if not all_dfs:
        st.warning("No organoids detected in any of the uploaded images.")
    else:
        combined = pd.concat(all_dfs, ignore_index=True)

        total_organoids = len(combined)
        n_live = int((combined["Status"] == "Live").sum())
        n_dead = int((combined["Status"] == "Dead").sum())
        viability = n_live / total_organoids * 100
        mean_area = combined["area"].mean()
        mean_ecc = combined["eccentricity"].mean()

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Total organoids", total_organoids)
        m2.metric("Live", n_live)
        m3.metric("Dead", n_dead)
        m4.metric("Viability", f"{viability:.1f}%")
        m5.metric("Mean area", f"{mean_area:.0f} px\u00b2")
        m6.metric("Mean eccentricity", f"{mean_ecc:.2f}")

        st.subheader("Per-image summary")
        summary_rows = []
        for name, df in results:
            t = len(df)
            lv = int((df["Status"] == "Live").sum()) if t > 0 else 0
            dd = int((df["Status"] == "Dead").sum()) if t > 0 else 0
            v = lv / t * 100 if t > 0 else 0
            summary_rows.append({
                "Image": name,
                "Organoids": t,
                "Live": lv,
                "Dead": dd,
                "Viability": f"{v:.1f}%",
            })
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

        st.subheader("Size distribution (aggregate)")
        order = ["Tiny", "Small", "Medium", "Large", "Huge"]
        sz_rows = []
        for sz in order:
            sub = combined[combined["Size"] == sz]
            if len(sub):
                sz_rows.append({
                    "Size": sz,
                    "Total": len(sub),
                    "Live": int((sub["Status"] == "Live").sum()),
                    "Dead": int((sub["Status"] == "Dead").sum()),
                })
        st.dataframe(pd.DataFrame(sz_rows), width="stretch", hide_index=True)

        st.subheader("Morphology distributions (aggregate)")
        figs = plot_morphology(combined)
        for fig in figs.values():
            st.pyplot(fig)

        csv = combined.to_csv(index=False).encode()
        st.download_button("Download all data as CSV", csv, "organoid_data.csv", "text/csv")
