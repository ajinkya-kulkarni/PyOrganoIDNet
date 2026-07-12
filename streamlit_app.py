import io

import numpy as np
import streamlit as st
import torch
import albumentations as A
from PIL import Image
from cellseg_models_pytorch.models.cellpose.cellpose_unet import cellpose_nuclei
from cellseg_models_pytorch.postproc.functional.cellpose.cellpose import (
    post_proc_cellpose,
)
from cellseg_models_pytorch.transforms.albu_transforms import MinMaxNormalization

CKPT = "models/best.pt"
DEVICE = "cpu"
TRANSFORM = A.Compose([MinMaxNormalization(always_apply=True)])


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


def color_instances(inst):
    ids = np.unique(inst)
    ids = ids[ids > 0]
    if len(ids) == 0:
        return np.zeros((*inst.shape, 3), dtype=np.uint8)
    np.random.seed(42)
    colors = np.random.randint(50, 255, (int(ids.max()) + 1, 3), dtype=np.uint8)
    colors[0] = 0
    return colors[inst]


st.set_page_config(page_title="OrganoIDNet", layout="centered")
st.title("OrganoIDNet")

model = load_model()

uploaded = st.file_uploader("Upload a 256x256 organoid image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = np.array(Image.open(io.BytesIO(uploaded.read())).convert("RGB"))
    if img.shape[:2] != (256, 256):
        st.warning(f"Expected 256x256, got {img.shape[1]}x{img.shape[0]}. Resizing.")
        img = np.array(Image.fromarray(img).resize((256, 256), Image.LANCZOS))

    instances, fg_prob = predict(model, img)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img, caption="Input", width="stretch")
    with col2:
        st.image(color_instances(instances), caption="Predicted instances", width="stretch")
    with col3:
        st.image(fg_prob, caption="Foreground probability", clamp=True, width="stretch")

    n_inst = len(np.unique(instances)) - 1
    st.metric("Organoids detected", n_inst)
