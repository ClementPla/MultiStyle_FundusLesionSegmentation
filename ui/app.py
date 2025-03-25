import albumentations as A
import numpy as np
import PIL
import streamlit as st
import torch
from fundus_data_toolkit.functional import autofit_fundus_resolution
import cv2
from adptSeg.adaptation.adversarial import PGD
from adptSeg.adaptation.const import _all_datasets, map_dataset_to_integer
from adptSeg.adaptation.utils import get_probe_model_and_loss
from fundseg.data.data_factory import ALL_DATASETS, FundusDataset
from fundseg.data.utils import ALL_CLASSES
from fundseg.utils.colors import COLORS

st.set_page_config(layout="wide", page_title="Fundus Lesions Segmentation", page_icon=":eye:")
# Set dark theme by default
st.markdown(
    """
    <style>
    .reportview-container {
        background: #1a1a1a;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_uploaded_file(uploaded_file):
    # Convert to numpy array
    img = np.array(PIL.Image.open(uploaded_file))
    img, roi, _ = preprocess_image(img)
    return img, roi


def convert_to_tensor(img, roi):
    img = A.Normalize()(image=img)["image"]
    roi = torch.tensor(roi).unsqueeze(0).float().cuda()
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().cuda(), roi


def preprocess_image(img):
    return autofit_fundus_resolution(img, 1024, return_roi=True)


@st.cache_resource
def load_model(model_name):
    if model_name == "ALL":
        dataset = ALL_DATASETS
    else:
        dataset = FundusDataset(model_name)
    probe, model, loss = get_probe_model_and_loss(
        model_type=dataset,
        probe_type=4,
        n_classes=5,
        as_regression=False,
    )
    # model = load_model_from_checkpoints(train_datasets=dataset).eval()
    return model.eval(), probe, loss


@torch.no_grad()
def segment_image(model, img, roi):
    batch = dict(image=img, roi=roi)
    return model.inference_step(batch)


@st.cache_data
def segment_image_with_style_adaptation(_model, _probe, _loss, _img, _roi, source, target, alpha, img_name):
    pgd = PGD(forward_func=_probe, loss_func=_loss, sign_grad=True)
    target = map_dataset_to_integer(FundusDataset(target))
    source = map_dataset_to_integer(FundusDataset(source))

    source_input = pgd.perturb(
        _img,
        target=source,
        step_size=0.01,
        step_num=5,
        radius=5 / 255,
        targeted=True,
    )

    target_input = pgd.perturb(
        _img,
        target=target,
        step_size=0.01,
        step_num=5,
        radius=5 / 255,
        targeted=True,
    )

    batch = dict(image=source_input * (1 - alpha) + alpha * target_input, roi=_roi)

    return _model.inference_step(batch)


def plot_image(img):
    st.image(img, use_container_width=True)


def plot_mask(segmentation, fig):
    for i in range(1, 5):
        mask = segmentation == i
        hex_color = COLORS[i]
        # Convert hex to RGB
        rgb_color = tuple(int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
        mask_rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
        mask_rgba[mask] = [*rgb_color, 127]
        mask_rgba[~mask] = [0, 0, 0, 0]
        fig = plot_image(mask_rgba, fig)
        # Add a size 0 square with the color in order to create a legend label:
        fig.scatter(0, 0, marker="square", color=hex_color, size=0, legend_label=ALL_CLASSES[i - 1].name)

    return fig


def app():
    uploaded_file = st.sidebar.file_uploader(
        "Upload a fundus image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )

    style_conversion = st.sidebar.checkbox("Style conversion", value=False)
    st.sidebar.subheader("Model")
    if not style_conversion:
        st.sidebar.write("Choose a model to use for the segmentation")
        model_name = st.sidebar.selectbox("Model", ["IDRID", "FGADR", "RETLES", "MESSIDOR", "DDR", "ALL"])

    if style_conversion:
        st.sidebar.subheader("Segmentation style")
        source = st.sidebar.radio(
            "Source", ["IDRID", "FGADR", "RETLES", "MESSIDOR", "DDR"], horizontal=True, key="source"
        )
        target = st.sidebar.radio(
            "Target", ["IDRID", "FGADR", "RETLES", "MESSIDOR", "DDR"], horizontal=True, key="target"
        )
        conversion_value = st.sidebar.slider("Interpolation", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        model_name = "ALL"

    if uploaded_file is not None:
        filename = uploaded_file.name
        model, probe, loss = load_model(model_name)
        img, roi = load_uploaded_file(uploaded_file)
        alpha = st.slider("Alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        col1, col3 = st.columns([1, 1])
        with col1:
            plot_image(img)

        tensor_img, tensor_roi = convert_to_tensor(img, roi)
        if style_conversion:
            pred_map = (
                segment_image_with_style_adaptation(
                    model, probe, loss, tensor_img, tensor_roi, source, target, conversion_value, filename
                )
                .squeeze()
                .argmax(0)
            )

        else:
            pred_map = segment_image(model, tensor_img, tensor_roi).squeeze().argmax(0)

        masks = np.zeros((*pred_map.shape, 3), dtype=np.uint8)
        for i in range(1, 5):
            pred_class = (pred_map == i).cpu().numpy()
            hex_color = COLORS[i]
            # Convert hex to RGB
            rgb_color = tuple(int(hex_color[j : j + 2], 16) for j in (1, 3, 5))
            masks[pred_class] = np.asarray(rgb_color)

        img_with_masks = cv2.addWeighted(img, 1 - alpha, masks, alpha, 0)
        with col3:
            st.image(img_with_masks, use_container_width=True)


if __name__ == "__main__":
    app()
