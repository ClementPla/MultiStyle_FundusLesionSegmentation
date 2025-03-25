from copy import deepcopy

import albumentations as A
import cv2
import numpy as np
import PIL
import streamlit as st
import torch
from fundus_data_toolkit.functional import autofit_fundus_resolution

from adptSeg.adaptation.adversarial import PGD
from adptSeg.adaptation.const import _all_datasets, map_dataset_to_integer
from adptSeg.adaptation.model_wrapper import ModelFeaturesExtractor
from adptSeg.adaptation.probe import ProbeModule
from adptSeg.adaptation.utils import get_probe_model_and_loss
from fundseg.data.data_factory import ALL_DATASETS, FundusDataset
from fundseg.data.utils import ALL_CLASSES
from fundseg.models.smp_model import SMPModel
from fundseg.utils.colors import COLORS

st.set_page_config(layout="wide", page_title="Fundus Lesions Segmentation", page_icon=":eye:")


def load_uploaded_file(uploaded_file):
    # Convert to numpy array
    img = np.array(PIL.Image.open(uploaded_file))
    img, roi, _ = preprocess_image(img)
    return img, roi


def convert_to_tensor(img, roi, device="cuda"):
    img = A.Normalize()(image=img)["image"]
    roi = torch.tensor(roi).unsqueeze(0).float().to(device)
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device), roi


def preprocess_image(img):
    return autofit_fundus_resolution(img, 1024, return_roi=True)


@st.cache_resource
def load_model(model_name, device="cuda"):
    model = SMPModel.from_pretrained(f"ClementP/MultiStyle_FundusLesionSegmentation", revision=model_name)
    model = model.to(device)
    model.eval()
    return model


@st.cache_resource
def load_probe_model(_model, device="cuda"):
    encoder = ModelFeaturesExtractor(_model, position=5, feature_type="encoder").to(device)
    model = ProbeModule.from_pretrained(
        "ClementP/MultiStyle_FundusLesionSegmentation",
        revision="probe",
        featureExtractor=deepcopy(encoder),
        weights=torch.ones(5),
        as_regression=False,
        n_classes=5,
    )
    model = model.to(device)

    model.eval()
    return model, model.criterion


@torch.no_grad()
def segment_image(model, img, roi):
    batch = dict(image=img, roi=roi)
    return model.inference_step(batch)


@st.cache_data
def segment_image_with_style_adaptation(
    _model, _probe, _loss, _img, _roi, source, target, alpha, img_name, radius=10, step_size=0.5, step_num=50
):
    pgd = PGD(forward_func=_probe, loss_func=_loss, sign_grad=True)
    target = map_dataset_to_integer(FundusDataset(target))
    source = map_dataset_to_integer(FundusDataset(source))

    source_input = pgd.perturb(
        _img,
        target=source,
        step_size=step_size,
        step_num=step_num,
        radius=radius / 255,
        targeted=True,
    )

    target_input = pgd.perturb(
        _img,
        target=target,
        step_size=step_size,
        step_num=step_num,
        radius=radius / 255,
        targeted=True,
    )
    new_img = source_input * (1 - alpha) + alpha * target_input
    new_img[:, :, ~_roi.bool().squeeze()] = new_img.min()
    batch = dict(image=new_img, roi=_roi)

    img = new_img.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    # Normalize the image to [0, 1]
    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).astype(np.uint8)
    return _model.inference_step(batch), img


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    introduction = st.expander("Introduction", icon=":material/info:")
    with introduction:
        st.write(
            "This app allows you to segment fundus images using a pre-trained model. \
            Four classes are available: Exudates, Microaneurysms, Hemorrhages, and Soft Exudates. \
            We propose different weights, depending on the training dataset used, to segment the images."
        )
        st.write(
            "We illustrate the notion of label adaptation using adversarial attacks \
            to fit different styles of segmentation. \n In addition, we provide a style conversion \
            tool to convert the segmentation style of a fundus image from one dataset to another. \
            The style conversion tool uses a probe model to adapt the style of the segmentation. \
            The model itself is never modified."
        )
    st.sidebar.title("Configuration")

    uploaded_file = st.sidebar.file_uploader(
        "Upload a fundus image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
    )

    style_conversion = st.sidebar.checkbox("Style conversion", value=False)
    if not style_conversion:
        model_name = st.sidebar.selectbox("Checkpoints", ["IDRID", "FGADR", "RETLES", "MESSIDOR", "DDR", "ALL"])

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

        st.sidebar.subheader("Conversion configuration")
        step_size = st.sidebar.slider("Step size", min_value=0.01, max_value=1.0, value=0.1, step=0.05)
        step_num = st.sidebar.slider("Step number", min_value=1, max_value=100, value=5, step=5)
        radius = st.sidebar.slider("Radius", min_value=1, max_value=55, value=5, step=5)

    if uploaded_file is not None:
        filename = uploaded_file.name
        model = load_model(model_name, device)
        img, roi = load_uploaded_file(uploaded_file)
        alpha = st.slider("Alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        col1, col3 = st.columns([1, 1])
        with col1:
            plot_image(img)

        tensor_img, tensor_roi = convert_to_tensor(img, roi, device)
        if style_conversion:
            probe, loss = load_probe_model(model, device)
            pred_map, img = segment_image_with_style_adaptation(
                model,
                probe,
                loss,
                tensor_img,
                tensor_roi,
                source,
                target,
                conversion_value,
                filename,
                radius,
                step_size,
                step_num,
            )

        else:
            pred_map = segment_image(model, tensor_img, tensor_roi)

        pred_map = pred_map.squeeze().argmax(0)
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
