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


def load_uploaded_file(uploaded_file, resolution):
    # Convert to numpy array
    img = np.array(PIL.Image.open(uploaded_file))
    img, roi, _ = preprocess_image(img, resolution)
    return img, roi


def convert_to_tensor(img, roi, device="cuda"):
    img = A.Normalize()(image=img)["image"]
    roi = torch.tensor(roi).unsqueeze(0).float().to(device)
    return torch.tensor(img).permute(2, 0, 1).unsqueeze(0).float().to(device), roi


def preprocess_image(img, resolution):
    return autofit_fundus_resolution(img, resolution, return_roi=True)


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


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


@st.cache_data
def segment_image_with_style_adaptation(
    _model,
    _probe,
    _loss,
    _img,
    _roi,
    source,
    target,
    img_name,
    interpolation,
    radius=10,
    step_size=0.5,
    step_num=50,
    interpolation_mode="Loss",
):
    xmin = _img.reshape(_img.shape[0], -1).min(1, keepdim=True)[0].view(_img.shape[0], 1, 1, 1)
    xmax = _img.reshape(_img.shape[0], -1).max(1, keepdim=True)[0].view(_img.shape[0], 1, 1, 1)
    pgd = PGD(forward_func=_probe, loss_func=_loss, sign_grad=True, lower_bound=xmin, upper_bound=xmax)
    target = map_dataset_to_integer(FundusDataset(target))
    source = map_dataset_to_integer(FundusDataset(source))

    labels = [source, target]
    perturbed_img = pgd.perturb(
        _img,
        labels,
        step_size=step_size,
        radius=radius / 255.0,
        step_num=step_num,
        interpolation=interpolation,
        interpolation_mode=interpolation_mode.lower(),
        as_regression=False,
    )
    new_img = perturbed_img.detach()

    batch = dict(image=new_img, roi=_roi)
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    img = unorm(new_img).cpu().squeeze().permute(1, 2, 0).numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return _model.inference_step(batch), img


def plot_image(img):
    st.image(img, use_container_width=True)


def colorize_multiselect_options(selected) -> None:
    rules = ""
    n_colors = len(selected)

    for i, label in enumerate(selected):
        index = ALL_CLASSES.index(label)
        color = COLORS[1:][index]
        rules += f""".stMultiSelect div[data-baseweb="select"] span[data-baseweb="tag"]:nth-child({-n_colors}n+{i + 1}){{background-color: {color};}}"""

    st.markdown(f"<style>{rules}</style>", unsafe_allow_html=True)


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
    resolution = 1024

    if device.type == "cpu":
        resolution = 416
        st.warning(
            f"Running on CPU. This may be slow. Resolution reduced to {resolution}x{resolution} "
            f"(model was trained at 1024x1014, expect over segmentation)."
        )

    introduction = st.expander("Introduction", icon=":material/info:")
    with introduction:
        st.write(
            "This app allows you to segment fundus images using a pre-trained model. \
            Four classes are available: Exudates, Microaneurysms, Hemorrhages, and Soft Exudates. \
            We propose different weights, depending on the training dataset used, to segment the images."
        )
        st.write(
            "We illustrate the notion of label adaptation using adversarial attacks \
            to fit different styles of segmentation. \n \
            The method converts the segmentation style of a fundus image from one dataset to another. \
            The style conversion tool uses a probe model to adapt the style of the segmentation. \
            The model itself is never modified."
        )

    selectedLabels = st.multiselect("Labels", ALL_CLASSES, default=ALL_CLASSES, format_func=lambda x: x.name)
    colorize_multiselect_options(selectedLabels)
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
        interpolation = st.sidebar.slider("Interpolation", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        model_name = "ALL"

        configuration = st.sidebar.expander("Conversion settings", expanded=False)
        with configuration:
            interpolation_mode = st.radio(
                "Interpolation mode", ["Input", "Loss"], key="interpolation_mode", horizontal=True
            )
            step_size = st.slider("Step size", min_value=0.01, max_value=0.2, value=0.1, step=0.01)
            step_num = st.slider("Step number", min_value=1, max_value=20, value=5, step=1)
            radius = st.slider("Radius", min_value=1, max_value=20, value=5, step=1)

    if uploaded_file is not None:
        filename = uploaded_file.name
        model = load_model(model_name, device)
        img, roi = load_uploaded_file(uploaded_file, resolution=resolution)
        col1, col3 = st.columns([1, 1])

        alpha = st.slider("Alpha", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

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
                source=source,
                target=target,
                interpolation=interpolation,
                img_name=filename,
                radius=radius,
                step_size=step_size,
                step_num=step_num,
                interpolation_mode=interpolation_mode,
            )

        else:
            pred_map = segment_image(model, tensor_img, tensor_roi)

        pred_map = pred_map.squeeze().argmax(0)
        masks = np.zeros((*pred_map.shape, 3), dtype=np.uint8)
        for i in range(1, 5):
            if ALL_CLASSES[i - 1] not in selectedLabels:
                continue
            pred_class = (pred_map == i).cpu().numpy()
            hex_color = COLORS[i]
            # Convert hex to RGB
            rgb_color = tuple(int(hex_color[j : j + 2], 16) for j in (1, 3, 5))
            masks[pred_class] = np.asarray(rgb_color)
        bg = ~masks.any(axis=-1)
        masks[bg] = img[bg]
        img_with_masks = cv2.addWeighted(img, 1 - alpha, masks, alpha, 0)
        with col3:
            st.image(img_with_masks, use_container_width=True)


if __name__ == "__main__":
    app()
