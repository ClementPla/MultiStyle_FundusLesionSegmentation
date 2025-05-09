import cv2
import numpy as np
import streamlit as st
import torch

from fundseg.data.data_factory import FundusDataset
from fundseg.data.utils import ALL_CLASSES
from fundseg.utils.colors import CMAP, COLORS
from ui.utils import (
    convert_to_tensor,
    load_cssn,
    load_model,
    load_probe_model,
    load_uploaded_file,
    segment_image,
    segment_image_with_style_adaptation,
)

st.set_page_config(layout="wide", page_title="Fundus Lesions Segmentation", page_icon=":eye:")


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


def segment_cssn(model, tensor_img, tensor_roi, target):
    batch = dict(image=tensor_img, roi=tensor_roi, tag=[target])
    with torch.inference_mode():
        pred_map = model.inference_step(batch)
    return pred_map


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
    if style_conversion:
        mode = st.sidebar.radio(
            "Mode",
            ["Style Conversion", "CSSN"],
            horizontal=True,
            help="Select the mode of the model.",
        )
    is_cssn = style_conversion and (mode == "CSSN")
    if not is_cssn:
        if not style_conversion:
            model_name = st.sidebar.selectbox("Checkpoints", ["IDRID", "FGADR", "RETLES", "MESSIDOR", "DDR", "ALL"])

        if style_conversion:
            st.sidebar.subheader("Segmentation style")
            target = st.sidebar.radio(
                "Target", ["IDRID", "FGADR", "RETLES", "MESSIDOR", "DDR"], horizontal=True, key="target"
            )
            interpolation = st.sidebar.slider("Interpolation", min_value=0.0, max_value=1.0, value=1.0, step=0.05)
            model_name = "ALL"

            configuration = st.sidebar.expander("Conversion settings", expanded=False)
            with configuration:
                step_size = st.slider(
                    "Step size", min_value=0.001, max_value=0.01, value=0.005, step=0.001, format="%0.3f"
                )
                step_num = st.slider("Step number", min_value=1, max_value=100, value=35, step=1)
                radius = st.slider("Radius", min_value=1, max_value=10, value=5, step=1)
    else:
        target = st.sidebar.radio(
            "Target", ["IDRID", "FGADR", "RETLES", "MESSIDOR", "DDR"], horizontal=True, key="target"
        )
    with st.sidebar:
        gradient_pred = st.checkbox(
            "Edges",
            value=True,
            help="Show the gradient of the segmentation map. \
            This is useful to see the boundaries of the lesions.",
        )
        gradient_width = st.slider(
            "Edges width",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            help="Width of the gradient. \
            This is useful to see the boundaries of the lesions.",
        )
    if uploaded_file is not None:
        filename = uploaded_file.name
        if is_cssn:
            model = load_cssn()
        else:
            model = load_model(model_name, device)
        img, roi = load_uploaded_file(uploaded_file, resolution=resolution)

        alpha = st.slider("Alpha", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
        col1, col3 = st.columns([1, 1])

        with col1:
            plot_image(img)

        tensor_img, tensor_roi = convert_to_tensor(img, roi, device)
        if style_conversion and not is_cssn:
            probe, loss = load_probe_model(model, device)
            pred_map, img = segment_image_with_style_adaptation(
                model,
                probe,
                loss,
                tensor_img,
                tensor_roi,
                target=target,
                interpolation=interpolation,
                img_name=filename,
                radius=radius,
                step_size=step_size,
                step_num=step_num,
            )

        elif is_cssn:
            pred_map = segment_cssn(model, tensor_img, tensor_roi, target=FundusDataset(target))

        else:
            pred_map = segment_image(model, tensor_img, tensor_roi)

        pred_map = pred_map.squeeze().argmax(0).cpu().numpy()
        gradient = cv2.morphologyEx(
            pred_map.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gradient_width, gradient_width)),
        )
        for i in range(1, 5):
            if ALL_CLASSES[i - 1] not in selectedLabels:
                pred_map[pred_map == i] = 0
                gradient[gradient == i] = 0

        masks_gradient = CMAP(gradient)[:, :, :3]
        masks_gradient = (masks_gradient * 255).astype(np.uint8)
        masks = CMAP(pred_map)[:, :, :3]
        masks = (masks * 255).astype(np.uint8)
        bg = ~masks.any(axis=-1)
        masks[bg] = img[bg]
        img_with_masks = cv2.addWeighted(img, 1 - alpha, masks, alpha, 0)
        if gradient_pred:
            img_with_masks[gradient > 0] = masks_gradient[gradient > 0]
        with col3:
            st.image(img_with_masks, use_container_width=True)


if __name__ == "__main__":
    app()
