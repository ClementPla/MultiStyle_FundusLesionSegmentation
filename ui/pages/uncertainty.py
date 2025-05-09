from copy import deepcopy

import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from matplotlib import pyplot as plt
from plotly.subplots import make_subplots

from adptSeg.adaptation.const import _all_datasets, map_dataset_to_integer
from fundseg.data.data_factory import FundusDataset
from fundseg.utils.colors import CMAP
from ui.utils import (
    PGD,
    convert_to_tensor,
    load_cssn,
    load_devis,
    load_model,
    load_probe_model,
    load_uploaded_file,
    load_voronoi_cnn,
)


def normalize(tensor):
    tensor = tensor - tensor.min()
    tensor = tensor / tensor.max()
    return tensor


def to_255(
    tensor,
):
    tensor = normalize(tensor)
    return tensor


@st.cache_data
def deep_ensembling(_tensor, _roi, softmax_alpha, image_id):
    hyps = []

    batch = dict(image=_tensor, roi=_roi)
    for dataset in _all_datasets:
        model = load_model(dataset).cuda()
        pred = model.inference_step(batch, temperature=softmax_alpha)
        pred = pred.cpu()
        hyps.append(pred)
    predictions = torch.concat(hyps, dim=0)
    std_pred = torch.std(predictions[:, 1:].sum(dim=1), dim=0)
    pred = torch.mean(predictions, dim=0).argmax(0)
    return pred.cpu().numpy(), to_255(std_pred.squeeze(0)).cpu().numpy()


@st.cache_data
def voronoi_uncertainty(_tensor, _roi, softmax_alpha, image_id):
    model = load_voronoi_cnn()
    with torch.inference_mode():
        pred = model(_tensor)
        pred = [torch.softmax(p * softmax_alpha, dim=1) * _roi for p in pred]

        pred = torch.stack(pred).squeeze()
        mean = pred.mean(dim=0)
        std_voronoi = to_255(torch.std(pred[:, 1:].sum(1), dim=0))
        std_voronoi = std_voronoi.cpu().numpy()
    return mean.argmax(0).cpu().numpy(), std_voronoi


@st.cache_data
def devis_uncertainty(_tensor, _roi, image_id):
    model = load_devis()
    batch = dict(image=_tensor, roi=_roi)
    with torch.inference_mode():
        prob, evidence_devis = model.get_prob_and_evidence(batch)
        evidence_devis = (1 / evidence_devis).mean(1).squeeze().cpu().numpy()
    return prob.argmax(1).squeeze(0).cpu().numpy(), evidence_devis


@st.cache_data
def CSNN_uncertainty(_tensor, _roi, softmax_alpha, image_id):
    tags = _all_datasets
    model = load_cssn()
    _tensor = torch.concat([_tensor] * len(tags), dim=0)
    _roi = torch.concat([_roi] * len(tags), dim=0)
    batch = dict(image=_tensor, roi=_roi, tag=tags)
    pred = model.inference_step(batch, temperature=softmax_alpha)
    mean = pred.mean(dim=0)
    std = (pred[:, 1:].sum(1)).std(dim=0)
    return mean.argmax(0).cpu().numpy(), to_255(std).cpu().numpy()


def generate_hypothesis(model, probe, loss, batch, temperature, target, n=50):
    pgd = PGD(forward_func=probe, loss_func=loss, sign_grad=False)
    batch = deepcopy(batch)
    img = batch["image"]
    roi
    img.grad = None
    perturbed_img = pgd.perturb(
        img,
        target,
        step_size=0.010,
        radius=10 / 255.0,
        step_num=25,
        interpolation=None,
        interpolation_mode="input",
        as_regression=False,
    )
    out = []

    alphas = np.linspace(0, 1, n)
    for alpha in alphas:
        with torch.inference_mode():
            x = perturbed_img * alpha + img * (1 - alpha)

            pred = model.inference_step(dict(image=x, roi=roi), temperature=1.0)
            out.append(pred)
    return torch.stack(out, dim=0)


def plot(image, uncertainty, pred, threshold, color, cmap):
    uncertainty[uncertainty < threshold] = np.nan
    uncertainty_cm = (plt.get_cmap(cmap)(uncertainty) * 255).astype(np.uint8)
    pred_cmap = CMAP(pred)
    tab1, tab2, tab3 = st.tabs(["Prediction", "Uncertainty", "Combined"])
    with tab1:
        st.image(pred_cmap, caption="Segmentation Map", use_container_width=True)
    with tab2:
        st.image(uncertainty_cm, caption="Uncertainty Map", use_container_width=True)
    with tab3:
        gradient = cv2.morphologyEx((pred > 0).astype(np.uint8), cv2.MORPH_GRADIENT, kernel)

        uncertainty_cm[gradient > 0] = color
        mask = uncertainty_cm[:, :, -1:] > 0
        combined = uncertainty_cm * mask + cv2.cvtColor(image, cv2.COLOR_RGB2RGBA) * (1 - mask)
        st.image(
            combined,
            caption="Combined Map",
            use_container_width=True,
        )


@st.cache_data
def multistyle_uncertainty(_tensor, _roi, softmax_alpha, n, uncertainty_targets, image_id):
    model = load_model("ALL").cuda()
    probe, loss = load_probe_model(model)
    probe = probe.cuda()
    loss = loss.cuda()
    hyps = []
    batch = dict(image=_tensor, roi=_roi)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        for target in uncertainty_targets:
            hyp = generate_hypothesis(
                model, probe, loss, batch, softmax_alpha, map_dataset_to_integer(FundusDataset(target)), n=n
            ).squeeze(1)
            hyps.append(hyp)

        hyps = torch.cat(hyps, dim=0)
        std_pred = torch.std(hyps[:, 1:].sum(dim=1), dim=0)

        pred = torch.mean(hyps, dim=0).argmax(0)
    return pred.cpu().numpy(), to_255(std_pred.squeeze(0)).cpu().numpy()


st.set_page_config(
    page_title="Uncertainty Estimation",
    page_icon="🧪",
    layout="wide",
)

uploaded_file = st.sidebar.file_uploader(
    "Upload a fundus image", type=["png", "jpg", "jpeg"], accept_multiple_files=False
)

with st.sidebar.expander("Options", expanded=True):
    softmax_alpha = st.slider(
        "Softmax temperature",
        min_value=0.01,
        max_value=1.0,
        value=0.03,
        step=0.01,
        help="Lower values make the model more uncertain.",
    )

    uncertainty_targets = st.multiselect(
        "Uncertainty target",
        options=["DDR", "IDRID", "RETLES", "MESSIDOR", "FGADR"],
        default=["DDR", "RETLES"],
        help="Select the targeted style to use for uncertainty estimation.",
    )

    n_iterations = st.slider(
        "Number of iterations",
        min_value=1,
        max_value=100,
        value=10,
        step=1,
        help="Number of iterations for the multi-style uncertainty estimation.",
    )
with st.sidebar:
    threshold = st.slider(
        "Uncertainty threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Threshold for the uncertainty map. Pixels below this value will be set to NaN.",
    )
    color = st.color_picker(
        "Select a color for the prediction gradients",
        value="#0000FF",
    )
    color = np.array([int(color[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    color = np.concatenate([color, np.array([255], dtype=np.uint8)]).reshape(1, 1, 4)
    gradient_width = st.slider(
        "Gradient width",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
        help="Width of the gradient around the prediction.",
    )
    cmap = st.selectbox(
        "Select a colormap",
        options=plt.colormaps(),
        index=0,
        help="Select a colormap for the uncertainty map.",
    )

if uploaded_file is not None:
    image, roi = load_uploaded_file(uploaded_file, resolution=1024)
    with st.sidebar:
        st.image(
            image,
            caption="Original Image",
            use_container_width=True,
        )

    tensor, roi = convert_to_tensor(image, roi=roi, device="cuda")

    (
        col1,
        col2,
        col3,
    ) = st.columns([1, 1, 1])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gradient_width, gradient_width))
    with col1:
        st.header("Voronoi CNN")
        pred, uncertainty = voronoi_uncertainty(tensor, roi, softmax_alpha, image_id=uploaded_file.file_id)
        plot(
            image,
            uncertainty,
            pred,
            threshold=threshold,
            color=color,
            cmap=cmap,
        )

    with col2:
        st.header("DEviS")

        pred, uncertainty = devis_uncertainty(tensor, roi, image_id=uploaded_file.file_id)
        uncertainty = 1 / uncertainty
        uncertainty = to_255(uncertainty)
        plot(
            image,
            uncertainty,
            pred,
            threshold=threshold,
            color=color,
            cmap=cmap,
        )

    with col3:
        st.header("Multi-Style")
        pred, uncertainty = multistyle_uncertainty(
            tensor,
            roi,
            softmax_alpha,
            n=n_iterations,
            uncertainty_targets=uncertainty_targets,
            image_id=uploaded_file.file_id,
        )
        plot(
            image,
            uncertainty,
            pred,
            threshold=threshold,
            color=color,
            cmap=cmap,
        )

    col4, col5, _ = st.columns([1, 1, 1])
    with col4:
        st.header("Ensembling")
        pred, uncertainty = deep_ensembling(
            tensor,
            roi,
            softmax_alpha,
            image_id=uploaded_file.file_id,
        )
        plot(
            image,
            uncertainty,
            pred,
            threshold=threshold,
            color=color,
            cmap=cmap,
        )

    with col5:
        st.header("CSNN")
        pred, uncertainty = CSNN_uncertainty(
            tensor,
            roi,
            softmax_alpha=softmax_alpha,
            image_id=uploaded_file.file_id,
        )
        plot(
            image,
            uncertainty,
            pred,
            threshold=threshold,
            color=color,
            cmap=cmap,
        )
