from copy import deepcopy

import albumentations as A
import numpy as np
import PIL
import streamlit as st
import torch
from fundus_data_toolkit.functional import autofit_fundus_resolution

from adptSeg.adaptation.adversarial import PGD
from adptSeg.adaptation.const import map_dataset_to_integer
from adptSeg.adaptation.model_wrapper import ModelFeaturesExtractor
from adptSeg.adaptation.probe import ProbeModule
from fundseg.data.data_factory import FundusDataset
from fundseg.models.attn_guided_uncert_unet import DEviSModule, UNetAFFModule
from fundseg.models.c_ssn import CSNNStyleModel
from fundseg.models.smp_model import SMPModel
from fundseg.models.voronoi_cnn import VoronoiCNN


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
def load_model(model_name, device="cuda", **kwargs):
    model = SMPModel.from_pretrained(f"ClementP/MultiStyle_FundusLesionSegmentation", revision=model_name, **kwargs)
    model = model.to(device)
    model.eval()
    return model


def load_probe_model(_model, device="cuda"):
    position = 4
    encoder = ModelFeaturesExtractor(_model, position=position, feature_type="encoder").to(device)
    model = ProbeModule.from_pretrained(
        "ClementP/MultiStyle_FundusLesionSegmentation",
        revision=f"probe-{position}",
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
    target,
    img_name,
    interpolation,
    radius=10,
    step_size=0.5,
    step_num=50,
):
    pgd = PGD(forward_func=_probe, loss_func=_loss, sign_grad=False)
    target = map_dataset_to_integer(FundusDataset(target))
    _img.grad = None
    labels = target
    perturbed_img = pgd.perturb(
        _img.clone(),
        labels,
        step_size=step_size,
        radius=radius / 255.0,
        step_num=step_num,
        interpolation=None,
        as_regression=False,
    )
    xmin = _img.flatten(-2).min(dim=2).values
    x = perturbed_img * _roi + xmin.unsqueeze(-1).unsqueeze(-1) * (1 - _roi)
    x = x * interpolation + _img * (1 - interpolation)
    with torch.inference_mode():
        pred = _model.inference_step(dict(image=x, roi=_roi), temperature=1.0)
        unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        img = unorm(x).cpu().squeeze().permute(1, 2, 0).numpy()
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return pred, img


@st.cache_resource
def load_voronoi_cnn(device="cuda"):
    model = (
        VoronoiCNN.from_pretrained(
            "ClementP/MultiStyle_FundusLesionSegmentation",
            revision="VORONOI_CNN",
        )
        .eval()
        .to(device)
    )
    return model


@st.cache_resource
def load_devis(device="cuda"):
    model = (
        DEviSModule.from_pretrained(
            "ClementP/MultiStyle_FundusLesionSegmentation", arch="unet", encoder="seresnext50_32x4d", revision="DEviS"
        )
        .eval()
        .to(device)
    )
    return model


@st.cache_resource
def load_cssn(device="cuda"):
    model = (
        CSNNStyleModel.from_pretrained(
            "ClementP/MultiStyle_FundusLesionSegmentation",
            revision="CSNN",
        )
        .eval()
        .to(device)
    )
    return model
