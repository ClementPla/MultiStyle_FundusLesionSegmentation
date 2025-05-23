from fundseg.models.attn_guided_uncert_unet import DEviSModule, UNetAFFModule
from fundseg.models.lesion_net import LesionNet
from fundseg.models.lseg import LSeg
from fundseg.models.pcmnet import ProgressiveMultiscaleConsistentNetwork
from fundseg.models.smp_model import SMPModel
from fundseg.models.voronoi_cnn import VoronoiCNN


def get_model(model_name, **kwargs):
    if model_name == "pcmnet":
        model = ProgressiveMultiscaleConsistentNetwork(use_batchnorm=True)
    elif model_name.startswith("lesionnet"):
        variant = int(model_name.split("lesionnet")[-1])
        model = LesionNet(variant=variant, use_batchnorm=True)
    elif model_name == "lseg":
        model = LSeg(encoder="vgg19", use_batchnorm=True)
    elif model_name.startswith("voronoi"):
        arch = model_name.split("_")[0]
        encoder = "_".join(model_name.split("_")[1:])
        model = VoronoiCNN(encoder="seresnext50_32x4d", arch="unet", **kwargs)
    elif model_name.startswith("attn_guided_uncert_unet"):
        model = UNetAFFModule(n_classes=5, **kwargs)
    elif model_name.startswith("devis"):
        arch = model_name.split("_")[1]

        encoder = "_".join(model_name.split("_")[2:])

        print(f"Using DEviSModule with encoder: {arch} and {encoder}")
        model = DEviSModule(encoder=encoder, arch=arch, **kwargs)
    else:
        arch = model_name.split("_")[0]
        encoder = "_".join(model_name.split("_")[1:])
        model = SMPModel(encoder=encoder, arch=arch, **kwargs)
    return model
