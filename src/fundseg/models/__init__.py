from fundseg.models.lesion_net import LesionNet
from fundseg.models.lseg import LSeg
from fundseg.models.pcmnet import ProgressiveMultiscaleConsistentNetwork
from fundseg.models.smp_model import SMPModel


def get_model(model_name, **kwargs):
    if model_name=="pcmnet":
        model = ProgressiveMultiscaleConsistentNetwork(use_batchnorm=True)
    elif model_name.startswith("lesionnet"):
        variant = int(model_name.split("lesionnet")[-1])
        model = LesionNet(variant=variant, use_batchnorm=True)
    elif model_name=="lseg":
        model = LSeg(encoder='vgg19', use_batchnorm=True)
    else:
        arch = model_name.split("_")[0]
        encoder = '_'.join(model_name.split("_")[1:])
        model = SMPModel(encoder=encoder, arch=arch, **kwargs)
    return model