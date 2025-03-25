import segmentation_models_pytorch as smp
import torch
from fundseg.models import LesionNet, LSeg, ProgressiveMultiscaleConsistentNetwork

if __name__=="__main__":
    archs = ['unet', 'unetplusplus', 'fpn']
    encoders = ['timm-efficientnet-b5', 'mit_b2', 'resnet50', 'se_resnext50_32x4d']
    foo = torch.rand(1, 3, 128, 128)
    for arch in archs:
        for encoder in encoders:
            if arch=='unetplusplus' and encoder=='mit_b2':
                continue
            print('Building model with arch: {} and encoder: {}'.format(arch, encoder))
            model = smp.create_model(arch, encoder_name=encoder, in_channels=3, classes=2, encoder_weights=None)
            try:
                out = model(foo)
            except Exception as e:
                print('Model failed with error: {}'.format(e))

    