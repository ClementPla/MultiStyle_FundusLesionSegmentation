from enum import Enum

import torch
import torch.nn as nn
from fundseg.models.smp_model import SMPModel


class FeatureType(str, Enum):
    ENCODER = 'encoder'
    DECODER = 'decoder'
    
    

class ModelFeaturesExtractor(nn.Module):
    def __init__(self, model:SMPModel, feature_type: FeatureType=FeatureType.ENCODER, position=0):
        super().__init__()
        self.model = model.model.cpu()
        self.model.eval()
        self.model.requires_grad_(False)
        
        self.feature_type = feature_type
        self.output = []
        self.position = position
        self.hook_handle = None 
        match self.feature_type:
            case FeatureType.ENCODER:
                self.out_chans = self.model.encoder.out_channels[self.position]
            case FeatureType.DECODER:
                self.hook_handle = self.attach_hook()
                foo = torch.rand(1, 3, 512, 512).to(device='cpu', )
                features = self.forward(foo)
                self.out_chans = features.shape[1]
            case _:
                raise ValueError(f'Invalid type {feature_type}')
        
    def forward(self, x):
        self.eval()
        match self.feature_type:
            case FeatureType.ENCODER:
                features = self.model.encoder(x)[self.position]
            case FeatureType.DECODER:
                features = self.model(x)
                features = self.output[0]
                self.output = []
            case _:
                raise ValueError(f'Invalid type {self.feature_type}')
        return features
    
    def attach_hook(self):
        def hook_fn(module, input, output):
            self.output.append(output)
        hook_handle = self.model.decoder.blocks[self.position].register_forward_hook(hook_fn)
        return hook_handle

    
    def __del__(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
    
    def delete_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None