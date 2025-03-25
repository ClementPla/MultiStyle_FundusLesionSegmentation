import os

import torch

import wandb
from adptSeg.adaptation.model_wrapper import FeatureType, ModelFeaturesExtractor
from adptSeg.adaptation.probe import ProbeModule


def load_probe_from_checkpoint(
    src_model,
    project_name: str = "Probing-Lesions-Segmentation-Positions",
    root_directory: str = "",
    matching_params: dict = None,
) -> ProbeModule:
    api = wandb.Api()
    runs = api.runs(f"liv4d-polytechnique/{project_name}")
    for r in runs:
        if matching_params is not None:
            is_matching = True
            for k, v in matching_params.items():
                if r.config.get(k, None) != v:
                    is_matching = False
            if not is_matching:
                continue

            checkpoint_path = os.path.join(root_directory, f"checkpoints/probing/{r.name}/")
            all_ckpts = os.listdir(checkpoint_path)
            best_model = next(_ for _ in all_ckpts if "epoch" in _)
            ckpt_path = os.path.join(checkpoint_path, best_model)
            break

        else:
            raise ValueError("No matching params provided")
    if not is_matching:
        raise ValueError("No matching checkpoint found")

    feature_extractor = ModelFeaturesExtractor(
        src_model, position=r.config["position"], feature_type=FeatureType[r.config["feature_type"]]
    )
    model = ProbeModule.load_from_checkpoint(
        ckpt_path, featureExtractor=feature_extractor, weights=torch.Tensor(r.config["weights"])
    )

    return model
