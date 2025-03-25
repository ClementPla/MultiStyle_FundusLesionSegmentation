import os

import torch

import wandb
from fundseg.data.data_factory import ALL_DATASETS
from fundseg.models.c_ssn import CSNNStyleModel
from fundseg.models.smp_model import SMPModel


def load_model_from_checkpoints(
    project_name="Retinal Lesions Segmentation - V2", train_datasets=None, root_directory="", filters=None
) -> SMPModel:
    if train_datasets is None:
        train_datasets = ALL_DATASETS
    if not isinstance(train_datasets, list):
        train_datasets = [train_datasets]

    train_datasets = [d.name for d in train_datasets]

    train_datasets = sorted(train_datasets)

    api = wandb.Api()

    runs = api.runs(
        f"liv4d-polytechnique/{project_name}",
    )
    for r in list(runs)[::-1]:
        if filters:
            matching = True
            for k, v in filters.items():
                if r.config.get(k, None) != v:
                    matching = False
                    break
            if not matching:
                continue
        tags = r.tags
        tags = sorted(tags)
        tags = [t.replace("RETINAL_LESIONS", "RETLES") for t in tags]
        if tags == train_datasets:
            print("Found matching run", r.name, "with tags", tags)
            checkpoint_path = os.path.join(root_directory, f"checkpoints/{project_name}/{r.name}/")
            all_ckpts = os.listdir(checkpoint_path)
            best_model = next(_ for _ in all_ckpts if "epoch" in _)
            ckpt_path = os.path.join(checkpoint_path, best_model)
            break

    if "V2" in project_name:
        model = SMPModel.load_from_checkpoint(ckpt_path)
    elif "Conditional-Style-Segmentation-Networks" in project_name:
        model = CSNNStyleModel.load_from_checkpoint(ckpt_path)
    else:
        model = load_old(ckpt_path)

    return model


def load_old(ckpt_path):
    """
    This is used as fix from moving from segmentation_models_pytorch to torchseg
    It is very ugly, sorry.
    Since, some refactoring has changed the way with the packaging of the library.
    The "legacy" imports only exist to prevent pickle from complaining when loading the old checkpoints.
    """

    import sys

    import fundseg.legacy

    sys.modules["data"] = fundseg.legacy
    state_dict = torch.load(ckpt_path)
    arch = state_dict["hyper_parameters"]["arch"]
    encoder = state_dict["hyper_parameters"]["encoder"].replace("se_resnet", "seresnet")
    optim = state_dict["hyper_parameters"]["optimizer"]
    in_chans = state_dict["hyper_parameters"]["in_chans"]
    n_classes = state_dict["hyper_parameters"]["n_classes"]
    state_dict = state_dict["state_dict"]
    model = SMPModel(
        arch=arch, encoder=encoder, pretrained=False, optimizer=optim, in_chans=in_chans, n_classes=n_classes
    )
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("model.encoder.", "model.encoder.model.")
        k = k.replace("layer0.", "")
        k = k.replace("se_module.", "se.")
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    del sys.modules["data"]
    return model


if __name__ == "__main__":
    model = load_model_from_checkpoints(
        project_name="Retinal Lesions Segmentation",
        train_datasets=ALL_DATASETS,
        filters={"model_name": "unet-se_resnet50"},
    ).cuda()
    print(model)
