import os

import torch

import wandb
from fundseg.data.data_factory import ALL_DATASETS, FundusDataset
from fundseg.models.smp_model import SMPModel


def load_model_from_checkpoints(
    project_name="Retinal Lesions Segmentation - V2", train_datasets=None, root_directory=""
) -> SMPModel:
    if train_datasets is None:
        train_datasets = ALL_DATASETS
    if not isinstance(train_datasets, list):
        train_datasets = [train_datasets]
    train_datasets = [d for d in train_datasets]

    train_datasets = sorted(train_datasets)
    api = wandb.Api()

    runs = api.runs(f"liv4d-polytechnique/{project_name}")

    for r in runs:
        tags = r.tags
        tags = sorted(tags)
        if tags == train_datasets:
            checkpoint_path = os.path.join(root_directory, f"checkpoints/{project_name}/{r.name}/")
            all_ckpts = os.listdir(checkpoint_path)
            best_model = next(_ for _ in all_ckpts if "epoch" in _)
            ckpt_path = os.path.join(checkpoint_path, best_model)
            break

    model = SMPModel.load_from_checkpoint(ckpt_path)
    return model


if __name__ == "__main__":
    model = load_model_from_checkpoints(train_datasets=[FundusDataset.IDRID, FundusDataset.MESSIDOR]).cuda()
    foo = torch.rand(1, 3, 512, 512).cuda()
    decoder_output = []

    def catch_output(module, input, output):
        decoder_output.append(output)

    hook_handle = model.model.decoder.blocks[-4].register_forward_hook(catch_output)

    output = model(foo)
    print(decoder_output[0].shape)

    hook_handle.remove()
