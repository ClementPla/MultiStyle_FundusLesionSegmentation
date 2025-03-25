from typing import List, Optional

import torch
import tqdm
from fundus_data_toolkit.data_aug import DAType
from fundus_data_toolkit.datamodules.segmentation import TJDRDataModule_s
from nntools.utils import Config
from torchmetrics.classification import Dice, MulticlassAccuracy, MulticlassJaccardIndex

import wandb
from adptSeg.adaptation.adversarial import PGD
from adptSeg.adaptation.const import (
    FundusDataset,
    map_dataset_to_integer,
)
from adptSeg.utils.checkpoints import load_probe_from_checkpoint
from fundseg.data.data_factory import ALL_DATASETS, SEG_PATHS
from fundseg.utils.checkpoints import load_model_from_checkpoints


def tjdr_testing(
    target_conversion: Optional[FundusDataset] = None,
    on_train: bool = True,
    project_name: str = "TJDR Testing",
    model_trained_on: Optional[List[FundusDataset]] = None,
):
    if model_trained_on is None:
        model_trained_on = ALL_DATASETS

    config = Config("configs/config.yaml")
    config["data"]["use_cache"] = False
    config["data"]["random_crop"] = None
    config["data"]["data_augmentation_type"] = DAType.NONE
    config["data"]["eval_batch_size"] = 32
    config["data"]["batch_size"] = 32
    config["data"]["num_workers"] = 10
    datamodule = TJDRDataModule_s(data_dir=SEG_PATHS.TJDR, **config["data"])
    datamodule.da_type = None
    datamodule.setup_all()

    model = load_model_from_checkpoints(train_datasets=model_trained_on).eval()
    metrics = model.single_test_metrics

    if target_conversion:
        position = 4
        ftype = "ENCODER"
        probe = load_probe_from_checkpoint(model, matching_params={"position": position, "feature_type": ftype})
        loss = probe.criterion

        pgd = PGD(forward_func=probe, loss_func=loss)
    if on_train:
        dataloader = datamodule.train_dataloader()
    else:
        dataloader = datamodule.test_dataloader()
    wandb.init(
        project=project_name,
        config={"target_conversion": target_conversion, "on_train": on_train},
        tags=[super(FundusDataset, d).name for d in model_trained_on],
    )
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        x = batch["image"].cuda()
        gt = batch["mask"].cuda().long()
        roi = batch["roi"].cuda()
        if target_conversion:
            perturbed_input = pgd.perturb(
                x,
                target=map_dataset_to_integer(target_conversion),
                step_size=0.005,
                step_num=100,
                radius=4 / 255,
                targeted=True,
            )
            xmin = x.flatten(-2).min(dim=2).values
            roi = roi.unsqueeze(1)
            x = perturbed_input * roi + xmin.unsqueeze(-1).unsqueeze(-1) * (1 - roi)
            roi = roi.squeeze(1)

        with torch.inference_mode():
            pred = model(x)
            pred = model.get_prob(pred, roi)
            metrics.update(pred, gt)
    scores = model.setup_scores(metrics)
    wandb.log(scores)
    wandb.finish()


def tjdr_testing_cssn(
    target_conversion: Optional[FundusDataset] = None,
    on_train: bool = True,
    project_name: str = "TJDR Testing CSSN",
):
    config = Config("configs/config.yaml")
    config["data"]["use_cache"] = False
    config["data"]["random_crop"] = None
    config["data"]["data_augmentation_type"] = DAType.NONE
    config["data"]["eval_batch_size"] = 32
    config["data"]["batch_size"] = 32
    config["data"]["num_workers"] = 10
    datamodule = TJDRDataModule_s(data_dir=SEG_PATHS.TJDR, **config["data"])
    datamodule.da_type = None
    datamodule.setup_all()

    model = load_model_from_checkpoints(
        project_name="Conditional-Style-Segmentation-Networks", train_datasets=ALL_DATASETS
    ).eval()
    metrics = model.single_test_metrics

    if on_train:
        dataloader = datamodule.train_dataloader()
    else:
        dataloader = datamodule.test_dataloader()
    wandb.init(
        project=project_name,
        config={"target_conversion": target_conversion, "on_train": on_train},
    )
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        x = batch["image"].cuda()
        gt = batch["mask"].cuda().long()
        batch["tag"] = [target_conversion] * x.shape[0]
        pred = model.inference_step(batch)
        metrics.update(pred, gt)

    scores = model.setup_scores(metrics)
    wandb.log(scores)
    wandb.finish()


if __name__ == "__main__":
    datasets = [
        FundusDataset.IDRID,
        FundusDataset.RETLES,
        FundusDataset.DDR,
        FundusDataset.MESSIDOR,
        FundusDataset.FGADR,
    ]
    for dataset in datasets:
        tjdr_testing_cssn(
            on_train=True,
            target_conversion=dataset,
            project_name="TJDR Testing CSSN - V2",
        )
        tjdr_testing_cssn(
            on_train=False,
            target_conversion=dataset,
            project_name="TJDR Testing CSSN - V2",
        )
