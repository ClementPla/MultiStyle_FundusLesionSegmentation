from pathlib import Path

import cv2
import torch
import tqdm
from nntools.utils import Config
from torchmetrics import Accuracy, MetricCollection, Precision, Recall, Specificity

import wandb
from adptSeg.adaptation.adversarial import PGD
from adptSeg.adaptation.const import _all_datasets, get_class_mapping
from adptSeg.adaptation.utils import get_aptos_dataloader, get_probe_model_and_loss
from fundseg.data import ALL_CLASSES
from fundseg.data.data_factory import ALL_DATASETS, get_datamodule_from_config
from fundseg.data.data_factory import FundusDataset as Dataset
from fundseg.utils.checkpoints import load_model_from_checkpoints


def test_ddr_model_trained_on_idrid_retles():
    step_size = 0.001
    step_num = 20
    signed_gradient = True
    radius = 0.015
    batch_size = 16

    config_file = "configs/config.yaml"
    config = Config(config_file)
    config["data"]["batch_size"] = batch_size
    config_data = Config("configs/data_config.yaml")

    fundus_datamodule = get_datamodule_from_config(config)
    fundus_datamodule.setup("test")
    dataloaders = fundus_datamodule.test_dataloader()
    for dataloader in dataloaders:
        if dataloader.dataset.id == Dataset.FGADR:
            break
    probe, model, loss = get_probe_model_and_loss(
        model_type=Dataset.IDRID | Dataset.RETLES,
        probe_type=Dataset.IDRID | Dataset.RETLES,
        n_classes=2,
        as_regression=False,
        probe_datasets=[Dataset.IDRID, Dataset.RETLES],
    )
    pgd = PGD(forward_func=probe, loss_func=loss, sign_grad=signed_gradient)
    test_metrics = model.test_metrics

    wandb.init(
        project="HQ-LQ Style Conversion Test",
        name="Model trained on IDRID and RETLES tested before and after conversion",
        tags=[dataloader.dataset.id.name],
        config={
            "step_size": step_size,
            "step_num": step_num,
            "batch_size": batch_size,
            "signed_gradient": signed_gradient,
            "radius": radius,
        },
    )
    class_mapping = get_class_mapping(datasets=[Dataset.IDRID, Dataset.RETLES])
    targets = ["Reference", Dataset.IDRID, Dataset.RETLES]
    for target in targets:
        if target != "Reference":
            metrics = test_metrics.clone(prefix=f"Converted_to_{target.name}")
            int_target = class_mapping[target.value]
        else:
            metrics = test_metrics.clone(prefix="Reference")

        for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
            x = batch["image"].cuda()
            roi = batch["roi"].cuda()
            mask = batch["mask"].long().cuda()
            batch_size = x.shape[0]

            if target != "Reference":
                y = torch.tensor([int_target]).cuda().expand(batch_size)
                x1 = pgd.perturb(x, target=y, step_size=step_size, step_num=step_num, radius=radius)
                x = (x1 + x) / 2
            with torch.inference_mode():
                predicted = model(x)
                prob = model.get_prob(predicted, roi)
                metrics.update(prob, mask)

        score = model.setup_scores(metrics)
        wandb.log(score)
    wandb.finish()


def test_conversion_vs_specialized():
    targets = _all_datasets
    batch_size = 8

    # Original values
    # step_size = 0.005
    # step_num = 25

    step_size = 0.025
    step_num = 10
    signed_gradient = True
    radius = 0.025

    dataloader = get_aptos_dataloader(batch_size, grade_filter=2)
    probe, model, loss = get_probe_model_and_loss()

    pgd = PGD(forward_func=probe, loss_func=loss, sign_grad=signed_gradient)
    test_metrics = model.test_metrics

    wandb.init(
        project="Adaptation Style Test",
        name="APTOS Converted VS Specialized",
        config={
            "step_size": step_size,
            "step_num": step_num,
            "batch_size": batch_size,
            "signed_gradient": signed_gradient,
            "radius": radius,
        },
    )
    class_mapping = get_class_mapping(datasets=_all_datasets)
    model.eval()
    for target in targets:
        for special_model in targets:
            specialized_model = load_model_from_checkpoints(train_datasets=special_model)
            metrics = test_metrics.clone(prefix=f"ConvertedTo{target.name}ComparedTo{special_model.name}_")
            int_target = class_mapping[target.value]
            for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
                x = batch["image"].cuda()
                roi = batch["roi"].cuda()
                batch_size = x.shape[0]
                y = torch.tensor([int_target]).cuda().expand(batch_size)
                perturbed_input = pgd.perturb(x, target=y, step_size=step_size, step_num=step_num, radius=radius)
                with torch.inference_mode():
                    predicted = model(perturbed_input)
                    prob = model.get_prob(predicted, roi)
                    gt = specialized_model(x)
                    gtprob = specialized_model.get_prob(gt, roi)
                    gt = torch.argmax(gtprob, dim=1)
                    metrics.update(prob, gt)
            score = model.setup_scores(metrics)
            wandb.log(score)
    wandb.finish()


def _test_aptos_conversion(step_size, step_num, signed_gradient, radius):
    batch_size = 16
    dataloader = get_aptos_dataloader(batch_size)
    probe, _, loss = get_probe_model_and_loss()
    pgd = PGD(forward_func=probe, loss_func=loss, sign_grad=signed_gradient)
    wandb.init(
        project="Conversion Test",
        name="APTOS Probe Conversion",
        config={
            "step_size": step_size,
            "step_num": step_num,
            "batch_size": batch_size,
            "signed_gradient": signed_gradient,
            "radius": radius,
        },
    )

    metric_kwargs = {"num_classes": 5, "task": "multiclass"}
    targets = _all_datasets
    batch_size = 16
    class_mapping = get_class_mapping(datasets=_all_datasets)
    for target in targets:
        int_target = class_mapping[target.value]
        metrics = MetricCollection(
            [
                Accuracy(**metric_kwargs),
                Recall(**metric_kwargs),
                Precision(**metric_kwargs),
                Specificity(**metric_kwargs),
            ],
            prefix=f"{target.name}_",
        ).cuda()
        predictions = []
        gts = []
        for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
            x = batch["image"].cuda()
            batch_size = x.shape[0]
            y = torch.tensor([int_target]).cuda().expand(batch_size)
            perturbed_input = pgd.perturb(x, target=y, step_size=step_size, step_num=step_num, radius=radius)
            pred = probe(perturbed_input).argmax(dim=1)

            predictions.append(pred.cpu())
            gts.append(y.cpu())
            metrics.update(pred, y)
        predictions = torch.cat(predictions).numpy()
        gts = torch.cat(gts).numpy()
        wandb.log(
            {
                f"Target {target.name}_confMat": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=gts,
                    preds=predictions,
                    class_names=[t.name for t in _all_datasets],
                    title=f"Confusion Matrix for {target.name}",
                )
            }
        )
        wandb.log(metrics.compute())

    wandb.finish()


def test_aptos_conversion():
    # Original values
    # step_size = 0.005
    # step_num = 25
    max_step = 0.025
    for T in [1, 5, 10, 25, 50]:
        step_size = max_step / T
        step_num = T
        signed_gradient = True
        radius = max_step
        _test_aptos_conversion(step_size, step_num, signed_gradient, radius)


@torch.inference_mode()
def inference_and_save_aptos(savepath):
    from pynotate.project import Project

    savepath = Path(savepath)

    batch_size = 32
    dataloader = get_aptos_dataloader(batch_size)
    model = load_model_from_checkpoints(train_datasets=Dataset.IDRID)
    dataloader.dataset.return_indices = True
    with Project(
        "Aptos Segmentation - IDRiD",
        input_dir=savepath / "Aptos",
        output_dir=savepath,
        segmentation_classes=[c.name for c in ALL_CLASSES],
        is_segmentation=True,
    ) as project:
        for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
            x = batch["image"].cuda()
            y = batch["index"]
            filenames = dataloader.dataset.filename(y)
            pred = model(x).argmax(dim=1)
            for i, filename in enumerate(filenames):
                image = x[i].cpu().numpy().transpose(1, 2, 0)
                masks = [(pred[i] == (j + 1)).cpu().numpy().astype("uint8") * 255 for j in range(len(ALL_CLASSES))]
                project.load_image(
                    segmentation_masks=masks,
                    image=image,
                    normalize=True,
                    filename=filename,
                )


if __name__ == "__main__":
    inference_and_save_aptos("/home/clement/Documents/Results/")
