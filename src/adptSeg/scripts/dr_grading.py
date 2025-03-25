import os

import timm
import torch
import torch.nn.functional as F
import tqdm
from nntools.utils import Config

from adptSeg.adaptation.adversarial import PGD
from adptSeg.adaptation.const import map_dataset_to_integer
from adptSeg.utils.checkpoints import load_probe_from_checkpoint
from fundseg.data.data_factory import ALL_DATASETS, FundusDataset, get_datamodule_from_config
from fundseg.utils.checkpoints import load_model_from_checkpoints


def main():
    batch_size = 8
    config = Config("configs/config.yaml")
    config["data"]["eval_batch_size"] = batch_size
    config["data"]["use_cache"] = False
    datamodule = get_datamodule_from_config(
        config["datasets"], training_datasets=ALL_DATASETS, dataset_args=config["data"], separate_test_test=True
    )
    position = 4
    ftype = "ENCODER"
    # model = load_model_from_checkpoints(
    #     project_name="Retinal Lesions Segmentation",
    #     train_datasets=ALL_DATASETS,
    #     filters={"model_name": "unet-se_resnet50"},
    # )
    path = f"ClementP/FundusDRGrading-convnext_base"
    dr_model = timm.create_model(f"hf_hub:{path}", pretrained=True, num_classes=1).cuda()

    model = load_model_from_checkpoints(train_datasets=ALL_DATASETS)
    probe = load_probe_from_checkpoint(model, matching_params={"position": position, "feature_type": ftype})

    loss = probe.criterion
    # loss.weight = torch.ones_like(loss.weight)

    probe_type = f"{ftype}_P{position}"
    test_dataloaders = datamodule.test_dataloader()
    to_dataset = FundusDataset.RETLES
    class_mapping = map_dataset_to_integer(to_dataset)
    to_dataset = to_dataset.name
    pgd = PGD(forward_func=probe, loss_func=loss)

    save_folder = f"results/DR_grading/{probe_type}/"
    os.makedirs(save_folder, exist_ok=True)

    for dataloader in test_dataloaders:
        from_dataset = dataloader.dataset.tag.name
        pred_x = []
        pred_perturbed = []
        print("Running on dataset", from_dataset)
        for i, batch in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            x = batch["image"].cuda()
            batch_size = x.size(0)
            roi = batch["roi"].cuda()
            target = torch.tensor([class_mapping]).cuda().expand(batch_size)
            perturbed_input = pgd.perturb(x, target=target, step_size=0.01, step_num=50, radius=5 / 255, targeted=True)
            # perturbed_input = perturbed_input * alpha + (1 - alpha) * x
            roi = roi.unsqueeze(1)
            perturbed_input = perturbed_input

            with torch.inference_mode():
                x = F.interpolate(x, size=(512, 512), mode="bilinear", align_corners=False)
                perturbed_input = F.interpolate(perturbed_input, size=(512, 512), mode="bilinear", align_corners=False)

                pred1 = dr_model(x).cpu()
                pred2 = dr_model(perturbed_input).cpu()
                pred_x.append(pred1)
                pred_perturbed.append(pred2)
        pred_x = torch.cat(pred_x, dim=0)
        pred_perturbed = torch.cat(pred_perturbed, dim=0)
        data = {
            "pred_x": pred_x,
            "pred_perturbed": pred_perturbed,
        }
        torch.save(data, os.path.join(save_folder, f"{from_dataset}_to_{to_dataset}.pth"))


if __name__ == "__main__":
    main()
