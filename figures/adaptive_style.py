import os

import cv2
import numpy as np
import torch
import tqdm
from adptSeg.adaptation.adversarial import PGD
from adptSeg.adaptation.const import batch_integer_to_dataset, map_dataset_to_integer
from adptSeg.utils.checkpoints import load_probe_from_checkpoint
from fundseg.data.data_factory import ALL_DATASETS, SEG_PATHS, FundusDataset, get_datamodule_from_config
from fundseg.utils.checkpoints import load_model_from_checkpoints
from fundseg.utils.colors import COLORS
from fundus_data_toolkit.data_aug import DAType
from fundus_data_toolkit.datamodules.segmentation import TJDRDataModule_s
from nntools.utils import Config


def main():
    batch_size = 4
    config = Config("configs/config.yaml")
    batch_size = 4
    config = Config("configs/config.yaml")
    config["data"]["eval_batch_size"] = batch_size
    config["data"]["use_cache"] = False
    config["data"]["batch_size"] = batch_size * 2
    config["data"]["random_crop"] = None
    config["data"]["data_augmentation_type"] = DAType.NONE
    # datamodule = get_datamodule_from_config(
    #     config["datasets"], training_datasets=ALL_DATASETS, dataset_args=config["data"], separate_test_test=True
    # )

    datamodule = TJDRDataModule_s(data_dir=SEG_PATHS.TJDR, **config["data"])
    datamodule.da_type = None
    datamodule.train_shuffle = False
    datamodule.setup_all()
    datamodule.return_tag(True)

    imgs_ids = [int(f.replace(".png", "").split("_")[-1]) for f in datamodule.train.filenames["image"]]
    imgs_ids = np.array(imgs_ids)
    indices = np.where(imgs_ids > 247)
    datamodule.train.subset(indices)
    test_dataloaders = [datamodule.train_dataloader()]
    if not isinstance(test_dataloaders, list):
        test_dataloaders = [test_dataloaders]

    position = 3
    ftype = "ENCODER"
    # model = load_model_from_checkpoints(
    #     project_name="Retinal Lesions Segmentation",
    #     train_datasets=ALL_DATASETS,
    #     filters={"model_name": "unet-se_resnet50"},
    # )
    model = load_model_from_checkpoints(train_datasets=ALL_DATASETS)
    probe = load_probe_from_checkpoint(model, matching_params={"position": position, "feature_type": ftype})

    loss = probe.criterion

    probe_type = f"{ftype}_P{position}"

    save_folder = f"figures/conversion_examples/TJDR/{probe_type}/"
    os.makedirs(save_folder, exist_ok=True)
    to_datasets = [
        FundusDataset.IDRID,
        FundusDataset.MESSIDOR,
        FundusDataset.RETLES,
        FundusDataset.FGADR,
        FundusDataset.DDR,
    ]
    for to_dataset in to_datasets:
        class_mapping = map_dataset_to_integer(to_dataset)
        to_dataset = to_dataset.name
        target = torch.tensor([class_mapping]).cuda().expand(batch_size)

        pgd = PGD(forward_func=probe, loss_func=loss)
        SAVE_GT = False
        SAVE_TENSORS = False
        SAVE_IMG = True
        for dataloader in tqdm.tqdm(test_dataloaders, total=len(test_dataloaders)):
            from_dataset = "TJDR"
            print("Running on dataset", from_dataset)
            for i, batch in enumerate(dataloader):
                if batch["mask"].sum() < 100_000:
                    continue
                x = batch["image"].cuda()
                roi = batch["roi"].cuda()
                perturbed_input = pgd.perturb(
                    x, target=target, step_size=0.01, step_num=50, radius=5 / 255, targeted=True
                )
                # perturbed_input = perturbed_input * alpha + (1 - alpha) * x
                xmin = x.flatten(-2).min(dim=2).values
                roi = roi.unsqueeze(1)
                perturbed_input = perturbed_input
                pred = probe(perturbed_input)
                pred = batch_integer_to_dataset(pred.argmax(dim=1).cpu().numpy())
                print(pred)
                batch["image"] = perturbed_input
                draw = model.get_grid_with_predicted_mask(
                    batch, ncol=2, alpha=0.5, colors=COLORS, border_alpha=0.9, kernel_size=5
                )
                draw = draw.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                cv2.imwrite(f"{save_folder}from_{from_dataset}_to_{to_dataset}.png", draw)
                if SAVE_GT:
                    batch["image"] = x
                    draw = model.get_grid_with_gt_mask(
                        batch, ncol=2, alpha=0.5, colors=COLORS, border_alpha=0.9, kernel_size=5
                    )

                    draw = draw.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                    cv2.imwrite(f"{save_folder}{from_dataset}_GT.png", draw)

                if SAVE_IMG:
                    batch["image"] = x
                    draw = model.get_grid_with_gt_mask(
                        batch, ncol=2, alpha=0.0, colors=COLORS, border_alpha=0.0, kernel_size=5
                    )

                    draw = draw.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                    cv2.imwrite(f"{save_folder}{from_dataset}_IMG.png", draw)

                if SAVE_TENSORS:
                    tensors = {"image": x, "perturbed_image": perturbed_input, "roi": roi}
                    torch.save(tensors, f"{save_folder}{from_dataset}_to_{to_dataset}_tensors.pth")

                break


if __name__ == "__main__":
    main()
