import os
import sys

import cv2
import numpy as np
import torch
import tqdm
from nntools.utils import Config
from torchvision.utils import make_grid

from adptSeg.adaptation.adversarial import PGD
from adptSeg.adaptation.const import batch_integer_to_dataset, map_dataset_to_integer
from adptSeg.utils.checkpoints import load_probe_from_checkpoint
from fundseg.data.data_factory import ALL_DATASETS, SEG_PATHS, FundusDataset, get_datamodule_from_config
from fundseg.utils.checkpoints import load_model_from_checkpoints

# from captum.robust import PGD
from fundseg.utils.colors import COLORS

sys.path.append("src/fundseg/")
sys.path.append("src/adptSeg/")


def main():
    batch_size = 4
    config = Config("configs/config.yaml")
    config["data"]["eval_batch_size"] = batch_size

    datamodule = get_datamodule_from_config(
        config["datasets"], training_datasets=ALL_DATASETS, dataset_args=config["data"], separate_test_test=True
    )

    position = 3
    ftype = "ENCODER"

    model = load_model_from_checkpoints(train_datasets=ALL_DATASETS)
    probe = load_probe_from_checkpoint(model, matching_params={"position": position, "feature_type": ftype})

    loss = probe.criterion

    test_dataloaders = datamodule.test_dataloader()
    save_folder = f"figures/uncertainty/"
    os.makedirs(save_folder, exist_ok=True)
    target = FundusDataset.RETLES
    to_dataset = target.name
    class_mapping = map_dataset_to_integer(to_dataset)
    target = torch.tensor([class_mapping]).cuda().expand(batch_size)

    pgd = PGD(forward_func=probe, loss_func=loss)
    for dataloader in test_dataloaders:
        from_dataset = dataloader.dataset.id
        if "IDR" not in from_dataset:
            continue
        print("Running on dataset", from_dataset)
        for i, batch in enumerate(dataloader):
            if batch["mask"].sum() < 100_000:
                continue
            x = batch["image"].cuda()
            probs = []
            for alpha in tqdm.tqdm(np.random.uniform(0.0, 1.0, 50)):
                perturbed_input = pgd.perturb(
                    x, target=target, step_size=0.005, step_num=50, radius=10 / 255, targeted=True
                )
                perturbed_input = perturbed_input * alpha + (1 - alpha) * x
                batch["image"] = perturbed_input
                prob = model.inference_step(batch).unsqueeze(-1)
                probs.append(prob.cpu())

            probs = torch.cat(probs, dim=-1)
            mean = probs.mean(dim=-1)
            std = probs.std(dim=-1)

            preds = mean.argmax(dim=1)
            batch["mask"] = preds

            draw = model.get_grid_with_gt_mask(batch, ncol=2, alpha=0.5, colors=COLORS, border_alpha=0.9, kernel_size=3)
            draw = draw.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            cv2.imwrite(f"{save_folder}from_{from_dataset}_to_{to_dataset}_mean.png", draw)

            draw = model.get_grid_with_gt_mask(batch, ncol=2, alpha=0, colors=COLORS, border_alpha=0, kernel_size=3)
            draw = draw.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            cv2.imwrite(f"{save_folder}from_{from_dataset}_to_{to_dataset}_img.png", draw)
            torch.save(std, f"{save_folder}from_{from_dataset}_to_{to_dataset}_std.pt")
            torch.save(mean, f"{save_folder}from_{from_dataset}_to_{to_dataset}_mean.pt")
            # for i in range(5):
            #     std_i = std[:, i].unsqueeze(1)
            #     grid_std = make_grid(std_i, nrow=2, normalize=True, scale_each=True)
            #     grid_std = (grid_std.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            #     cv2.imwrite(f"{save_folder}from_{from_dataset}_to_{to_dataset}_std_class{i}.png", grid_std)
            break


if __name__ == "__main__":
    main()
