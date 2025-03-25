import os

import cv2
import tqdm
from fundseg.data.data_factory import ALL_DATASETS, get_datamodule_from_config
from fundseg.utils.checkpoints import load_model_from_checkpoints
from fundseg.utils.colors import COLORS
from nntools.utils import Config


def main():
    config = Config("configs/config.yaml")
    config["data"]["eval_batch_size"] = 4

    model = load_model_from_checkpoints(
        "Retinal Lesions Segmentation", train_datasets=ALL_DATASETS, filters={"model_name": "unet-se_resnet50"}
    )

    datamodule = get_datamodule_from_config(
        config["datasets"], training_datasets=ALL_DATASETS, dataset_args=config["data"], separate_test_test=True
    )
    test_dataloaders = datamodule.test_dataloader()
    mname = "_".join(ALL_DATASETS)

    os.makedirs(f"figures/qualitative_examples/model_{mname}", exist_ok=True)

    for dataloader in tqdm.tqdm(test_dataloaders, total=len(test_dataloaders)):
        dname = dataloader.dataset.id
        print("Running on dataset", dname)
        for i, batch in enumerate(dataloader):
            if batch["mask"].sum() < 100_000:
                continue

            draw = model.get_grid_with_predicted_mask(
                batch, ncol=2, alpha=0.5, colors=COLORS, border_alpha=1.0, kernel_size=5
            )
            draw = draw.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            cv2.imwrite(f"figures/qualitative_examples/model_{mname}/{dname}.png", draw)
            break


if __name__ == "__main__":
    main()
