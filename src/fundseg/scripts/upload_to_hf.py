import argparse
import logging
import os
import warnings

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi
from nntools.utils import Config
from pytorch_lightning import seed_everything

import wandb
from fundseg.data import ALL_DATASETS
from fundseg.data.data_factory import get_datamodule_from_config
from fundseg.models import get_model
from fundseg.utils.hf_api import get_modelcard

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.WARNING)
seed_everything(1234, workers=True)

torch.set_float32_matmul_precision("high")

load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")


def upload(architecture, config):
    project_name = config["logger"]["project"]

    model = get_model(architecture, **config["model"], test_dataset_id=ALL_DATASETS)
    api = wandb.Api()

    runs = api.runs(f"liv4d-polytechnique/{project_name}")
    hfapi = HfApi()
    hfapi.create_repo(
        f"ClementP/fundus-lesions-segmentation-{architecture}", token=HF_TOKEN, exist_ok=True, repo_type="model"
    )

    for r in runs:
        tags = r.tags

        checkpoint_path = f"checkpoints/{project_name}/{r.name}/"
        all_ckpts = os.listdir(checkpoint_path)
        best_model = next(_ for _ in all_ckpts if "epoch" in _)
        ckpt_path = os.path.join(checkpoint_path, best_model)
        state_dict = torch.load(ckpt_path)["state_dict"]
        model.load_state_dict(state_dict)
        model = model.model
        branch_name = "_".join(tags)
        if sorted(tags) == sorted(ALL_DATASETS):
            branch_name = "main"
        hfapi.create_branch(
            f"ClementP/fundus-lesions-segmentation-{architecture}", branch=branch_name, token=HF_TOKEN, exist_ok=True
        )
        model.push_to_hub(f"ClementP/fundus-lesions-segmentation-{architecture}", branch=branch_name, token=HF_TOKEN)

        model_card = get_modelcard(run=r, arch=architecture)
        model_card.push_to_hub(
            f"ClementP/fundus-lesions-segmentation-{architecture}", token=HF_TOKEN, revision=branch_name
        )
    # hfapi.delete_branch(f"ClementP/fundus-lesions-segmentation-{architecture}", branch="IDRID", token=HF_TOKEN)


def main():
    parser = argparse.ArgumentParser(prog="Segmentation Lesions in Fundus")
    parser.add_argument("--model", type=str, help="Model name")

    args = parser.parse_args()

    config_file = "configs/config.yaml"
    config = Config(config_file)

    config["model"] = {}
    config["model"]["lr"] = 0.003
    config["model"]["optimizer"] = "adamw"
    config["model"]["smooth_dice"] = 0.4
    config["model"]["log_dice"] = False

    model_name = args.model

    upload(model_name, config)


if __name__ == "__main__":
    main()
