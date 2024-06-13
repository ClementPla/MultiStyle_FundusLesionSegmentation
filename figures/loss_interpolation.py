import os
import sys

import cv2
import imageio
import numpy as np
import torch
import tqdm
from adptSeg.adaptation.adversarial import PGD
from adptSeg.adaptation.const import class_mapping, trained_probe_path
from adptSeg.adaptation.model_wrapper import ModelEncoder
from adptSeg.adaptation.probe import ProbeModule
from adptSeg.adaptation.utils import get_aptos_dataloader

# from captum.robust import PGD
from fundseg.models.smp_model import SMPModel
from fundseg.utils.colors import COLORS
from fundseg.utils.runs import ALL_DATASETS, Dataset, models_path
from PIL import Image, ImageDraw, ImageFont

sys.path.append("src/fundseg/")
sys.path.append("src/adptSeg/")


def main():
    batch_size = 4
    model_type = ALL_DATASETS
    folder = "interpolation_examples"
    model = SMPModel.load_from_checkpoint(models_path[model_type]).cuda()
    probe_position = 5
    as_regression = probe_position == -1
    encoder = ModelEncoder(model, 
                           encoding_position=probe_position).cuda()
    
    probe = ProbeModule.load_from_checkpoint(trained_probe_path[probe_position], 
                                             encoder=encoder, 
                                             as_regression=as_regression,
                                             weights=torch.ones(5)).cuda()
    loss = probe.criterion
    target = Dataset.IDRID
    to_dataset = target.name

    int_targets =  class_mapping[target.value]
    dataloader = get_aptos_dataloader(batch_size, grade_filter=2)
    pgd = PGD(forward_func=probe, loss_func=loss)
    
    
    os.makedirs(f"figures/{folder}/to_{to_dataset}/", exist_ok=True)
    os.makedirs(f"figures/{folder}/samples/to_{to_dataset}/", exist_ok=True)
    
    for j, batch in enumerate(dataloader):
        
        preds = []
        x = batch["image"].cuda()
        for i, lerp in tqdm.tqdm(enumerate(np.linspace(0, 1, 25))):
            perturbed_input = pgd.perturb(
                x,
                target=int_targets,
                step_size=0.005,
                step_num=25,
                radius=0.1,
                targeted=True,
                interpolation=None,
                as_regression=as_regression,
            )
            batch["image"] = x*(1-lerp) + lerp * perturbed_input
            draw = model.get_grid_with_predicted_mask(
                batch, ncol=1, alpha=0.7, colors=COLORS, border_alpha=1.0, kernel_size=5
            )
            draw = draw.permute(1, 2, 0).cpu().numpy()
            
            pil_im = Image.fromarray(draw)
            draw = ImageDraw.Draw(pil_im)
            draw.text((0, 0), f"Interpolation: {lerp:.2f}")
            draw = np.array(pil_im)
            preds.append(draw)
            # x = perturbed_input
            cv2.imwrite(f"figures/{folder}/samples/to_{to_dataset}/iter_{i}.png", 
                        draw[:, :, ::-1])
            # batch["image"] = x
        imageio.mimsave(f"figures/{folder}/to_{to_dataset}/animation.gif", preds, duration=100.)
        break

if __name__ == "__main__":
    main()
