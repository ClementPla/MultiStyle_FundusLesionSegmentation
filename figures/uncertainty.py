import os
import sys

import cv2
import numpy as np
import torch
import tqdm
from adptSeg.adaptation.adversarial import PGD
from adptSeg.adaptation.const import _all_datasets, get_class_mapping, get_reverse_class_mapping, trained_probe_path
from adptSeg.adaptation.encoder_wrapper import ModelEncoder
from adptSeg.adaptation.probe import ProbeModule

# from captum.robust import PGD
from fundseg.data.datamodule import FundusSegmentationDatamodule
from fundseg.models.smp_model import SMPModel
from fundseg.utils.colors import COLORS
from fundseg.utils.runs import ALL_DATASETS, Dataset, models_path
from nntools.utils import Config
from torchvision.utils import make_grid

sys.path.append('src/fundseg/')
sys.path.append('src/adptSeg/')

def main():
    batch_size = 4
    config = Config('configs/config.yaml')
    config_data = Config('configs/data_config.yaml')
    config['data']['eval_batch_size'] = batch_size
    model_type = ALL_DATASETS
    datamodule = FundusSegmentationDatamodule(config_data, **config['data'])
    datamodule.persistent_workers = False
    datamodule.setup('test')
    model = SMPModel.load_from_checkpoint(models_path[model_type]).cuda()
    probe_type = 5
    encoder = ModelEncoder(model, encoding_position=5).cuda()
    datasets_probe = _all_datasets
    probe = ProbeModule.load_from_checkpoint(trained_probe_path[probe_type], encoder=encoder,
                                             weights=torch.ones(5), n_classes=5, 
                                             datasets=datasets_probe).cuda()
    
    
    loss = probe.criterion
    
    test_dataloaders = datamodule.test_dataloader()
    save_folder = f'figures/uncertainty/'
    os.makedirs(save_folder, exist_ok=True)
    target = Dataset.RETINAL_LESIONS
    to_dataset = target.name
    class_mapping = get_class_mapping(datasets_probe)
    int_target = class_mapping[target.value]
    target = torch.tensor([int_target]).cuda().expand(batch_size)

    pgd = PGD(forward_func=probe, loss_func=loss)
    for dataloader in test_dataloaders:
        from_dataset = dataloader.dataset.id
        print('Running on dataset', from_dataset)
        for i, batch in enumerate(dataloader):
            
            if batch['mask'].sum() < 100_000:
                continue
            x = batch['image'].cuda()
            probs = []
            for alpha in tqdm.tqdm(np.random.uniform(0., 1.0, 15)):
                perturbed_input = pgd.perturb(x, target=target, step_size=0.005, 
                                              step_num=10, 
                                              radius=0.1, targeted=True)
                perturbed_input = perturbed_input*alpha + (1-alpha)*x 
                batch['image'] = perturbed_input
                prob = model.inference_step(batch).unsqueeze(-1)
                probs.append(prob.cpu())
            probs = torch.cat(probs, dim=-1)
            mean = probs.mean(dim=-1)
            std = probs.std(dim=-1)
            preds = mean.argmax(dim=1)
            batch['mask'] = preds
            
            draw = model.get_grid_with_gt_mask(batch, ncol=2, alpha=0.5, colors=COLORS,
                                                      border_alpha=0.9, kernel_size=3)
            draw = draw.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            cv2.imwrite(f'{save_folder}from_{from_dataset}_to_{to_dataset}_mean.png', draw)
            for i in range(5):
                std_i = std[:, i].unsqueeze(1)
                grid_std = make_grid(std_i, nrow=2, normalize=True, scale_each=True)                
                grid_std = (grid_std.permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
                cv2.imwrite(f'{save_folder}from_{from_dataset}_to_{to_dataset}_std_class{i}.png', grid_std) 
            break
    
    

if __name__=='__main__':
    main()