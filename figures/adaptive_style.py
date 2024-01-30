import os
import sys

import cv2
import torch
import tqdm
from adptSeg.adaptation.adversarial import PGD
from adptSeg.adaptation.const import get_class_mapping, get_reverse_class_mapping, trained_probe_path
from adptSeg.adaptation.encoder_wrapper import ModelEncoder
from adptSeg.adaptation.probe import ProbeModule

# from captum.robust import PGD
from fundseg.data.datamodule import FundusSegmentationDatamodule
from fundseg.models.smp_model import SMPModel
from fundseg.utils.colors import COLORS
from fundseg.utils.runs import ALL_DATASETS, Dataset, models_path
from nntools.utils import Config

sys.path.append('src/fundseg/')
sys.path.append('src/adptSeg/')

def main():
    batch_size = 4
    config = Config('configs/config.yaml')
    config_data = Config('configs/data_config.yaml')
    config['data']['eval_batch_size'] = batch_size
    model_type = Dataset.IDRID | Dataset.RETINAL_LESIONS
    datamodule = FundusSegmentationDatamodule(config_data, **config['data'])
    datamodule.persistent_workers = False
    datamodule.setup('test')
    model = SMPModel.load_from_checkpoint(models_path[model_type]).cuda()
    probe_type = Dataset.IDRID | Dataset.RETINAL_LESIONS
    encoder = ModelEncoder(model, encoding_position=5).cuda()
    datasets_probe = [Dataset.IDRID, Dataset.RETINAL_LESIONS]
    probe = ProbeModule.load_from_checkpoint(trained_probe_path[probe_type], encoder=encoder,
                                             weights=torch.ones(2), n_classes=2, 
                                             datasets=datasets_probe).cuda()
    
    
    loss = probe.criterion
    
    test_dataloaders = datamodule.test_dataloader()
    save_folder = f'figures/conversion_examples/{probe_type}/'
    os.makedirs(save_folder, exist_ok=True)
    target = Dataset.RETINAL_LESIONS
    to_dataset = target.name
    class_mapping = get_class_mapping(datasets_probe)
    reverse_class_mapping = get_reverse_class_mapping(datasets_probe)
    int_target = class_mapping[target.value]
    target = torch.tensor([int_target]).cuda().expand(batch_size)

    pgd = PGD(forward_func=probe, loss_func=loss)
    alpha = 0.5
    SAVE_GT = True
    for dataloader in tqdm.tqdm(test_dataloaders, total=len(test_dataloaders)):
        from_dataset = dataloader.dataset.id
        print('Running on dataset', from_dataset)
        for i, batch in enumerate(dataloader):
            
            if batch['mask'].sum() < 100_000:
                continue
            x = batch['image'].cuda()
            perturbed_input = pgd.perturb(x, target=target, step_size=0.001, step_num=20, radius=0.015, targeted=True)
            perturbed_input = perturbed_input*alpha + (1-alpha)*x 
            pred = probe(perturbed_input)
            pred = [reverse_class_mapping[i] for i in pred.argmax(dim=1).cpu().numpy()]
            batch['image'] = perturbed_input
            draw = model.get_grid_with_predicted_mask(batch, ncol=2, alpha=0.5, colors=COLORS,
                                                      border_alpha=0.9, kernel_size=3)
            draw = draw.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            cv2.imwrite(f'{save_folder}from_{from_dataset}_to_{to_dataset}.png', draw)
            if SAVE_GT:
                draw = model.get_grid_with_gt_mask(batch, ncol=2, alpha=0.5, colors=COLORS,
                                                      border_alpha=0.9, kernel_size=3)
                
                draw = draw.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
                cv2.imwrite(f'{save_folder}{from_dataset}_GT.png', draw)

            break
    
    


if __name__=='__main__':
    main()