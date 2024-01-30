import os
import sys

import cv2
import tqdm
from fundseg.data.datamodule import FundusSegmentationDatamodule
from fundseg.models.smp_model import SMPModel
from fundseg.utils.colors import COLORS
from fundseg.utils.runs import ALL_DATASETS, Dataset, models_path
from nntools.utils import Config

sys.path.append('src/fundseg/')

def main():
    config = Config('configs/config.yaml')
    config_data = Config('configs/data_config.yaml')
    config['data']['eval_batch_size'] = 4
    model_type = Dataset.IDRID | Dataset.RETINAL_LESIONS
    datamodule = FundusSegmentationDatamodule(config_data, **config['data'])
    datamodule.persistent_workers = False
    datamodule.setup('test')
    model = SMPModel.load_from_checkpoint(models_path[model_type]).cuda()
    if model_type == ALL_DATASETS:
        model_type = 'ALL'
    mname = str(model_type)
    test_dataloaders = datamodule.test_dataloader()
    os.makedirs(f'figures/qualitative_examples/model_{mname}', exist_ok=True)
        
    for dataloader in tqdm.tqdm(test_dataloaders, total=len(test_dataloaders)):
        dname = dataloader.dataset.id
        print('Running on dataset', dname)
        for i, batch in enumerate(dataloader):
            if batch['mask'].sum() < 100_000:
                continue
                
            draw = model.get_grid_with_predicted_mask(batch, ncol=2, alpha=0.5, colors=COLORS,
                                                      border_alpha=0.9, kernel_size=3)
            draw = draw.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
            cv2.imwrite(f'figures/qualitative_examples/model_{mname}/{dname}.png', draw)
            break
    
    


if __name__=='__main__':
    main()