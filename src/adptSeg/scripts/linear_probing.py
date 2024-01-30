import os
import sys

import torch
from adptSeg.adaptation.encoder_wrapper import ModelEncoder
from adptSeg.adaptation.probe import ProbeModule
from fundseg.data.datamodule import FundusSegmentationDatamodule
from fundseg.models.smp_model import SMPModel
from fundseg.utils.runs import ALL_DATASETS, Dataset, models_path
from nntools.utils import Config
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb

# This hack is needed to allow deserializing the model
sys.path.append('src/fundseg/')



def main():
    config_file = "configs/config.yaml"
    config = Config(config_file)
    config_data = Config("configs/data_config.yaml")
    config['data']['batch_size'] = 64
    config['data']['eval_batch_size'] = 64
    config['data']['num_workers'] = 8
    datasets = Dataset.IDRID | Dataset.RETINAL_LESIONS
    model_name = 'unet_se_resnet50'
    
    encoder_positions = [5]
    model = SMPModel.load_from_checkpoint(models_path[datasets])
    as_regression = False
    for encoder_position in encoder_positions:
        logger = WandbLogger(project="Probing-Lesions-Segmentation",
                            config={'encoder_position': encoder_position,
                                    'as_regression': as_regression,
                                    'model_name': model_name})
        
        
        fundus_datamodule = FundusSegmentationDatamodule(
            datasets=datasets, data_config=config_data, 
            disable_cropping=True,
            disable_data_aug=True,
            return_tag=True, **config["data"]
        )
        fundus_datamodule.setup('fit')
        weights = []
        dataset = fundus_datamodule.train
        for d in dataset.datasets:
            weights.append(len(d))
        weights = torch.Tensor(weights)
        
        weights = weights.sum() / (weights*len(weights))
        modelEncoder = ModelEncoder(model, encoding_position=encoder_position)
        
        model = ProbeModule(modelEncoder, n_classes=len(datasets), weights=weights, 
                            as_regression=as_regression)
        
        run_name = wandb.run.name
        checkpoint = ModelCheckpoint(monitor="MulticlassAccuracy",
            mode="max",
            save_last=True,
            auto_insert_metric_name=True,
            save_top_k=1,
            dirpath=os.path.join("checkpoints_probing", run_name),)
        
        trainer = Trainer(accelerator='gpu', devices='auto', 
                        max_epochs=50, 
                        callbacks=[checkpoint],
                        log_every_n_steps=30,
                        logger=logger,
                        check_val_every_n_epoch=1)
        
        trainer.fit(model, fundus_datamodule)
        trainer.test(model, dataloaders=fundus_datamodule.val_dataloader(), 
                    ckpt_path='best')
        wandb.finish()
        

if __name__=="__main__":
    main()