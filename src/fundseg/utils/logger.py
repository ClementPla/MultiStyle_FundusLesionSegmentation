import wandb
from pytorch_lightning.loggers import WandbLogger


def init_logger(logger_config, code_dir='.', tags=None, **tracked_params):
    wandb_logger = WandbLogger(**logger_config,
                               settings=wandb.Settings(code_dir=code_dir),
                               config=tracked_params,
                               tags=tags)
    
    return wandb_logger

def check_if_run_already_started(project_name, discriminating_value, discriminating_key='model_name', return_runname=False,
                                 failed_state=False):
    api = wandb.Api()
    project = api.project(project_name)
    runs = api.runs(path='/'.join(project.path))
    try:
        runs = list(runs)
    except ValueError:
        return False
    for r in runs:
        if discriminating_key not in r.config:
            continue
        if r.config[discriminating_key] == discriminating_value:
            if r.state == 'running' or r.state == 'finished' or failed_state:
                print(f'Run {r.id} already exists and is {r.state}.')
                if return_runname:
                    return r.id, r.name
                return r.id
    return False


    