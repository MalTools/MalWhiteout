from typing import Dict, List, Any
from warnings import filterwarnings

from hydra.experimental import initialize_config_dir, compose
from omegaconf import DictConfig

from utils.filesystem import get_config_directory


def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore",
                   category=UserWarning,
                   module="pytorch_lightning.utilities.distributed",
                   lineno=45)
    # "Please also save or load the state of the optimizer when saving or loading the scheduler."
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch.optim.lr_scheduler",
                   lineno=216)  # save
    filterwarnings("ignore",
                   category=UserWarning,
                   module="torch.optim.lr_scheduler",
                   lineno=234)  # load


def get_config(
    dataset: str = "example",
    model:str = "detector",
    log_offline: bool = False,
    pb_refresh_rate: int = 1,
    additional_params: Dict[str, Any] = None,
) -> DictConfig:
   
    overrides = [
        f"model={model}",
        f"log_offline={log_offline}",
        f"progress_bar_refresh_rate={pb_refresh_rate}",
    ]
    
    if dataset is not None:
        overrides.append(f"dataset.name={dataset}")
  
    if additional_params is not None:
        for key, value in additional_params.items():
            overrides.append(f"{key}={value}")
    print(overrides)
    with initialize_config_dir(get_config_directory()):
        config = compose("main.yaml", overrides=overrides)
  
    return config

