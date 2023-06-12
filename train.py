from typing import Tuple, Optional

import torch
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler
from transformers import set_seed

from tv.common import freeze_parameters_
from tv.pl_module import ViTPlModule


def fit(pl_module: LightningModule, config: dict) -> Tuple[Trainer, str]:
    set_seed(config['seed'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(**config['checkpoint_callback'])
    trainer = Trainer(
        logger=TensorBoardLogger(**config['logger']),
        **config['trainer'],
        callbacks=[checkpoint_callback, lr_monitor]
    )
    trainer.fit(pl_module)
    return trainer, checkpoint_callback.last_model_path

def train_stage1(config: dict, last_model_path: Optional[str]) -> str:
    print(f'begin stage-1, {config["id"]=}')
    profiler1 = SimpleProfiler()
    pl_module = ViTPlModule(config, profiler1)
    if last_model_path is not None:
        pl_module.load_state_dict(torch.load(last_model_path)['state_dict'])
    _, last_model_path = fit(pl_module, config)
    return last_model_path

def train_stage2(config: dict, last_model_path: Optional[str], num_blocks: int) -> Tuple[Trainer, LightningModule, str]:
    print(f'begin stage-2, {config["id"]=}')
    assert config['model']['args']['unfreeze_vit_blocks_idxes'] == ()
    config['model']['args']['unfreeze_vit_blocks_idxes'] = list(range(config['model']['args']['mask_patches_config']['layer_indices'][0], num_blocks))
    config['model']['args']['mask_patches_config']['hard_mask_training'] = True
    profiler2 = SimpleProfiler()
    pl_module = ViTPlModule(config, profiler2)
    if last_model_path is not None:
        state_dict = torch.load(last_model_path)['state_dict']
        pl_module.load_state_dict(state_dict)
    freeze_parameters_(pl_module.model.blocks.patch_life_predictor)
    trainer, last_model_path = fit(pl_module, config)
    return trainer, pl_module, last_model_path

def train(config: dict):
    last_model_path = None
    num_blocks = ViTPlModule(config, SimpleProfiler()).model.num_blocks
    last_model_path = train_stage1(config, last_model_path)
    trainer, pl_module, last_model_path = train_stage2(config, last_model_path, num_blocks)
    trainer.test(pl_module)
