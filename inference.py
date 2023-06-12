import os.path
from argparse import ArgumentParser

import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler
from transformers import set_seed

from tv.pl_module import ViTPlModule

if __name__ == '__main__':
    args_parser = ArgumentParser()
    args_parser.add_argument('--checkpoint_dir', required=True)
    args = args_parser.parse_args()
    #
    config = yaml.safe_load(open(os.path.join(args.checkpoint_dir, 'hparams.yaml'), 'r'))
    profiler = SimpleProfiler()
    pl_module = ViTPlModule(config, profiler)
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoints/last.ckpt'))
    pl_module.load_state_dict(checkpoint['state_dict'])
    checkpoint_callback = ModelCheckpoint(**config['checkpoint_callback'])
    set_seed(pl_module.config['seed'])
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(
        logger=TensorBoardLogger(**config['logger']),
        **config['trainer'],
        callbacks=[checkpoint_callback, lr_monitor],
        profiler=profiler
    )
    trainer.test(pl_module)
