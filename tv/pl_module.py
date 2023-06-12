from functools import cached_property

from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch import nn
from torch.nn import CrossEntropyLoss
from torchmetrics import Accuracy

import tv.models as models
from tv import datasets
from tv.common import configure_adamw_linear_schedule_optimizers, train_dataloader, val_dataloader, test_dataloader, image_classification_running_step, running_epoch_end, train_timm_image_transform, \
    eval_timm_image_transform, Scalar


class ViTPlModule(LightningModule):
    def __init__(self, config: dict, profiler):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        self.model = getattr(models, config['model']['name'])(profiler=profiler, **config['model']['args'])
        self.train_transform = train_timm_image_transform(**config['data_transform'])
        self.val_transform = eval_timm_image_transform(**config['data_transform'])
        self.test_transform = self.val_transform
        self.top1_acc = nn.ModuleDict({
            'train_acc': Accuracy(),
            'val_acc': Accuracy(),
            'test_acc': Accuracy()
        })
        self.top5_acc = nn.ModuleDict({
            'train_acc': Accuracy(top_k=5),
            'val_acc': Accuracy(top_k=5),
            'test_acc': Accuracy(top_k=5)
        })
        self.life_logger = Scalar()
        self.loss_fn = CrossEntropyLoss(**config['loss'])

    def parameter_groups(self):
        return self.model.parameters_groups()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        return configure_adamw_linear_schedule_optimizers(self)

    def train_dataloader(self):
        return train_dataloader(self)

    def val_dataloader(self):
        return val_dataloader(self)

    def test_dataloader(self):
        return test_dataloader(self)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError()

    @cached_property
    def train_dataset(self):
        return getattr(datasets, self.config['train_dataset']['name'])(**self.config['train_dataset']['args'], transform=self.train_transform)

    @cached_property
    def val_dataset(self):
        return getattr(datasets, self.config['val_dataset']['name'])(**self.config['val_dataset']['args'], transform=self.val_transform)

    @cached_property
    def test_dataset(self):
        return getattr(datasets, self.config['test_dataset']['name'])(**self.config['test_dataset']['args'], transform=self.test_transform)

    def training_step(self, batch, batch_idx: int):
        return image_classification_running_step(self, batch, batch_idx, mode='train')

    def validation_step(self, batch, batch_idx: int):
        return image_classification_running_step(self, batch, batch_idx, mode='val')

    def test_step(self, batch, batch_idx: int):
        return image_classification_running_step(self, batch, batch_idx, mode='test')

    def training_epoch_end(self, outputs):
        return running_epoch_end(self, outputs, mode='train')

    def validation_epoch_end(self, outputs):
        return running_epoch_end(self, outputs, mode='val')

    def test_epoch_end(self, outputs):
        return running_epoch_end(self, outputs, mode='test')
