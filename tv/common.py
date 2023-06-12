import uuid
from typing import Tuple

import timm
import torch
import transformers
from pytorch_lightning import LightningModule
from timm.data import create_transform, resolve_data_config
from timm.scheduler import CosineLRScheduler
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics import Metric



def configure_adamw_linear_schedule_optimizers(pl_module: 'ViTPlModule'):
    optimizer_type = pl_module.config['optimizer']['type']
    if optimizer_type == 'AdamW':
        optimizer = AdamW(pl_module.parameter_groups(), **pl_module.config['optimizer']['args'])
    else:
        raise NotImplementedError()
    scheduler_type = pl_module.config['scheduler']['type']
    if scheduler_type == 'Linear':
        lr_scheduler = transformers.get_linear_schedule_with_warmup(optimizer, **pl_module.config['scheduler']['args'])
    elif scheduler_type == 'Cosine':
        lr_scheduler = CosineLRScheduler(optimizer, **pl_module.config['scheduler']['args'])
    else:
        raise NotImplementedError()
    return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                "interval": pl_module.config['scheduler']['interval'],
            }
        }

def train_dataloader(pl_module: LightningModule) -> DataLoader:
    return DataLoader(pl_module.train_dataset, **pl_module.config['train_dataloader'])


def val_dataloader(pl_module: LightningModule) -> DataLoader:
    return DataLoader(pl_module.val_dataset, **pl_module.config['val_dataloader'])


def test_dataloader(pl_module: LightningModule) -> DataLoader:
    return DataLoader(pl_module.test_dataset, **pl_module.config['test_dataloader'])

def clamp_life(life: Tensor, base_life: int, max_life: int) -> Tensor:
    return torch.clamp(life, min=base_life, max=max_life).mean()

def image_classification_running_step(pl_module: LightningModule, batch, batch_idx: int, mode: str) -> Tensor:
    image, labels = batch
    if mode == 'train':
        logits, original_life = pl_module.model.train_forward(image)
    elif mode == 'eval':
        logits, original_life = pl_module.model.eval_forward(image)
    else:
        logits, original_life = pl_module.model.predict_forward(image)
    life = original_life.mean()
    pl_module.life_logger(clamp_life(original_life, pl_module.model.blocks.base_life, pl_module.model.blocks.max_life))
    top1_acc = pl_module.top1_acc[f'{mode}_acc'](logits, labels)
    top5_acc = pl_module.top5_acc[f'{mode}_acc'](logits, labels)
    cls_loss = pl_module.loss_fn(logits, labels)
    loss = cls_loss
    pl_module.log(f'steps/{mode}-cls_loss', cls_loss)
    pl_module.log(f'steps/{mode}-avg_life', life)
    pl_module.log(f'steps/{mode}-loss', loss)
    pl_module.log(f'steps/{mode}-top1_acc', top1_acc)
    pl_module.log(f'steps/{mode}-top5_acc', top5_acc)
    return loss

def running_epoch_end(pl_module: LightningModule, outputs, mode: str):
    top1_acc = pl_module.top1_acc[f'{mode}_acc'].compute()
    top5_acc = pl_module.top5_acc[f'{mode}_acc'].compute()
    life = pl_module.life_logger.compute()
    pl_module.log(f'epochs/{mode}-top1_acc', top1_acc)
    pl_module.log(f'epochs/{mode}-top5_acc', top5_acc)
    pl_module.log(f'epochs/{mode}-life', life)

def train_timm_image_transform(**kwargs):
    return timm_image_transform(is_training=True, **kwargs)

def eval_timm_image_transform(**kwargs):
    return timm_image_transform(is_training=False, **kwargs)

# follow https://github.com/raoyongming/DynamicViT/blob/master/main_dynamic_vit.py
def timm_image_transform(
        is_training: bool,
        model_name: str
):
    model = timm.create_model(model_name, pretrained=True)
    config = resolve_data_config({}, model=model)
    transform = create_transform(is_training=is_training, **config)
    return transform

def freeze_parameters_(model: nn.Module):
    for param in model.parameters(recurse=True):
        param.requires_grad = False

def unfreeze_parameters_(model: nn.Module):
    for param in model.parameters(recurse=True):
        param.requires_grad = True

class Scalar(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("scalar", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, scalar):
        if isinstance(scalar, torch.Tensor):
            scalar = scalar.detach().to(self.scalar.device)
        else:
            scalar = torch.tensor(scalar).float().to(self.scalar.device)
        self.scalar += scalar
        self.total += 1

    def compute(self):
        return self.scalar / self.total

def generate_id() -> str:
    return f'{uuid.uuid1().hex}'[:8]

def get_num_patches_reserved(keep_rate: float) -> Tuple[int, int, int]:
    if keep_rate == 0.7:
        return 140, 100, 72
    elif keep_rate == 0.8:
        return 159, 129, 105
    elif keep_rate == 0.9:
        return 179, 163, 148
    else:
        raise NotImplementedError()
