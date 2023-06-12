from typing import List

from timm import create_model
from torch import nn

from tv.common import get_num_patches_reserved
from tv.models import TimmVitAdapter


def get_model(name: str, **kwargs) -> nn.Module:
    if name == 'ours':
        return get_ours(**kwargs)
    else:
        return get_backbone(name)

def get_backbone(name: str):
    return create_model(name, img_size=(224, 224))

def get_ours(backbone_name: str, keep_rate: float):
    num_patches_reserved = get_num_patches_reserved(keep_rate)
    #
    mask_patches_config = {
        'layer_indices': (4, 7, 10),
        'hard_mask_training': False,
        'pruning_eval': True,
        'num_patches_reserved': num_patches_reserved,
        'temperature': 1.0,
    }
    model = TimmVitAdapter(
        num_patches=197,
        model_name=backbone_name,
        mask_patches_config=mask_patches_config )
    return model

def get_model_configs() -> List[dict]:
    return [
        {
            'name': 'deit_small_patch16_224'
        },
        {
            'name': 'deit_base_patch16_224'
        },
        {
            'name': 'vit_tiny_patch16_224'
        },
        {
            'name': 'vit_small_patch16_224'
        },
        {
            'name': 'vit_base_patch16_224'
        },
        {
            'name': 'ours',
            'backbone_name': 'deit_small_patch16_224',
            'keep_rate': 0.7
        },
        {
            'name': 'ours',
            'backbone_name': 'deit_small_patch16_224',
            'keep_rate': 0.8
        },
        {
            'name': 'ours',
            'backbone_name': 'deit_small_patch16_224',
            'keep_rate': 0.9
        },
        {
            'name': 'ours',
            'backbone_name': 'deit_base_patch16_224',
            'keep_rate': 0.7
        },
        {
            'name': 'ours',
            'backbone_name': 'vit_base_patch16_224',
            'keep_rate': 0.7
        },
        {
            'name': 'ours',
            'backbone_name': 'vit_small_patch16_224',
            'keep_rate': 0.7
        },
        {
            'name': 'ours',
            'backbone_name': 'vit_tiny_patch16_224',
            'keep_rate': 0.7
        }
    ]