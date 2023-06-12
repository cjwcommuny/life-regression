from __future__ import annotations

from itertools import chain
from statistics import mean, stdev
from typing import List, Dict, Any, Tuple, Sequence

import timm
import torch
from pytorch_lightning.profiler import PassThroughProfiler
from timm.models.vision_transformer import VisionTransformer, Block
from torch import Tensor, nn

from tv.common import freeze_parameters_, unfreeze_parameters_
from tv.models.patch_pruning import PatchMaskerConfig, block_forward, PatchLifePredictor


class CustomBlocks(nn.Module):
    def __init__(
            self,
            patch_life_predictor: PatchLifePredictor,
            #
            blocks: Sequence[Block] & nn.Module,
            num_patches: int,
            layer_indices: Tuple[int, ...],
            num_patches_reserved: Tuple[int, ...],
            temperature: float = 1.0,
            hard_mask_training: bool = False,
            pruning_eval: bool = True,
            profiler=PassThroughProfiler()
    ):
        super().__init__()
        self.patch_life_predictor = patch_life_predictor
        self.hard_mask_training = hard_mask_training
        self.pruning_eval = pruning_eval
        self.num_blocks = len(blocks)
        self.temperature = temperature
        #
        assert len(layer_indices) == len(num_patches_reserved)
        self.base_life = layer_indices[0]
        self.max_life = self.num_blocks
        self.num_patches_per_layers = self.generate_num_patches_per_layers(layer_indices, num_patches_reserved, self.num_blocks)
        self.patch_life_mean, self.patch_life_std = self.compute_patch_life_mean_std(
            layer_indices, num_patches_reserved, num_patches, self.base_life, self.num_blocks )
        #
        self.head_blocks = nn.Sequential(*blocks[:self.base_life])
        self.tail_blocks = nn.ModuleList(blocks[self.base_life:])
        self.plugin_norm_layer = nn.LayerNorm(blocks[self.base_life].norm1.normalized_shape)
        # cache
        self.register_buffer('real_ages', torch.arange(self.base_life, self.max_life, requires_grad=False), persistent=False)
        #
        self.profiler = profiler

    @staticmethod
    def generate_num_patches_per_layers(layer_indices: Tuple[int, ...], num_patches_reserved: Tuple[int, ...], num_layers: int) -> Tuple[int, ...]:
        def get_num_patches(layer_idx: int) -> int:
            i = next(filter(lambda i: layer_indices[i + 1] > layer_idx, range(len(layer_indices) - 1)), len(layer_indices) - 1)
            return num_patches_reserved[i]
        #
        num_patches_per_layer = tuple(get_num_patches(layer_idx) for layer_idx in range(layer_indices[0], num_layers))
        return num_patches_per_layer

    @staticmethod
    def compute_patch_life_mean_std(
            layer_indices: Tuple[int, ...],
            num_patches_reserved: Tuple[int, ...],
            num_patches: int,
            base_layer_index: int,
            num_layers: int
    ) -> Tuple[float, float]:
        num_patches_reserved = tuple(num_patches_reserved)
        prev_num_patches_reserved = [num_patches, *num_patches_reserved]
        num_lifes = [prev - current for prev, current in zip(prev_num_patches_reserved, num_patches_reserved + (0,))]
        layer_indices_normalized = [(i - base_layer_index) / (num_layers - base_layer_index) for i in layer_indices] + [1.0]
        assert len(num_lifes) == len(layer_indices_normalized)
        lifes = list(chain.from_iterable([l for _ in range(n)] for l, n in zip(layer_indices_normalized, num_lifes)))
        mean_life = mean(lifes)
        std_life = stdev(lifes)
        return mean_life, std_life

    def forward_head(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param x: (B, N, d)
        :return:
            - x_at_layer_index: (B, N, d)
            - life_normed: (B, N)
        """
        x_at_layer_index = self.head_blocks(x)
        x_normed = self.plugin_norm_layer(x_at_layer_index)
        life_normed = self.patch_life_predictor.predict_life(x_normed, self.patch_life_mean, self.patch_life_std) # (B, N), element in (0, 1)
        return x_at_layer_index, life_normed

    def forward_tail_with_pruning(self, x: Tensor, life: Tensor) -> Tensor:
        """
        :param x: (B, original_N, d)
        :param life: (B, original_N)
        :return: (B, new_N, d)
        """
        B, original_N, d = x.shape
        life[:, 0] = 1e+5 # cls should be the largest
        indices = life.argsort(dim=-1, descending=True).unsqueeze(-1).expand(B, original_N, d) # (B, N, d)
        x = x.gather(dim=1, index=indices)
        #
        for i, block in enumerate(self.tail_blocks):
            num_patches = self.num_patches_per_layers[i]
            x_pruned = x[:, :num_patches, :]
            weight = torch.ones(*x_pruned.shape[:2], device=x_pruned.device)
            x = block_forward(block, x_pruned, weight)
        return x

    def forward_tail_without_pruning(self, x: Tensor, mask_weights: Tensor) -> Tensor:
        with self.profiler.profile("forward_tail_impl"):
            for i, block in enumerate(self.tail_blocks):
                x = block_forward(block, x, mask_weights[i])
        return x

    def eval_forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        with self.profiler.profile("forward_head"):
            x_at_layer_index, life_normed = self.forward_head(x)
            real_life = life_normed * (self.max_life - self.base_life) + self.base_life
        with self.profiler.profile("forward_tail"):
            if self.pruning_eval:
                cls_last_layer = self.forward_tail_with_pruning(x_at_layer_index, life_normed)
            else:
                mask_weights = PatchLifePredictor.generate_mask_weights_smooth(real_life, self.real_ages, self.temperature) # (T, B, N)
                reserve_mask: Tensor = mask_weights > 0.5
                cls_last_layer = self.forward_tail_without_pruning(x_at_layer_index, reserve_mask)
        return cls_last_layer, real_life

    def train_forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x_at_layer_index, life_normed = self.forward_head(x)
        real_life = life_normed * (self.max_life - self.base_life) + self.base_life
        if self.hard_mask_training:
            cls_last_layer = self.forward_tail_with_pruning(x_at_layer_index, life_normed)
        else:
            mask_weights = PatchLifePredictor.generate_mask_weights_smooth(real_life, self.real_ages, self.temperature) # (T, B, N)
            reserve_mask = mask_weights
            cls_last_layer = self.forward_tail_without_pruning(x_at_layer_index, reserve_mask)
        return cls_last_layer, real_life

class TimmVitAdapter(nn.Module):
    def __init__(
            self,
            num_patches: int,
            model_name: str,
            pretrained: bool = False,
            freeze_vit: bool = False,
            unfreeze_vit_blocks_idxes: Tuple[int] = (),
            mask_patches_config: dict = PatchMaskerConfig()._asdict(),
            base_lr: float = 1e-5,
            extension_lr: float = 1e-3,
            profiler=PassThroughProfiler(),
    ):
        super().__init__()
        self.base_lr = base_lr
        self.extension_lr = extension_lr
        self.model: VisionTransformer = timm.create_model(model_name, pretrained=pretrained)
        self.num_blocks = len(self.model.blocks)
        self.num_heads = self.model.blocks[0].attn.num_heads
        self.dim_per_head = self.model.embed_dim // self.num_heads
        #
        self.masker_config = PatchMaskerConfig(**mask_patches_config)
        self.blocks = CustomBlocks(
            patch_life_predictor=PatchLifePredictor(dim=self.model.embed_dim),
            blocks=self.model.blocks,
            num_patches=num_patches,
            layer_indices=self.masker_config.layer_indices,
            num_patches_reserved=self.masker_config.num_patches_reserved,
            temperature=self.masker_config.temperature,
            hard_mask_training=self.masker_config.hard_mask_training,
            pruning_eval=self.masker_config.pruning_eval,
            profiler=profiler )
        #
        self.freeze_unfreeze(self.model, freeze_vit, unfreeze_vit_blocks_idxes)

    @staticmethod
    def freeze_unfreeze(model: VisionTransformer, freeze_vit: bool, unfreeze_vit_blocks_idxes: Tuple[int,...]):
        if freeze_vit:
            freeze_parameters_(model)
        for vit_block_idx in unfreeze_vit_blocks_idxes:
            unfreeze_parameters_(model.blocks[vit_block_idx])

    def parameters_groups(self) -> List[Dict[str, Any]]:
        return [
            { 'params': self.model.parameters(), 'lr': self.base_lr },
            { 'params': self.blocks.patch_life_predictor.parameters(), 'lr': self.extension_lr }
        ]

    def head_forward(self, x: Tensor) -> Tensor:
        x = self.model.patch_embed(x)
        cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.model.pos_drop(x + self.model.pos_embed)
        return x

    def tail_forward(self, x: Tensor) -> Tensor:
        x = self.model.norm(x)
        x_cls, x_spatial = torch.split(x, [1, x.shape[1] - 1], dim=1)
        x_cls = self.model.head(x_cls.squeeze(1)) # (B, d)
        return x_cls

    @torch.jit.export
    def train_forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.head_forward(x)
        x, life = self.blocks.train_forward(x)
        x = self.tail_forward(x)
        return x, life

    @torch.jit.export
    def eval_forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.head_forward(x)
        x, life = self.blocks.eval_forward(x)
        x = self.tail_forward(x)
        return x, life

    @torch.jit.export
    def predict_forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.eval_forward(x)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        return self.predict_forward(x)