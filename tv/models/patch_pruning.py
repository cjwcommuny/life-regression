from typing import NamedTuple, Tuple

import torch
from timm.models.vision_transformer import Attention, Block
from torch import Tensor, nn, softmax


class PatchMaskerConfig(NamedTuple):
    layer_indices: Tuple[int, ...] = ()
    hard_mask_training: bool = False
    pruning_eval: bool = True
    num_patches_reserved: Tuple[int, ...] = ()
    temperature: float = 1

def numerical_stable_weighted_softmax(logits: Tensor, weights: Tensor) -> Tensor:
    """
    :param logits: (*, d)
    :param weights: (*, d)
    :return: (*, d)
    """
    logits = logits - logits.max(dim=-1, keepdim=True).values
    logits_exp = logits.exp()
    logits_exp_weighted = weights * logits_exp
    denominator = logits_exp_weighted.sum(dim=-1, keepdim=True)
    return logits_exp_weighted / denominator

def test_numerical_stable_weighted_softmax():
    logits = torch.rand(3, 5, 7, 11)
    weights = torch.ones(3, 1, 1, 11)
    result1 = numerical_stable_weighted_softmax(logits, weights)
    result2 = softmax(logits, dim=-1)
    assert torch.allclose(result1, result2)

def attention_forward_with_weighted_softmax(self: Attention, x: Tensor, weight: Tensor):
    """
    :param x: (B, N, C)
    :param weight: (B, N)
    :return:
    """
    B, N, C = x.shape
    qkv = self.qkv.forward(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

    attn_before_softmax = (q @ k.transpose(-2, -1)) * self.scale # (B, num_heads, N, N)
    attn = numerical_stable_weighted_softmax(attn_before_softmax, weight.view(B, 1, 1, N))
    attn = self.attn_drop.forward(attn)

    new_x = (attn @ v).transpose(1, 2).reshape(B, N, C)
    new_x = self.proj.forward(new_x)
    new_x = self.proj_drop.forward(new_x)
    return new_x

def block_forward(block: Block, x: Tensor, weights: Tensor) -> Tensor:
    skip_lam = getattr(block, 'skip_lam', 1) # for LV-ViT
    x = x + block.drop_path.forward(attention_forward_with_weighted_softmax(block.attn, block.norm1.forward(x), weights)) / skip_lam
    x = x + block.drop_path.forward(block.mlp.forward(block.norm2.forward(x))) / skip_lam
    return x

class PatchLifePredictor(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.rand(1, dim, dim))
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def smooth_function(spatial_logits: Tensor, mean_spatial_life: float, std_spatial_life: float, epsilon: float = 1e-8) -> Tensor:
        """
        :param spatial_logits: (B, N - 1)
        :param mean_spatial_life: in range (0, 1)
        :param std_spatial_life:
        :param epsilon
        :return: (B, N), s.t. 0 < element < 1
        """
        B, N_minus_1 = spatial_logits.shape
        N = N_minus_1 + 1
        device = spatial_logits.device
        spatial_real_mean = spatial_logits.mean(dim=-1, keepdim=True)
        spatial_real_std = spatial_logits.std(dim=-1, keepdim=True)
        spatial_life_normalized = (spatial_logits - spatial_real_mean) / (spatial_real_std + epsilon) * std_spatial_life + mean_spatial_life
        cls_life_normalized = torch.ones(B, 1, device=device)
        life_normalized = torch.cat((cls_life_normalized, spatial_life_normalized), dim=-1)
        return life_normalized

    def predict_life(self, features: Tensor, mean_spatial_life: float, std_spatial_life: float) -> Tensor:
        """
        :param features: (B, N, d)
        :param mean_spatial_life
        :param std_spatial_life
        :return: (B, N)
        """
        B, N, d = features.shape
        cls = features[:, :1, :]
        spatial = features[:, 1:, :]
        assert cls.shape == (B, 1, d)
        assert spatial.shape == (B, N - 1, d)
        W = self.W.expand(B, d, d)
        spatial_logits = spatial.bmm(W).bmm(cls.transpose(1, 2)).view(B, N - 1)
        life = self.smooth_function(spatial_logits, mean_spatial_life, std_spatial_life)
        assert life.shape == (B, N)
        return life

    @staticmethod
    def generate_mask_weights_smooth(life: Tensor, ages: Tensor, temperature: float = 1) -> Tensor:
        """
        :param life: (B, N), normalized in (T0, T)
        :param ages: (T,), arange(T0, T)
        :param temperature
        :return: (num_life_milestones, B, N), onehot
        """
        B, N = life.shape
        T = ages.shape[0]
        life = life.view(1, B, N)
        ages = ages.view(T, 1, 1)
        distribution = torch.sigmoid(- temperature * (ages - life))
        return distribution
