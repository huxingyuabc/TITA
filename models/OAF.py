# Ref: CVPR 2023: Ingredient-oriented Multi-Degradation Leaning for Image Restoration & TPAMI 2024: Frequency-aware Feature Fusion for Dense Image Prediction

import torch
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

try:
    from mmcv.ops.carafe import normal_init, xavier_init, carafe
except ImportError:

    def xavier_init(module: nn.Module,
                    gain: float = 1,
                    bias: float = 0,
                    distribution: str = 'normal') -> None:
        assert distribution in ['uniform', 'normal']
        if hasattr(module, 'weight') and module.weight is not None:
            if distribution == 'uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            else:
                nn.init.xavier_normal_(module.weight, gain=gain)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)

    def carafe(x, normed_mask, kernel_size, group=1, up=1):
            b, c, h, w = x.shape
            _, m_c, m_h, m_w = normed_mask.shape
            assert m_h == up * h
            assert m_w == up * w
            pad = kernel_size // 2
            pad_x = F.pad(x, pad=[pad] * 4, mode='reflect')
            unfold_x = F.unfold(pad_x, kernel_size=(kernel_size, kernel_size), stride=1, padding=0)
            unfold_x = unfold_x.reshape(b, c * kernel_size * kernel_size, h, w)
            unfold_x = F.interpolate(unfold_x, scale_factor=up, mode='nearest')
            unfold_x = unfold_x.reshape(b, c, kernel_size * kernel_size, m_h, m_w)
            normed_mask = normed_mask.reshape(b, 1, kernel_size * kernel_size, m_h, m_w)
            res = unfold_x * normed_mask
            res = res.sum(dim=2).reshape(b, c, m_h, m_w)
            return res

    def normal_init(module, mean=0, std=1, bias=0):
        if hasattr(module, 'weight') and module.weight is not None:
            nn.init.normal_(module.weight, mean, std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias, bias)


def hamming2D(M, N):
    """
    生成二维Hamming窗

    参数：
    - M：窗口的行数
    - N：窗口的列数

    返回：
    - 二维Hamming窗
    """
    # 生成水平和垂直方向上的Hamming窗
    hamming_x = np.hamming(M)
    hamming_y = np.hamming(N)
    # 通过外积生成二维Hamming窗
    hamming_2d = np.outer(hamming_x, hamming_y)
    return hamming_2d


class OAFBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 scale_factor=1,
                 lowpass_kernel=5,
                 highpass_kernel=3,
                 up_group=1,
                 encoder_kernel=3,
                 encoder_dilation=1,
                 compressed_channels=64,
                 use_high_pass=True,
                 use_addition=True,
                 use_multiplication=True,
                 hamming_window=True,  # for regularization, do not matter really
                 **kwargs):
        super().__init__()
        self.scale_factor = scale_factor
        self.highpass_kernel = highpass_kernel
        self.up_group = up_group
        self.encoder_kernel = encoder_kernel
        self.encoder_dilation = encoder_dilation
        self.compressed_channels = compressed_channels
        self.channel_compressor = nn.Conv2d(in_channels, self.compressed_channels, 1)

        self.use_high_pass = use_high_pass
        self.use_addition = use_addition
        self.use_multiplication = use_multiplication
        self.hamming_window = hamming_window

        if self.use_high_pass:
            self.ahpf_generator = nn.Sequential(
                nn.Conv2d(  # AHPF generator
                    self.compressed_channels,
                    highpass_kernel ** 2 * self.up_group * self.scale_factor * self.scale_factor,
                    self.encoder_kernel,
                    padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                    dilation=self.encoder_dilation,
                    groups=1)
                )

        if self.use_addition:
            self.residual_generator = nn.Sequential(
                nn.Conv2d(
                    self.compressed_channels,
                    self.compressed_channels,
                    self.encoder_kernel,
                    padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                    dilation=self.encoder_dilation,
                    groups=1)
                )

        if self.use_multiplication:
            self.weight_generator = nn.Sequential(
                nn.Conv2d(
                    self.compressed_channels,
                    self.compressed_channels,
                    self.encoder_kernel,
                    padding=int((self.encoder_kernel - 1) * self.encoder_dilation / 2),
                    dilation=self.encoder_dilation,
                    groups=1)
                )

        highpass_pad = 0

        if self.hamming_window:
            if self.use_high_pass:
                self.register_buffer('hamming_highpass', torch.FloatTensor(
                    hamming2D(highpass_kernel + 2 * highpass_pad, highpass_kernel + 2 * highpass_pad))[None, None,])
        else:
            if self.use_high_pass:
                self.register_buffer('hamming_highpass', torch.FloatTensor([1.0]))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
        if self.use_high_pass:
            normal_init(self.ahpf_generator, std=0.001)

    def kernel_normalizer(self, mask, kernel, scale_factor=None, hamming=1):
        if scale_factor is not None:
            mask = F.pixel_shuffle(mask, self.scale_factor)
        n, mask_c, h, w = mask.size()
        mask_channel = int(mask_c / float(kernel ** 2))  # group

        mask = mask.view(n, mask_channel, -1, h, w)
        mask = F.softmax(mask, dim=2, dtype=mask.dtype)
        mask = mask.view(n, mask_channel, kernel, kernel, h, w)
        mask = mask.permute(0, 1, 4, 5, 2, 3).view(n, -1, kernel, kernel)
        mask = mask * hamming
        mask /= mask.sum(dim=(-1, -2), keepdims=True)
        mask = mask.view(n, mask_channel, h, w, -1)
        mask = mask.permute(0, 1, 4, 2, 3).view(n, -1, h, w).contiguous()
        return mask

    def forward(self, x):
        feat_list = []
        compressed_x = self.channel_compressor(x)
        if self.use_high_pass:
            filter_hp = self.ahpf_generator(compressed_x)
            filter_hp = self.kernel_normalizer(filter_hp, self.highpass_kernel, hamming=self.hamming_highpass)
            feat_hp = x - carafe(x, filter_hp, self.highpass_kernel, self.up_group, 1)
            feat_list.append(feat_hp)
        else:
            filter_hp = None
        if self.use_addition:
            residual = self.residual_generator(compressed_x)
            feat_add = x + residual
            feat_list.append(feat_add)
        if self.use_multiplication:
            weight = self.weight_generator(compressed_x)
            feat_mul = x * weight
            feat_list.append(feat_mul)

        return feat_list
