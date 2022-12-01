# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
from abc import ABC, abstractmethod
import logging
import math
import numpy as np
import torch
import torch.nn as nn

from forger.util.logging import log_tensor

logger = logging.getLogger(__name__)


class PositionalEncoderBase(ABC, nn.Module):
    def __init__(self, resolution):
        super().__init__()
        self.resolution = resolution

    @abstractmethod
    def out_channels(self):
        pass

    @abstractmethod
    def encode_position(self, pos):
        """
        Encodes generated patch start location x, y into a vector.

        @param pos:  int tensor of any shape in range [0 ... resolution-1] (or will apply modulo)
        @return: input_shape x out_channels//2 float tensor
        """
        pass

    def encode_grid(self, start_x, start_y, resolution):
        """
        Encodes a grid of positions of size resolution x resolution.
        Each pixel (r,c)'s encoded position is (start_y + r, start_x + c) modulo encoder.resolution (not grid resolution).

        @param start_x: B int64 tensor of x starts
        @param start_y: B int64 tensor of y starts
        @param resolution: int
        @return: B x out_channels x resolution x resolution grid of encodings float32
        """
        assert len(start_x.shape) == 1
        increment = self.resolution // resolution

        # B x resolution
        shift = torch.arange(0, increment * resolution, increment, dtype=torch.int64).unsqueeze(0).to(start_x.device)
        xs = start_x.unsqueeze(1) + shift
        ys = start_y.unsqueeze(1) + shift

        # B x resolution x C
        encoding_x = self.encode_position(xs % self.resolution)
        encoding_y = self.encode_position(ys % self.resolution)

        # want B x resolution x resolution x C
        return torch.cat([encoding_x.unsqueeze(1).expand(-1, resolution, -1, -1),
                          encoding_y.unsqueeze(2).expand(-1, -1, resolution, -1)], dim=-1).permute(0, 3, 1, 2)

    def forward(self, x, y):
        assert x.shape == y.shape
        x %= self.resolution
        y %= self.resolution
        return torch.cat([self.encode_position(x), self.encode_position(y)], dim=-1)


class GridPositionalEncoder(PositionalEncoderBase):
    def __init__(self, resolution):
        super().__init__(resolution)

    def out_channels(self):
        return 2

    def encode_position(self, pos):
        res = 2 * pos.to(torch.float32) / (self.resolution - 1) - 1.0  # Normalize
        return res.unsqueeze(-1)


class SinusoidalPositionalEncoder(PositionalEncoderBase):
    """
    Encodes patch position using sinusoidal encoding.

    Refer to: https://nbei.github.io/gan-pos-encoding.html (Sec. 3.4 and code)
    """
    def __init__(self, out_channels, resolution):
        assert out_channels % 2 == 0, f'Require out_channels to be even, but got {out_channels}'
        assert out_channels > 0, f'out_channel of positional encoding cannot be 0'
        super().__init__(resolution)
        self.out_ch = out_channels
        encoding_len = self.out_ch // 2

        position = torch.arange(self.resolution).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, encoding_len, 2) * (-math.log(10000.0) / encoding_len))
        pe = torch.zeros((self.resolution, encoding_len))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('posenc', pe)

    def out_channels(self):
        return self.out_ch

    def encode_position(self, pos):
        return self.posenc[pos, :]


class SimpleSinusoidalPositionalEncoder(PositionalEncoderBase):
    """
    Encodes patch position using periodic functions.
    """
    def __init__(self, resolution):
        super().__init__(resolution)
        position = torch.arange(self.resolution).to(torch.float32) / (self.resolution) * 2 * np.pi
        pe = torch.zeros((self.resolution, 2))
        pe[:, 0] = torch.cos(position)
        pe[:, 1] = torch.sin(position)
        self.register_buffer('posenc', pe)

    def out_channels(self):
        return 4

    def encode_position(self, pos):
        return self.posenc[pos, :]


class PositionalEncodingFactory:
    @staticmethod
    def create_from_string(encoding_type: str,
                           resolution: int
                           ):
        if encoding_type == "grid":
            return GridPositionalEncoder(resolution)
        elif encoding_type.startswith('sine'):
            out_channels = int(encoding_type.split(':')[-1])
            return SinusoidalPositionalEncoder(out_channels, resolution)
        elif encoding_type == 'simplesine':
            return SimpleSinusoidalPositionalEncoder(resolution)
        else:
            raise RuntimeError(f'Unknown encoding type {encoding_type}')
