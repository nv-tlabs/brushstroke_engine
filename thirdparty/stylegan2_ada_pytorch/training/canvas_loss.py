# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import logging
import lpips
import random
import torch
import torch.nn as nn

from . import loss_util
import warnings
warnings.warn("this module is deprecated", DeprecationWarning,
              stacklevel=2)

logger = logging.Logger(__name__)


class CanvasLoss:
    """
    CanvasLoss is imposed on the 'canvas' component of the new canvas forger model.
    """
    def __init__(self, internal_type: str, weight: float, resolution: int, device='cuda'):
        """
        @param internal_type: The name string of the internal loss function.
                            See the help string of --canvas_loss_type for more information.
        @param weight: The weight of CanvasLoss.
        @param resolution: The width (dimension along axis 2) of the canvas tensor
        @param device: The device where
        """

        self.gray = True
        if internal_type == 'grad':
            self.loss_func = gradientSumLoss(device=device)
        elif internal_type == 'std':
            self.loss_func = StdLoss
        elif internal_type == 'lpips':
            self.gray = False
            self.loss_func = LPIPSLoss(canvas_resolution=resolution)
        else:
            raise RuntimeError(f"Unsupported internal type for CanvasLoss: {internal_type}")

        if weight is None:
            raise RuntimeError("The weight for CanvasLoss is not set.")
        logger.debug("Canvas loss got resolution", resolution)
        self.weight = weight
        self.rgb2gray = loss_util.RGB2Gray()

    def compute(self, canvas):
        """
        @param canvas: torch.Tensor of shape [N, 3, W, H]
        @return
        """
        if self.weight == 0.0:
            return 0.0
        input_canvas = canvas
        if self.gray:
            input_canvas = self.rgb2gray(canvas)
        return self.loss_func(input_canvas) * self.weight


class gradientSumLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.sobel_layer = loss_util.gradientLayer(edge_ksize=3)
        self.sobel_layer.requires_grad_(False).to(device)

    def forward(self, gray_canvas):
        return torch.sum(self.sobel_layer(gray_canvas))


def StdLoss(gray_canvas):
    assert gray_canvas.shape[1] == 1
    sample_std = torch.std(gray_canvas, dim=(1, 2, 3))
    return torch.mean(sample_std)


class LPIPSLoss(nn.Module):
    def __init__(self, canvas_resolution, device='cuda'):
        super().__init__()
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)
        assert canvas_resolution > 0
        half_res = canvas_resolution // 2
        self.quad_half = [(0, half_res), (half_res, canvas_resolution)]

    def forward(self, rgb_canvas):
        # TODO: Use randomly-cropped patches
        assert rgb_canvas.shape[1] == 3
        quad1_key = random.randint(0, 3)
        quad2_key = random.randint(0, 3)
        quad2_key += int(quad1_key == quad2_key)
        quad2_key %= 4

        patch_1 = rgb_canvas[:, :,
                  self.quad_half[bool(quad1_key & 1)][0]:self.quad_half[bool(quad1_key & 1)][1],
                  self.quad_half[bool(quad1_key & 2)][0]:self.quad_half[bool(quad1_key & 2)][1]]
        patch_2 = rgb_canvas[:, :,
                  self.quad_half[bool(quad2_key & 1)][0]:self.quad_half[bool(quad2_key & 1)][1],
                  self.quad_half[bool(quad2_key & 2)][0]:self.quad_half[bool(quad2_key & 2)][1]]
        return torch.mean(self.loss_fn_alex(patch_1, patch_2))
