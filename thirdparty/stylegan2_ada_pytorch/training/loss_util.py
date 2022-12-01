# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import copy
try:
    import cv2
except Exception:
    print('Failed to import cv2')

import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.warn("this module is deprecated", DeprecationWarning,
              stacklevel=2)

"""
Some commonly used classes for losses
"""

class gradientLayer(nn.Module):
    def __init__(self,
                 edge_ksize=3):
        super(gradientLayer, self).__init__()
        sobel_x = cv2.getDerivKernels(1, 0, ksize=edge_ksize, normalize=True)
        sobel_x_kernel = np.outer(sobel_x[0], sobel_x[1])

        self.edge_x_conv = nn.Conv2d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=edge_ksize,
                                     stride=1,
                                     padding=(edge_ksize - 1) // 2,
                                     bias=False,
                                     padding_mode='reflect')
        self.edge_y_conv = copy.deepcopy(self.edge_x_conv)
        sobel_x_kernel = np.expand_dims(sobel_x_kernel, axis=(0, 1))
        sobel_y_kernel = np.transpose(sobel_x_kernel, (0, 1, 3, 2))
        self.edge_x_conv.weight = nn.Parameter(torch.from_numpy(sobel_x_kernel))
        self.edge_y_conv.weight = nn.Parameter(torch.from_numpy(sobel_y_kernel))

    def forward(self, img):
        assert img.shape[1] == 1, ''
        x = torch.abs(self.edge_x_conv(img))
        x += torch.abs(self.edge_y_conv(img))
        return x


class RGB2Gray(nn.Module):
    def __init__(self):
        super().__init__()
        self.channel_weight = [0.30, 0.59, 0.11]

    def forward(self, img):
        assert img.shape[1] == 3, "Could not convert from RGB to Gray because image is not RGB"
        # Following cv.RGB2GRAY
        x = torch.zeros((img.shape[0], 1, img.shape[2], img.shape[3]), device='cuda')
        for i_channel, weight in enumerate(self.channel_weight):
            x += weight * img[:, i_channel:i_channel+1, :, :]
        return x
