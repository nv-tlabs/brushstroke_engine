# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

def get_w_stats(num_samples: int,
                z_dim: int,
                mapping_network: torch.nn.Module,
                device='cuda'):
    """
    Given a pretrained Generator, return the average and standard deviation of randomly sampled w
    @param num_samples: The number of samples used to compute the statistics.
    @param z_dim: The number of dimensions of the z latent space.
    @param mapping_network: The mapping network of the pretrained Generator.
    @param device: The device on which the computation is performed.
    """
    logger.info(f"Computing w statistics with {num_samples} samples...")
    z_samples = np.random.RandomState(123).randn(num_samples, z_dim)
    w_samples = mapping_network(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / num_samples) ** 0.5
    return w_avg, w_std