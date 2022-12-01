# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import logging
import torch

from forger.util.logging import log_tensor
import forger.util.color

logger = logging.getLogger(__name__)

# TIP(aliceli): never import a main function into a module -- bad circular dependencies can result!
#from . import metric_main

# TODO: The channel used as primary weight is INCORRECT. Needs to be fixed in a later MR.
def primary_ratio(triad_data, geom_data):
    assert len(triad_data['uvs'].shape) == 4 and len(geom_data.shape) == 4
    primary_weight = triad_data['uvs'][:, :1, :, :]
    # secondary_weight = triad_data['uvs'][:, 1:2, :, :]
    primary_fg = torch.sum(primary_weight * (1.0 - geom_data), dim=(2, 3))

    return primary_fg / torch.sum(1.0 - geom_data, dim=(2, 3))


def compute_lab_deltas(target_colors, renders, ignore_transparency=False):
    """
    Computes delta between entire image and a color for images in batch.

    @param target_colors: B x 3 torch float32 tensor [0...1]
    @param renders: B x 4 x W x W rendering float32 tensor [0...1]
    @param ignore_transparency: if to composite renders with white canvas or ignore its alpha instead
    @return: B x W x W L2 distance in LAB
    """
    if ignore_transparency:
        renders_rgb = renders[:, :3, ...]
    else:
        # Step 0: account for transparency, assuming white canvas
        alpha = renders[:, 3, ...].unsqueeze(1)
        renders_rgb = alpha * renders[:, :3, ...] + (1 - alpha) * 1.0

    renders_lab = forger.util.color.rgb2lab_anyshape(renders_rgb, rgb_dim=1)
    target_colors_lab = forger.util.color.rgb2lab(target_colors)
    deltas = torch.linalg.norm(renders_lab - target_colors_lab.unsqueeze(-1).unsqueeze(-1), dim=1)
    log_tensor(deltas, 'deltas', logger)
    return deltas


def compute_lab_metrics(target_colors, renders, geom, lab_thresh=10, ignore_transparency=False):
    """
    Computes metrics based on L2 distance in Lab color space.

    @param target_colors: B x 3 torch float32 tensor [0...1] of user-specified colors per stroke
    @param renders: B x 4 x W x W rendering float32 tensor [0...1] rendered stroke images
    @param geom: B x 1 x W x W geometry guidance float32 tensor [0...1] (0 == FG) geometry guidance
    @param lab_thresh: barely noticeable difference in Lab space (10 is used in ColorTriads, SIGG2020)
    @param ignore_transparency: if to composite renders with white canvas or ignore its alpha instead
    @return:
    """
    deltas = compute_lab_deltas(target_colors, renders, ignore_transparency=ignore_transparency)
    masks = (1 - geom).squeeze(1)  # B x W x W
    fg_pixels = torch.sum(masks, dim=(1, 2)).clip(min=1)

    mean_delta = torch.mean(masks * deltas, dim=(1, 2))
    e_percent = torch.sum((deltas > lab_thresh).to(torch.float32) * masks, dim=(1, 2)) / fg_pixels * 100
    return {'LAB_E%': torch.mean(e_percent).item(),
            'LAB_L2': torch.mean(mean_delta).item()}




