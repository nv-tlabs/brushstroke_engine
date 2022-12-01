# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)


def generate_stitching_crops(stroke_image, patch_width, mode='all', overlap_margin=15):
    """
    @param stroke_image: numpy H x W x C uint8 image
    @param patch_width:
    @param mode: 'all' or 'full'
    @param overlap_margin:
    @return:
    """
    rwidth = patch_width - overlap_margin * 2

    img_height = stroke_image.shape[0]
    img_width = stroke_image.shape[1]
    nchannels = stroke_image.shape[2]
    assert nchannels in [1, 2, 3, 4], f'Wrong shape {stroke_image.shape}'

    nrows = img_height // rwidth + 1
    ncols = img_width // rwidth + 1
    geom_padded = np.ones((nrows * rwidth + patch_width, ncols * rwidth + patch_width, nchannels),
                           dtype=np.uint8) * 255
    geom_padded[0:img_height, 0:img_width, ...] = stroke_image

    stitching_crops = []
    for r in range(nrows):
        for c in range(ncols):
            y = r * rwidth
            x = c * rwidth
            logger.debug(
                '{},{} --> {}:{},{}:{}'.format(r, c, r * rwidth, r * rwidth + patch_width, c * rwidth, c * rwidth + patch_width))
            geom_input = geom_padded[y:y + patch_width, x:x + patch_width, ...]

            if mode == 'all' or np.sum(geom_input < 0.001) > 10:
                stitching_crops.append((y, x, patch_width, patch_width))

    return stitching_crops, geom_padded



