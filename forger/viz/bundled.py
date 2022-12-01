# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import os
import torch
from skimage.io import imread
import torchvision

BUNDLED_IMAGES_PATH = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 os.path.pardir, 'images'))

BUNDLED_GEOMETRY_PATH = os.path.join(BUNDLED_IMAGES_PATH, 'spline_patches_curated')

BUNDLED_FONTS_PATH = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 os.path.pardir, 'resources', 'fonts'))


def load_bundled_geometry_image(basename, width=None):
    if not os.path.isdir(BUNDLED_GEOMETRY_PATH):
        raise RuntimeError('Expected images at path {}'.format(BUNDLED_GEOMETRY_PATH))

    path = os.path.join(BUNDLED_GEOMETRY_PATH, basename)
    if not os.path.isfile(path):
        raise RuntimeError('Expected image not found: {}'.format(path))

    return load_geometry_image(path, width)


def load_geometry_image(path, width=None):
    res = torch.from_numpy(imread(path))
    if width is not None:
        res = torchvision.transforms.Resize(width)(res.permute(2, 0, 1)).permute(1, 2, 0)
    return res