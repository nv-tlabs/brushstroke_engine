# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch
import logging

import torch.utils.data

from thirdparty.stylegan2_ada_pytorch.training.dataset import ImageFolderDataset
from thirdparty.stylegan2_ada_pytorch.torch_utils.misc import InfiniteSampler

logger = logging.getLogger(__name__)


def get_image_data_iterator(data_path, batch_size, shuffle=True, return_batch_size=False, regexp=None):
    """

    @param data_path: path to a folder with geometry images
    @param batch_size: int batch size or -1 for a special case where the dataset is tiny
    @param shuffle:
    @return:
    """
    dataset = ImageFolderDataset(
        path=data_path, regexp=regexp, use_labels=False, max_size=None, xflip=False)
    if batch_size == -1:
        batch_size = len(dataset)
        if len(dataset) > 200:
            logger.warning('Setting batch size automatically to a really large value {}'.format(batch_size))

    data_iter = get_image_data_iterator_from_dataset(dataset, batch_size, shuffle=shuffle)
    if return_batch_size:
        return data_iter, batch_size
    return data_iter


def get_image_data_iterator_from_dataset(dataset, batch_size, shuffle=True,
                                         num_workers=1, prefetch_factor=2):
    sampler = InfiniteSampler(dataset=dataset, shuffle=shuffle)
    data_iter = iter(torch.utils.data.DataLoader(
        dataset=dataset, sampler=sampler, batch_size=batch_size,
        pin_memory=True, num_workers=num_workers, prefetch_factor=prefetch_factor, persistent_workers=True))
    return data_iter