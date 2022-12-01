# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import logging
import torch
import torch.nn as nn

import forger.experimental.autoenc.simple_autoencoder as simple_autoencoder
import forger.experimental.autoenc.ae_conv as ae_conv

logger = logging.getLogger(__name__)


def create_autoencoder_from_checkpoint(encoder_checkpt: str = None, loaded_encoder_checkpt=None):
    assert (encoder_checkpt is None) != (loaded_encoder_checkpt is None)

    if encoder_checkpt is not None:
        logger.info(f'Loading autoencoder...{encoder_checkpt}')
        loaded_encoder_checkpt = torch.load(encoder_checkpt)
    else:
        logger.info(f'Loading autoencoder from pre-loaded pkl...')
    geom_autoencoder, _ = create_autoencoder(loaded_encoder_checkpt['args'])
    geom_autoencoder.load_state_dict(loaded_encoder_checkpt['model_state'])
    return geom_autoencoder


def create_autoencoder(args):
    if args.model_name == 'sauto':
        model = simple_autoencoder.model_from_flags(args)
        model_summary = simple_autoencoder.summary_from_flags(args)
    elif args.model_name == 'conv':
        model = ae_conv.model_from_flags(args)
        model_summary = ae_conv.summary_from_flags(args)
    else:
        raise RuntimeError(f'Unknown model: {args.model_name}')

    # Set the type of input preprocessing encoder was trained with
    if hasattr(args, 'preproc_type'):
        model.set_preprocessing(args.preproc_type)
    return model, model_summary


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)