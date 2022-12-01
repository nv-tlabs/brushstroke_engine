# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import argparse
import copy
import os
import logging
import numpy as np
import time
from skimage.io import imsave
import torch
import torch.utils.data
import torchvision

import forger.util.logging
import forger.metrics.util
from forger.util.logging import log_tensor
import forger.train.stitching
import forger.ui.brush
import forger.viz.visualize
from forger.util.torch_data import get_image_data_iterator_from_dataset, get_image_data_iterator

from thirdparty.stylegan2_ada_pytorch.training.dataset import ImageFolderDataset
from thirdparty.stylegan2_ada_pytorch.torch_utils.misc import InfiniteSampler

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='ArtForger user interface.')
    aparser.add_argument('--gan_checkpoint', action='store', type=str, required=True)
    aparser.add_argument('--encoder_checkpoint', action='store', type=str, default=None)
    aparser.add_argument('--output_dir', action='store', type=str, required=True)
    aparser.add_argument('--crop_margin', action='store', type=int, default=10)
    aparser.add_argument('--style_seeds', action='store', type=str, required=True,
                         help='If int, will create this many random styles. '
                              'If csv ints, will use these seeds. '
                              'If file, will load seeds from file.')
    aparser.add_argument('--feature_blending_level', action='store', type=int, default=0)
    aparser.add_argument('--file_suffix', action='store', type=str, default='')
    aparser.add_argument('--seed', action='store', type=int, default=None)
    aparser.add_argument('--clear_mode', action='store', type=int, default=0, help='0-with bg, 1-over white, 2-alpha')
    forger.util.logging.add_log_level_flag(aparser)
    args = aparser.parse_args()
    forger.util.logging.default_log_setup(args.log_level)
    device = torch.device(0)

    default_subdir = 'stitching'
    if args.clear_mode == 1:
        default_subdir += '_clear'
    elif args.clear_mode == 2:
        default_subdir += '_clear_alpha'

    if args.style_seeds == 'default':
        default_subdir = os.path.join('curated', default_subdir)

    if args.output_dir == 'default':
        args.output_dir = os.path.join(
            forger.viz.visualize.get_default_eval_directory(args.gan_checkpoint), default_subdir)
        logger.warning(f'Using default output directory: {args.output_dir}')

    random_state = forger.metrics.util.RandomState(args.seed)
    style_seeds = forger.metrics.util.style_seeds_from_flag(args.style_seeds, args.gan_checkpoint, random_state)

    paint_engine = forger.ui.brush.PaintEngineFactory.create(
        encoder_checkpoint=args.encoder_checkpoint,
        gan_checkpoint=args.gan_checkpoint,
        device=device)

    crop_margin = args.crop_margin
    output_resolution = paint_engine.G.img_resolution
    stitching_image, stitching_patches_raw, stitching_crops = \
        forger.viz.visualize.load_default_stitching_image(output_resolution)
    stitching_image = stitching_image.to(device).permute(2, 0, 1).to(torch.float32) / 255.0 * 2 - 1.0
    stitching_patches = stitching_patches_raw.to(torch.float32) / 255.0
    stitching_patches_features = paint_engine.encoder.encode(stitching_patches[:, :1, ...].to(device))
    stitching_positions = stitching_crops[:, :2].to(torch.int64).to(device)

    # Emulate the way things work in the UI for feature blending
    stitching_patches_raw = stitching_patches_raw.permute(0, 2, 3, 1)  # B x H x W x 3 numpy uint8
    stitching_patches_raw = torch.cat(
        [stitching_patches_raw, torch.ones_like(stitching_patches_raw[..., :1])], dim=-1)
    stitching_patches_raw[..., -1] = 255 - stitching_patches_raw[..., 0]  # Alpha
    stitching_patches_raw = stitching_patches_raw.numpy()

    result_size = [stitching_image.shape[1], stitching_image.shape[2]]

    helper = None
    if args.feature_blending_level > 0 or args.clear_mode > 0:
        helper = forger.ui.brush.PaintingHelper(paint_engine)
        helper.make_new_canvas(result_size[0], result_size[1], feature_blending=args.feature_blending_level)
        helper.set_render_mode('clear' if args.clear_mode > 0 else 'full')

    os.makedirs(args.output_dir, exist_ok=True)
    with torch.no_grad():
        for idx, seed in enumerate(style_seeds):
            logger.info('Evaluating style ({}) {} / {}'.format(seed, idx, len(style_seeds)))

            z = paint_engine.random_style(seed)

            if helper is not None:
                opts = forger.ui.brush.GanBrushOptions()
                opts.set_style(z)
                helper.make_new_canvas(result_size[0], result_size[1])
                res_patches = []
                res_crops = torch.zeros_like(stitching_crops)  # helper modifies crops based on margin
                for i in range(stitching_patches.shape[0]):
                    y = int(stitching_crops[i, 0].item())
                    x = int(stitching_crops[i, 1].item())
                    opts.set_position(x, y)
                    # HACK
                    #opts.set_color(0, torch.tensor([19, 221, 236.0], dtype=torch.float32).to(device) / 128.0 - 1)
                    res, _, meta = helper.render_stroke(
                        stitching_patches_raw[i, ...], None, opts,
                        meta={'x': x, 'y': y, 'crop_margin': crop_margin})
                    res_crops[i, 0] = meta['y']
                    res_crops[i, 1] = meta['x']
                    res_crops[i, 2] = res.shape[0]
                    res_crops[i, 3] = res.shape[1]
                    res_patches.append(
                        torch.from_numpy(res).to(torch.float32).permute(2, 0, 1) / 255 * 2 - 1)
                res_patches = torch.stack(res_patches)
                image = forger.viz.visualize.generate_stitched_image(
                    res_crops, result_size, res_patches, margin=0, clear=(args.clear_mode == 2))  # Note, helper already applies margin
                # Put over white canvas, as this is a visualization utility mostly
                if args.clear_mode < 2:
                    alpha = image[3:, ...] / 2 + 0.5
                    image = torch.ones_like(image[:3, ...]) * (1 - alpha) + image[:3, ...] * alpha
            else:
                image = forger.viz.visualize.generate_stitched_image(
                        stitching_crops, result_size,
                        paint_engine.G(z=z.expand(stitching_patches_features[0].shape[0], -1), c=[],
                                       geom_feature=stitching_patches_features,
                                       positions=stitching_positions,
                                       noise_mode='const', return_debug_data=False), margin=crop_margin)
            image = ((image.permute(1, 2, 0) / 2.0 + 0.5) * 255).cpu().clip(0, 255).to(torch.uint8).numpy()

            imsave(os.path.join(args.output_dir, 'seed%d_stitch_%s.png' % (seed, args.file_suffix)), image)
    print(f'Done: {args.output_dir}')
