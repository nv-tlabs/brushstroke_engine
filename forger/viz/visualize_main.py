# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import argparse
import os
import logging
import numpy as np
import time
from skimage.io import imsave
import torch
import torch.utils.data

import forger.util.logging
import forger.metrics.util
import forger.viz.visualize as visualize
from forger.util.logging import log_tensor
from forger.util.torch_data import get_image_data_iterator_from_dataset, get_image_data_iterator

from thirdparty.stylegan2_ada_pytorch.training.dataset import ImageFolderDataset
from thirdparty.stylegan2_ada_pytorch.torch_utils.misc import InfiniteSampler

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='ArtForger user interface.')
    aparser.add_argument('--gan_checkpoint', action='store', type=str, required=True)
    aparser.add_argument('--encoder_checkpoint', action='store', type=str, required=False)
    aparser.add_argument('--output_dir', action='store', type=str, required=True)
    aparser.add_argument('--file_suffix', action='store', type=str, default='')
    aparser.add_argument('--geom_data', help='Training geometry data (directory or zip)', type=str)
    aparser.add_argument('--max_geom_samples', action='store', type=int, default=5)
    aparser.add_argument('--primary_colors', nargs='+', type=str,
                         help='List of R,G,B triplets used in place of random colors.')
    aparser.add_argument('--num_random_colors', type=int, default=1,
                         help='Number of random colors to include.')
    aparser.add_argument('--primary_color_idx', action='store', type=int, default=0,
                         help='Index of the primary color in the user-set colors; legacy.')
    aparser.add_argument('--geom_input_channel', help='Channel to use for geometry conditioning.', type=int, default=1)
    aparser.add_argument('--style_seeds', action='store', type=str, required=True,
                         help='If int, will create this many random styles. '
                              'If csv ints, will use these seeds. '
                              'If file, will load seeds from file.')
    aparser.add_argument('--seed', action='store', type=int, default=None)
    aparser.add_argument('--mode', action='store', type=int, default=1,
                         help='Modes: 0 - many simple patches per style, 1 - one style per file with geom/color grid')
    aparser.add_argument('--use_black_white_secondary_colors_for_custom_colors', action='store_true')
    aparser.add_argument('--randomize_all_colors_for_random_colors', action='store_true')
    aparser.add_argument('--debug', action='store_true')
    forger.util.logging.add_log_level_flag(aparser)
    args = aparser.parse_args()

    forger.util.logging.default_log_setup(args.log_level)
    device = torch.device(0)

    default_subdir = 'styles_mode%d' % args.mode
    if args.style_seeds == 'default':
        default_subdir = os.path.join('curated', default_subdir)

    if args.output_dir == 'default':
        args.output_dir = os.path.join(
            forger.viz.visualize.get_default_eval_directory(args.gan_checkpoint), default_subdir)
        logger.warning(f'Using default output directory: {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)

    random_state = forger.metrics.util.RandomState(args.seed)
    custom_primary_colors = forger.viz.visualize.parse_color_list(args.primary_colors)
    if custom_primary_colors is not None:
        custom_primary_colors = custom_primary_colors.reshape(-1, 1, 3).to(device)
    style_seeds = forger.metrics.util.style_seeds_from_flag(args.style_seeds, args.gan_checkpoint, random_state)

    generator = forger.metrics.util.PaintStrokeGenerator.create(
        encoder_checkpoint=args.encoder_checkpoint,
        gan_checkpoint=args.gan_checkpoint,
        device=device,
        batch_size=args.max_geom_samples,
        random_state=random_state)
    generator.primary_color_idx = args.primary_color_idx
    output_resolution = generator.engine.patch_width
    generator.set_render_mode('full')

    if args.geom_data is None:
        images = visualize.load_default_curated_geometry_images(output_resolution).to(torch.float32)[..., 0] / 255.0
        images = images.unsqueeze(1).to(device)
        log_tensor(images, 'curated_images', logger)
        generator.batch_size = images.shape[0]
        generator.set_new_geom(images)
    else:
        geom_set_cropped = ImageFolderDataset(path=args.geom_data, use_labels=False, max_size=None, xflip=False,
                                              resolution=output_resolution, resize_mode='resize')
        generator.set_geometry_source_from_iterator(
            get_image_data_iterator_from_dataset(geom_set_cropped, args.max_geom_samples, num_workers=1), args.max_geom_samples,
            geom_input_channel=args.geom_input_channel)
        generator.set_new_geom()

    margin = 5
    cwidth = 25
    width = output_resolution
    num_geom = generator.batch_size
    visualize.DEFAULT_MARGIN = margin
    num_colors = args.num_random_colors + 1  # also include original color and random color
    if custom_primary_colors is not None:
        num_colors = custom_primary_colors.shape[0] + num_colors

    for idx, seed in enumerate(style_seeds):
        logger.info('Visualizing style {} / {} ({})'.format(idx, len(style_seeds), seed))
        generator.set_new_styles(generator.get_random_style(seed))
        generator.unset_colors()

        if args.mode == 1:
            viz_img = np.zeros(((1 + num_colors) * (width + margin),
                                (1 + num_geom) * (width + margin),
                                4), dtype=np.uint8)
            visualize.fill_image_row(viz_img, 0, width + margin, generator.geom)

        for cidx in range(num_colors):
            rstart = (width + margin) * (cidx + 1)
            color0 = None
            color1 = None
            color2 = None
            if custom_primary_colors is not None and cidx < custom_primary_colors.shape[0]:
                color0 = custom_primary_colors[cidx, ...].expand(generator.batch_size, -1)
                if args.use_black_white_secondary_colors_for_custom_colors:
                    color1 = torch.zeros_like(color0)
                    color2 = torch.ones_like(color0)
            elif cidx < num_colors - 1:
                color0 = generator.random_color()
                if args.randomize_all_colors_for_random_colors:
                    color0 = generator.random_colors()
                    color1 = generator.random_colors()
                    color2 = generator.random_colors()

            if color0 is None:
                generator.unset_colors()
            else:
                generator.set_new_color(0, color0)
                generator.set_new_color(1, color1)
                generator.set_new_color(2, color2)
                if args.mode == 1:
                    viz_img[rstart:rstart + cwidth, width - cwidth:width, 0:3] = (color0[0] * 255).to(
                        torch.uint8).cpu().numpy()
                    viz_img[rstart:rstart + cwidth, width - cwidth:width, 3] = 255

            # B x 3 x W x W
            render = generator.generate(rgb_on_white_canvas=False) #True)
            assert width == render.shape[-1]
            if args.mode == 0:
                render = (render.detach().permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()[...,:3]
                for b in range(render.shape[0]):
                    bname = 'seed{}_geo{}_color{}.png'.format(seed, b, cidx)
                    imsave(os.path.join(args.output_dir, bname), render[b, ...])
            elif args.mode == 1:
                visualize.fill_image_row(viz_img, rstart, width + margin, render)
            else:
                raise RuntimeError('Unknown mode {}'.format(args.mode))

        if args.mode == 1:
            bname = 'seed{}_stylegrid{}.png'.format(seed, args.file_suffix)
            imsave(os.path.join(args.output_dir, bname), viz_img)

    print(f'Done: {args.output_dir}')

