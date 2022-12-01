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
from skimage.io import imsave, imread
import torch
import torch.utils.data

import forger.ui.brush
import forger.util.logging
import forger.metrics.util
import forger.viz.visualize as visualize
from forger.util.logging import log_tensor
import forger.ui.library
import forger.viz.style_transfer
import forger.util.img_proc
import tqdm

logger = logging.getLogger(__name__)


def _read_any_geo(fname):
    """
    @param fname: filename of an image
    @return: 1 x 1 x W' x H' torch float32 image
    """
    img = torch.from_numpy(imread(fname)).to(torch.float32)

    if len(img.shape) == 2:
        img = img.unsqueeze(-1)

    if img.shape[2] == 3:
        img = img[..., :3].mean(dim=2).unsqueeze(-1)
    elif img.shape[2] == 4:
        mean = img[..., :3].mean(dim=2)
        alpha = img[..., 3] / 255
        img = (mean * alpha + 255 * (1 - alpha)).unsqueeze(-1)

    mn = torch.min(img)
    if mn > 0:
        img = img - mn

    mx = torch.max(img)
    if 0 < mx < 255:
        img = img * (255.0 / mx.item())

    img = img.to(torch.uint8).numpy()
    img = (forger.util.img_proc.threshold_img(img, to_float=False).astype(np.float32) * 255).astype(np.uint8)
    return img


def pad_geo(geo, crop_margin):
    geo_padded = np.ones((geo.shape[0] + crop_margin, geo.shape[1] + crop_margin, geo.shape[2]), dtype=np.uint8) * 255
    geo_padded[crop_margin:, crop_margin:, :] = geo
    return geo_padded


def set_colors(color_mode, library, mapper, style_id1, style_id2, brush_options):

    if color_mode in ['1', '2']:
        opts = forger.ui.brush.GanBrushOptions()
        if color_mode == '1':
            library.set_style(style_id1, opts)
        elif color_mode == '2':
            library.set_style(style_id2, opts)
        colors = mapper.get_colors_raw(opts)
        print(colors)
        brush_options.set_color(0, colors[0, :, 0] / 2 + 0.5)
        brush_options.set_color(1, colors[0, :, 1] / 2 + 0.5)
    else:
        color_specs = color_mode.split(';')
        for i, cspec in enumerate(color_specs):
            if len(cspec) > 0:
                rgb = [int(x) for x in cspec.split(',')]
                assert len(rgb) == 3
                brush_options.set_color(i, torch.tensor(rgb, dtype=torch.float32) / 255.0)
                logger.info(f'Set color {i} to {rgb}')

def visualize_crops(geom, crops):
    result = np.concatenate([geom, geom, geom], axis=2)
    for crop in crops:
        y = crop[0]
        x = crop[1]
        width = crop[2]
        result[y:y+width, x, :] = 0
        result[y:y + width, x + width - 1, :] = 0
        result[y, x:x+width, :] = 0
        result[y + width - 1, x:x + width, :] = 0
        result[y:y + width, x, 0] = 255
        result[y:y + width, x + width - 1, 0] = 255
        result[y, x:x + width, 0] = 255
        result[y + width - 1, x:x + width, 0] = 255
    return result


if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='ArtForger user interface.')
    aparser.add_argument('--gan_checkpoint', action='store', type=str, required=True)
    aparser.add_argument('--encoder_checkpoint', action='store', type=str, required=False)
    aparser.add_argument('--output_file_prefix', action='store', type=str, required=True)
    aparser.add_argument('--geom_image', help='Large geometry guidance', type=str)
    aparser.add_argument('--stitching_mode', help='Which patches to paint', type=str, default='all')
    aparser.add_argument('--feature_blending_level', action='store', type=int, default=0)
    aparser.add_argument('--library', help='Which library to use', default='rand100')
    aparser.add_argument('--style_id', action='store', type=str, required=True)
    aparser.add_argument('--style_id2', action='store', type=str, default=None)
    aparser.add_argument('--style_blend_alpha', action='store', type=float, default=0.5)
    aparser.add_argument('--crop_margin', action='store', type=int, default=10)
    aparser.add_argument('--render_mode', action='store', type=str, default='clear')
    aparser.add_argument('--no_uvs_mapping', action='store_true', help='Disable UVS mapping.')
    aparser.add_argument('--color_mode', action='store', type=str, default=None)
    aparser.add_argument('--on_white', action='store_true')
    aparser.add_argument('--debug', action='store_true')
    forger.util.logging.add_log_level_flag(aparser)
    args = aparser.parse_args()

    forger.util.logging.default_log_setup(args.log_level)
    device = torch.device(0)

    engine = forger.ui.brush.PaintEngineFactory.create(
        encoder_checkpoint=args.encoder_checkpoint,
        gan_checkpoint=args.gan_checkpoint,
        device=device)
    library = forger.ui.library.BrushLibrary.from_arg(args.library, z_dim=engine.G.z_dim)

    brush_options = forger.ui.brush.GanBrushOptions()
    brush_options.debug = False
    brush_options.enable_uvs_mapping = not args.no_uvs_mapping
    if args.color_mode is not None:
        set_colors(args.color_mode, library, engine.uvs_mapper, args.style_id, args.style_id2, brush_options)
    if args.style_id2 is None:
        library.set_style(args.style_id, brush_options)
    else:
        library.set_interpolated_style(args.style_id, args.style_id2, args.style_blend_alpha, brush_options)

    patch_width = engine.G.img_resolution
    geom = _read_any_geo(args.geom_image)
    orig_geo_shape = geom.shape
    geom = pad_geo(geom, args.crop_margin)

    # Gets crops and pads geometry
    stitching_crops, geom = forger.viz.style_transfer.generate_stitching_crops(
        geom, patch_width, mode=args.stitching_mode, overlap_margin=args.crop_margin * 2)
    result = np.zeros((geom.shape[0], geom.shape[1], 4), dtype=np.uint8)
    if args.debug:
        imsave('/tmp/geo.png', visualize_crops(geom, stitching_crops))
        raise RuntimeError('stop')

    helper = forger.ui.brush.PaintingHelper(engine)
    helper.make_new_canvas(result.shape[0], result.shape[1], feature_blending=args.feature_blending_level)
    helper.set_render_mode(args.render_mode)

    os.makedirs(os.path.dirname(args.output_file_prefix), exist_ok=True)
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(stitching_crops)), total=len(stitching_crops)):
            y = stitching_crops[i][0]
            x = stitching_crops[i][1]
            brush_options.set_position(x, y)
            geom_patch = 255 - geom[y:y+patch_width, x:x+patch_width, :]  # why is this reverse??

            res, _, meta = helper.render_stroke(
                geom_patch, None, brush_options,
                meta={'x': x, 'y': y, 'crop_margin': args.crop_margin})

            res_y = meta['y']
            res_x = meta['x']
            res_height = res.shape[0]
            res_width = res.shape[1]
            result[res_y:res_y+res_height, res_x:res_x+res_width, :] = res

    if args.on_white:
        alpha = result[..., 3:].astype(np.float32) / 255
        result = result[..., :3].astype(np.float32) * alpha + 255 * (1 - alpha)
        result[..., 3:] = 255
        result = result.clip(0, 255).astype(np.uint8)

    result = result[args.crop_margin:args.crop_margin + orig_geo_shape[0],
             args.crop_margin:args.crop_margin + orig_geo_shape[1], :]

    style_name = args.style_id
    if args.style_id2 is not None:
        style_name += '_%0.1f%s' % (args.style_blend_alpha, args.style_id2)
    output_file = args.output_file_prefix + '_' + args.render_mode + '_' + str(style_name) + '.png'
    imsave(output_file, result)
    logger.info(f'Saved result to: {output_file}')


