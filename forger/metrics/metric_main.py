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
import pickle
from skimage.io import imsave
import torch
import torch.utils.data
import torchvision

from . import geom_metric
from . import color_metric
import forger.util.logging
import forger.metrics.util
from forger.util.logging import log_tensor
import forger.train.stitching
from forger.util.torch_data import get_image_data_iterator_from_dataset, get_image_data_iterator
import forger.ui.library

from thirdparty.stylegan2_ada_pytorch.training.dataset import ImageFolderDataset
from thirdparty.stylegan2_ada_pytorch.torch_utils.misc import InfiniteSampler

logger = logging.getLogger(__name__)


def update_sum_dict(accum_dict, updates):
    for k, v in updates.items():
        if k not in accum_dict:
            accum_dict[k] = v
        else:
            accum_dict[k] = accum_dict[k] + v


def normalize_sum_dict(accum_dict, denominator):
    for k, v in accum_dict.items():
        accum_dict[k] = v / denominator


def ordered_dict_values(accum_dict, ordered_keys, to_string=True):
    assert set(ordered_keys) == set(accum_dict.keys()), '{} vs {}'.format(ordered_keys, accum_dict.keys())

    def frmt_str(length):
        return '{:>' + ('%d' % length) + '.4f}'

    if to_string:
        return [frmt_str(len(k)).format(accum_dict[k]) for k in ordered_keys]
    else:
        return [accum_dict[k] for k in ordered_keys]


def to_file_line(val_list, separator=' ', do_strip=True, min_length=8):
    def _strip(val):
        return ''.join(val.split()).replace(separator, '')

    if do_strip:
        str_vals = [_strip('{}'.format(v)) for v in val_list]
    else:
        str_vals = ['{}'.format(v) for v in val_list]

    if min_length > 0:
        ft = '{:>' + ('%d' % min_length) + '}'
        str_vals = [ft.format(s) for s in str_vals]
    return separator.join(str_vals) + '\n'


def stitching_metric_loop(random_state, geom_iterator, geom_input_channel, geom_enc, G, stitcher, batch_size, num_batches, device):
    output_resolution = G.img_resolution
    crop_transform = torchvision.transforms.RandomCrop(output_resolution)
    summary_losses = {}
    nprocessed = 0
    for b in range(num_batches):
        z = random_state.random_tensor((batch_size, G.z_dim)).to(device)
        c = []

        geom, _ = next(geom_iterator)
        geom = geom[:, geom_input_channel:geom_input_channel + 1, :, :].to(device).to(torch.float32) / 255.0

        crop1 = crop_transform.get_params(geom, (output_resolution, output_resolution))
        crop2 = stitcher.gen_overlapping_square_crop(geom.shape[-1], crop1)

        geom1 = torchvision.transforms.functional.crop(geom, *crop1)
        geom2 = torchvision.transforms.functional.crop(geom, *crop2)

        enc1 = geom_enc.encode(geom1)
        enc2 = geom_enc.encode(geom2)

        res = stitcher.generate_with_stitching(G, z, c, enc1, enc2, crop1, crop2)
        batch_losses = geom_metric.compute_stitching_metrics(res, margin=stitcher.margin)
        update_sum_dict(summary_losses, batch_losses)
        nprocessed += 1

    normalize_sum_dict(summary_losses, float(nprocessed))
    return summary_losses


def paint_engine_metric_loop(generator,
                             style_generator,
                             nbatches_per_style,
                             fullres_geom_iterator,
                             stitcher,
                             geom_input_channel,
                             eval_output_dir=None,
                             debug=False,
                             files_prefix=''):
    """
    Metric loop specific to metrics calculated for this project.

    @return:
    """
    generator.set_render_mode('clear')
    style_ofile = None
    summary_ofile = None
    if eval_output_dir is not None:
        os.makedirs(args.eval_output_dir, exist_ok=True)
        style_ofile = open(os.path.join(eval_output_dir, '%sstyle_metrics.txt' % files_prefix), 'w')
        summary_ofile = open(os.path.join(eval_output_dir, '%ssummary_metrics.txt' % files_prefix), 'w')

    debug_dir = None
    if eval_output_dir is not None and debug:
        debug_dir = os.path.join(eval_output_dir, 'debug')
        os.makedirs(debug_dir, exist_ok=True)

    output_resolution = generator.engine.G.img_resolution
    crop_transform = torchvision.transforms.RandomCrop(output_resolution)

    ordered_keys = None
    summary_losses = {}
    nprocessed = 0
    style_ws = {}
    total_styles = len(style_library.get_style_ids())
    with torch.no_grad():
        for style_id in style_library.get_style_ids():
            seed_str = '{:<15}'.format(style_id)
            logger.info('Evaluating style ({}) {} / {}'.format(style_id, nprocessed, total_styles))
            style_library.set_style(style_id, generator.brush_options)

            style_losses = {}
            if generator.brush_options.style_ws is not None:
                style_ws[seed_str] = generator.brush_options.style_ws
            else:
                style_ws[seed_str] = generator.engine.G.mapping(
                    generator.brush_options.style_z.to(generator.engine.device), None).detach().cpu()

            for bidx in range(nbatches_per_style):
                colors = generator.random_colors()
                generator.set_new_geom()
                generator.set_new_primary_color(colors)
                render = generator.generate()

                # TODO(mshugrina): figure out how to treat transparency by default
                batch_losses = color_metric.compute_lab_metrics(colors, render, generator.geom, ignore_transparency=False)
                batch_losses.update(geom_metric.compute_transparency_metrics(render, generator.geom))

                # Compute within-image background uniformity for custom colors
                res, debug_img = geom_metric.compute_uniform_bg_lpips_metric(
                    render, generator.geom, same_style=False, return_debug=(bidx == 0 and debug_dir is not None),
                    key_suffix='multicolor')
                if debug_img is not None and debug_dir is not None:
                    imsave(os.path.join(debug_dir, '{}geom_debug_multi_{}.png'.format(files_prefix, seed_str)), debug_img)
                batch_losses.update(res)

                # Compute between-image background uniformity, with no color variations among batch elements
                generator.unset_colors()
                render = generator.generate()
                res, debug_img = geom_metric.compute_uniform_bg_lpips_metric(
                    render, generator.geom, same_style=True, return_debug=(bidx == 0 and debug_dir is not None))
                if debug_img is not None and debug_dir is not None:
                    imsave(os.path.join(debug_dir, '{}geom_debug_{}.png'.format(files_prefix, seed_str)), debug_img)
                batch_losses.update(res)
                batch_losses.update(geom_metric.compute_lpips_across_geo(render))

                # Also compute stitching loss
                geom, _ = next(fullres_geom_iterator)
                geom = geom[:, geom_input_channel:geom_input_channel + 1, :, :].to(torch.float32) / 255.0

                crop1 = crop_transform.get_params(geom, (output_resolution, output_resolution))
                crop2 = stitcher.gen_overlapping_square_crop(geom.shape[-1], crop1)

                geom1 = torchvision.transforms.functional.crop(geom, *crop1)[0:generator.batch_size, ...].to(generator.engine.device)
                geom2 = torchvision.transforms.functional.crop(geom, *crop2)[0:generator.batch_size, ...].to(generator.engine.device)

                enc1 = generator.engine.encoder.encode(geom1)
                enc2 = generator.engine.encoder.encode(geom2)
                del geom1
                del geom2

                # TODO: fix for ws styles
                if generator.current_styles() is not None:
                    res = stitcher.generate_with_stitching(generator.engine.G, generator.current_styles(),
                                                           [], enc1, enc2, crop1, crop2)
                    batch_losses.update(geom_metric.compute_stitching_metrics(res, margin=stitcher.margin))
                update_sum_dict(style_losses, batch_losses)

            if ordered_keys is None:
                ordered_keys = sorted(style_losses.keys())

                if style_ofile is not None:
                    header_str = 'SEED            ' + to_file_line(ordered_keys)
                    style_ofile.write(header_str)

                if summary_ofile is not None:
                    summary_ofile.write(to_file_line(ordered_keys))

            normalize_sum_dict(style_losses, float(nbatches_per_style))
            update_sum_dict(summary_losses, style_losses)
            style_line = seed_str + ' ' + to_file_line(ordered_dict_values(style_losses, ordered_keys), do_strip=False)
            if style_ofile is not None:
                style_ofile.write(style_line)
                style_ofile.flush()
            logger.debug(('%d: ' % nprocessed) + style_line)
            nprocessed += 1

    normalize_sum_dict(summary_losses, float(nprocessed))

    if style_ofile is not None:
        style_ofile.close()
    if summary_ofile is not None:
        summary_ofile.write(to_file_line(ordered_dict_values(summary_losses, ordered_keys), do_strip=False))
        summary_ofile.close()

    if eval_output_dir is not None:
        ws_pkl_file = os.path.join(eval_output_dir, '%sstyle_ws.pkl' % files_prefix)

        with open(ws_pkl_file, 'wb') as f:
            pickle.dump(style_ws, f)
            logger.info(f'Wrote styles to pkl: {ws_pkl_file}')
    return summary_losses


def summary_losses_to_file(fname, losses, step=None, do_print=False):
    ordered_keys = sorted(losses.keys())
    if step is not None:
        ordered_keys = ['Iteration'] + ordered_keys

    exists = os.path.isfile(fname)
    if exists:
        with open(fname) as f:
            header = [x.strip() for x in f.readline().strip().split()]
            if header != ordered_keys:
                raise RuntimeError('Error! New header keys {} do not match existing {} in {}'.format(
                    header, ordered_keys, fname))
    with open(fname, 'a') as summary_ofile:
        if not exists:
            summary_ofile.write(to_file_line(ordered_keys))

        if step is not None:
            oline = '{:<8}'.format(step) + ' ' + \
                    to_file_line(ordered_dict_values(losses, ordered_keys[1:]), do_strip=False)
        else:
            oline = to_file_line(ordered_dict_values(losses, ordered_keys), do_strip=False)

        summary_ofile.write(oline)
        if do_print:
            print(to_file_line(ordered_keys).strip())
            print(oline.strip())


if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='ArtForger user interface.')
    aparser.add_argument('--gan_checkpoint', action='store', type=str, required=True)
    aparser.add_argument('--encoder_checkpoint', action='store', type=str, default=None)
    aparser.add_argument('--eval_output_dir', action='store', type=str, required=True)
    aparser.add_argument('--files_prefix', action='store', type=str, default='')
    aparser.add_argument('--geom_data', help='Training geometry data (directory or zip)', type=str)
    aparser.add_argument('--batch_size', action='store', type=int, default=30)
    aparser.add_argument('--primary_color_idx', action='store', type=int, default=0,
                         help='Index of the primary color in the user-set colors; legacy: 0, new: 1.')
    aparser.add_argument('--nbatches_per_style', action='store', default=1, type=int)
    aparser.add_argument('--geom_input_channel', help='Channel to use for geometry conditioning.', type=int, default=1)
    aparser.add_argument('--library_or_seeds', action='store', type=str, required=True,
                         help='If int, will create this many random styles with seeds. '
                              'If rand{int}, e.g. rand10 will create random zs. '
                              'If csv ints, will use these seeds. '
                              'If file, will load seeds or W library from file.')
    aparser.add_argument('--seed', action='store', type=int, default=None)
    aparser.add_argument('--debug', action='store_true')
    forger.util.logging.add_log_level_flag(aparser)
    args = aparser.parse_args()
    forger.util.logging.default_log_setup(args.log_level)

    random_state = forger.metrics.util.RandomState(args.seed)

    output_resolution = 128  # Hack
    geom_set_cropped = ImageFolderDataset(path=args.geom_data, use_labels=False, max_size=None, xflip=False,
                                          resolution=output_resolution, resize_mode='crop')

    empty_canvas = None
    generator = forger.metrics.util.PaintStrokeGenerator.create(
        encoder_checkpoint=args.encoder_checkpoint,
        gan_checkpoint=args.gan_checkpoint,
        device=torch.device(0),
        batch_size=args.batch_size,
        random_state=random_state)
    generator.primary_color_idx = args.primary_color_idx
    generator.set_geometry_source_from_iterator(
        get_image_data_iterator_from_dataset(geom_set_cropped, args.batch_size, num_workers=1), args.batch_size,
        geom_input_channel=args.geom_input_channel)

    # Initialize styles
    style_library = forger.ui.library.BrushLibrary.from_arg(args.library_or_seeds)

    geom_set = ImageFolderDataset(path=args.geom_data, use_labels=False, max_size=None, xflip=False)
    geom_iterator = get_image_data_iterator_from_dataset(geom_set, args.batch_size, num_workers=1)
    stitcher = forger.train.stitching.RandomStitcher()

    losses = paint_engine_metric_loop(
        generator,
        style_library,
        args.nbatches_per_style,
        geom_iterator,
        stitcher,
        args.geom_input_channel,
        args.eval_output_dir,
        debug=args.debug,
        files_prefix=args.files_prefix)

    logger.info('Evaluation complete, see: {}'.format(args.eval_output_dir))
