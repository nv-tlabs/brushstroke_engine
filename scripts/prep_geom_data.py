# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import argparse
import glob
import importlib
import time
import itertools
import logging
import numpy as np
import os
from skimage.io import imread, imsave
import random
import torch
import multiprocessing
from skimage.io import imsave
from skimage.transform import resize
from skimage.measure import shannon_entropy

from thirdparty.stylegan2_ada_pytorch.training.augment import AugmentPipe
from forger.util.logging import default_log_setup, log_tensor, add_log_level_flag, log_tensor_dict

import forger.util.img_proc as img_proc
import forger.train.run_util as run_util
from forger.train.run_util import reclaim_cuda_memory
from forger.util.img_proc import RandomPatchGenerator

logger = logging.getLogger(__name__)


def overlay_as_red(bg, fg, factor=0.8):
    bg = bg.expand(3, -1, -1)
    alpha = fg * factor
    fg = torch.cat([torch.ones_like(fg), torch.zeros_like(fg), torch.zeros_like(fg)], dim=0)

    return bg * (1 - alpha) + fg * alpha


def process_image(im, use_alpha=False, binary_input=False, device='cpu'):
    if use_alpha:
        gray = img_proc.alpha_to_torch_gray(im)
    else:
        gray = img_proc.to_torch_gray(im)

    gray = gray.to(device)

    if binary_input:
        blurred = torch.zeros_like(gray)
        binimg = gray
    else:
        blurred = img_proc.blur_img(gray)
        binimg = img_proc.threshold_img_local(blurred)
    blurred2 = img_proc.blur_img(binimg)

    gray_bin_blurred = torch.cat([gray, binimg, blurred2], dim=0)
    res = img_proc.get_rolling_confidence(gray_bin_blurred)
    res_1ch = img_proc.encode_confidence_to_one_channel(res)

    overlay = overlay_as_red(gray, 1 - res_1ch)

    return {'gray': gray.cpu(),
            'blurred': blurred.cpu(),
            'binary': binimg.cpu(),
            'conf_rgb': res.cpu(),
            'conf': res_1ch.cpu(),
            'viz': overlay.cpu()}


def torch_ensure_three_channels(img):
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    if img.shape[0] == 1:
        return img.expand(3, -1, -1)
    elif img.shape[0] == 4:
        # Place on white canvas
        return img[:3, ...] * img[3:, ...] + 1.0 * (1 - img[3:, ...])
    else:
        return img[:3, ...]


def task_function(in_args):
    fname = in_args[1]
    device_id = in_args[0]

    bname = '.'.join(os.path.basename(fname).split('.')[:-1])
    ofname = os.path.join(args.output_dir, f'proc_{bname}.png')
    if os.path.isfile(ofname):
        logger.info(f'Skipping {bname}, exists: {ofname}')
        return

    use_cuda = False
    if device_id == 0:
        device = run_util.default_device(0)
        use_cuda = True
    else:
        device = torch.device('cpu')

    logger.info(f'Processing {fname}')

    im = imread(fname)
    res = process_image(im, use_alpha=args.use_alpha, binary_input=args.binary_input, device=device)
    if use_cuda:
        run_util.reclaim_cuda_memory()
    single_channel_result = torch.cat([res['gray'], res['binary'], res['conf']], dim=0)
    imsave(ofname, img_proc.ensure_np_uint8(single_channel_result))
    logger.debug(f'  Done: {ofname}')
    del single_channel_result

    if args.viz_output_dir is not None:
        viz_img = torch.cat(
            [torch_ensure_three_channels(res[x]) for x in ['gray', 'binary', 'conf_rgb', 'conf', 'viz']], dim=2)
        ofname = os.path.join(args.viz_output_dir, f'debug_{bname}.png')
        imsave(ofname, img_proc.ensure_np_uint8(viz_img))
        logger.debug(f'  Debug: {ofname}')
        del viz_img
    del res
    if use_cuda:
        run_util.reclaim_cuda_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random splines.')

    parser.add_argument(
        '--input_dirs', action='store', type=str, required=True,
        help='Directory (or CSV directories) with input images.')
    parser.add_argument(
        '--output_dir', action='store', type=str, required=True,
        help='Directory for output augmented images and patches')
    parser.add_argument(
        '--use_alpha', action='store_true',
        help='If set, will use alpha to help threshold.')
    parser.add_argument(
        '--binary_input', action='store_true',
        help='If set, assumes input is already thresholded.')
    parser.add_argument(
        '--viz_output_dir', action='store', type=str, default=None,
        help='Directory for output augmented images and patches')
    add_log_level_flag(parser)
    args = parser.parse_args()
    default_log_setup(args.log_level)
    device = torch.device(0)

    input_dirs = args.input_dirs.split(',')
    image_filenames = sorted(list(itertools.chain.from_iterable(
        [glob.glob(os.path.join(d, '*')) for d in input_dirs])))
    logger.info('Found {} filenames in {} directories'.format(len(image_filenames), len(input_dirs)))

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'Outputting to {args.output_dir}')
    if args.viz_output_dir is not None:
        logger.info(f'Also outputting visualization to {args.viz_output_dir}')
        os.makedirs(args.viz_output_dir, exist_ok=True)

    pool_obj = multiprocessing.Pool(4) #multiprocessing.cpu_count())
    samples = list(enumerate(image_filenames))

    pool_obj.map(task_function, samples)

    logger.info(f'Done: {args.output_dir}')
