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
import PIL.Image
import random
import torch
from skimage.io import imsave
from skimage.transform import resize
from skimage.measure import shannon_entropy

from thirdparty.stylegan2_ada_pytorch.training.augment import AugmentPipe
from forger.util.logging import default_log_setup, log_tensor, add_log_level_flag

from forger.train.run_util import reclaim_cuda_memory
from forger.util.img_proc import RandomPatchGenerator

logger = logging.getLogger(__name__)


def crop_to_square_chw(img):
    height = img.shape[1]
    width = img.shape[2]
    if height == width:
        return img
    mindim = min(height, width)
    diff = int(abs(height - width))
    if random.random() > 0.5:
        start = diff
    else:
        start = 0
    end = start + mindim
    if height > width:
        return img[:, start:end, :]
    else:
        return img[:, :, start:end]


def ensure_three_channels(img):
    if len(img.shape) == 2:
        return np.stack([img, img, img], axis=2)
    if img.shape[-1] == 1:
        return np.concatenate([img, img, img], axis=2)
    if img.shape[-1] == 3:
        return img
    return img[:, :, :3]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Main training script.')
    # RUN FLAGS ---------------------------------------------------------------
    parser.add_argument(
        '--input_dirs', action='store', type=str, required=True,
        help='Directory (or CSV directories) with input images.')
    parser.add_argument(
        '--output_dir', action='store', type=str, required=True,
        help='Directory for output augmented images and patches')
    parser.add_argument(
        '--width', action='store', type=int, default=256,
        help='Width to which to resize the image patches.')
    parser.add_argument(
        '--fixed_crops', action='store_true',
        help='If true, will use the same crops for ALL images without augmentation.')
    parser.add_argument(
        '--naugment', action='store', type=int, default=50,
        help='Number of augmentatations per image.')
    parser.add_argument(
        '--augment_batch', action='store', type=int, default=10,
        help='Number of images per augmentation batch.')
    parser.add_argument(
        '--norig_patches', action='store', type=int, default=0,
        help='Number of original, unaugmented, patches to include from each image; '
        'original image resolution will be used, unless --randomize_orig_patch_size.')
    parser.add_argument(
        '--randomize_orig_patch_size', action='store_true',
        help='If set and --norig_patches > 0, will randomize scale of the patch within '
        'the original.')
    parser.add_argument(
        '--min_patch_entropy', action='store', type=float, default=None,
        help='If set, resamples patches until target entropy or number of retries is reached; '
        'discards patches that do not meet criterion even after resampling.')
    parser.add_argument('--max_patch_entropy', type=float, default=None,
                        help='The maximum entropy for the patch to be kept.')
    parser.add_argument('--entropy_channel', type=int, default=None)
    parser.add_argument(
        '--npatches', action='store', type=int, default=4,
        help='Number of patches per augmented images.')
    parser.add_argument('--range_max', type=float, default=1.0,
                        help='THe upper bound of patch size, as fraction to the size of the original image')
    parser.add_argument('--range_min', type=float, default=0.2,
                        help='THe lower bound of patch size, as fraction to the size of the original image')
    # Augmentation options
    parser.add_argument(
        '--xflip', type=int, choices=(0, 1), default=1,
        help='')
    parser.add_argument(
        '--scale', type=int, choices=(0, 1), default=1,
        help='')
    parser.add_argument(
        '--rotate', type=int, choices=(0, 1), default=1,
        help='')
    parser.add_argument(
        '--brightness', type=int, choices=(0, 1), default=1,
        help='')
    parser.add_argument(
        '--contrast', type=int, choices=(0, 1), default=1,
        help='')
    parser.add_argument(
        '--luma', type=int, choices=(0, 1), default=0,
        help='')
    parser.add_argument(
        '--hue', type=int, choices=(0, 1), default=1,
        help='')
    parser.add_argument(
        '--saturation', type=int, choices=(0, 1), default=1,
        help='')
    parser.add_argument(
        '--imgfilter', type=int, choices=(0, 1), default=0,
        help='')

    add_log_level_flag(parser)
    args = parser.parse_args()
    default_log_setup(args.log_level)
    device = torch.device(0)

    aug_settings = dict(  #strength=0.9,
        xflip=args.xflip,  # default to 1 --> good
        scale=args.scale,  # default to 1,  --> ok, but probably not needed for random patches
        rotate=args.rotate,  # default to 1, --> great
        brightness=args.brightness,  # default to 1,  --> good
        contrast=args.contrast,  # default to 1,  --> good (occasional gray canvas)
        lumaflip=args.luma,  # default to 1,  --> NO (black canvass)
        hue=args.hue,  # default to 1,   --> good (but not enough variation)
        saturation=args.saturation,  # default to 1,  --> good
        imgfilter=args.imgfilter)  # default to 1)  --> bad with standard settings, and even std=0.2

    augmentor = AugmentPipe(**aug_settings).to(device)
    # Randomize patch scale (prior to resizing)
    assert 0.0 < args.range_min <= args.range_max <= 1.0, "Something's wrong with the range parameters"
    default_patch_range = (args.range_min, args.range_max)
    orig_patch_range = default_patch_range if args.randomize_orig_patch_size else None
    patch_generator = RandomPatchGenerator(args.width, patch_range=default_patch_range)
    orig_patch_generator = RandomPatchGenerator(args.width, patch_range=orig_patch_range)
    patch_generator.filter_by_entropy(args.min_patch_entropy, args.max_patch_entropy, channel=args.entropy_channel)
    orig_patch_generator.filter_by_entropy(args.min_patch_entropy, args.max_patch_entropy, channel=args.entropy_channel)

    input_dirs = args.input_dirs.split(',')
    image_filenames = sorted(list(itertools.chain.from_iterable(
        [glob.glob(os.path.join(d, '*')) for d in input_dirs])))
    logger.info('Found {} filenames in {} directories'.format(len(image_filenames), len(input_dirs)))

    if args.naugment < args.augment_batch:
        args.augment_batch = args.naugment

    pidx = 0
    fixed_seed = time.time()
    nbatches = 1 if args.naugment <= 0 else args.naugment // args.augment_batch
    for fname in image_filenames:
        bname = '.'.join(os.path.basename(fname).split('.')[:-1])
        orig_img = ensure_three_channels(np.asarray(PIL.Image.open(fname)))
        orig_img = orig_img.astype(np.float32) / 255.0
        img = orig_img.transpose([2, 0, 1]) * 2 - 1  # HWC => CHW
        log_tensor(img, 'img {}'.format(fname), logger, level=logging.INFO, print_stats=True)

        for b in range(nbatches):
            patches = []
            if args.naugment > 0:
                clip_img = crop_to_square_chw(img)
                images = torch.from_numpy(np.stack([clip_img for x in range(args.augment_batch)])).to(device)
                log_tensor(images, 'images orig', logger)
                augmented = augmentor(images)
                log_tensor(augmented, 'images augmented', logger)
                augmented_images = np.clip((augmented.detach().cpu().numpy() + 1) / 2.0, 0, 1.0).transpose([0, 2, 3, 1])
                patches = [patch_generator.random_patch(augmented_images[i, ...])
                           for x in range(args.npatches)
                           for i in range(augmented_images.shape[0])]
            if b == 0 and args.norig_patches > 0:
                if args.fixed_crops:
                    random.seed(fixed_seed)
                patches.extend([orig_patch_generator.random_patch(orig_img)
                               for x in range(args.norig_patches)])

            patches = np.stack(patches)
            patches = patches * 255

            for i in range(patches.shape[0]):
                patch = patches[i, ...].astype(np.uint8)
                if args.min_patch_entropy is not None:
                    entropy = shannon_entropy(patch if args.entropy_channel is None else
                                              patch[..., args.entropy_channel])
                    logger.debug(f'patch has entropy {entropy}')
                    if entropy < patch_generator.min_entropy:
                        logger.warning('Discarding patch with low entropy: {}'.format(entropy))
                        continue
                    elif entropy > patch_generator.max_entropy:
                        logger.warning('Discarding patch with high entropy: {}'.format(entropy))
                        continue
                patch_bname = '%s_patch%06d.png' % (bname, pidx)
                imsave(os.path.join(args.output_dir, patch_bname), patch)
                logger.debug('Saving: {} (patch {})'.format(patch_bname, pidx))
                pidx += 1
            del patches
            reclaim_cuda_memory()


# installed requests, click, tqdm, psutil, tensorboard, skimage

# python -m scripts.patch_augment \
# --input_dirs=/tmp/src/ --output_dir=/tmp/augment_final2/ \
# --naugment=1 --augment_batch=1 --norig_patches=4 --npatches=4 --min_patch_entropy=0.2 \
# --log_level=0