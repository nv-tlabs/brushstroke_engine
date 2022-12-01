# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import argparse
import logging
import numpy as np
import os
import random
from skimage.io import imsave
from skimage import morphology
import multiprocessing

try:
    import splines
except Exception as e:
    print('Could not import splines')

import forger.util.logging
from forger.util.logging import log_tensor
from forger.core.curve import CatmullRomSpline, draw_spline, sample_control_pts2
from forger.util.spline_dist import map_flag_to_distrib_class
import thirdparty.stylegan2_ada_pytorch.dnnlib as dnnlib

logger = logging.getLogger(__name__)

# Sample command:
# python -m scripts.random_splines --samples=50 --out_dir=/tmp/splines --log_level=10 --pts_min=6 --pts_max=20


def generate_sample(i):
    npts = random.randint(args.pts_min, args.pts_max)
    if args.smart_sampling:
        pts = sample_control_pts2(npts)
    else:
        pts = np.random.rand(npts, 2).astype(np.float32) * 2.2 - 1

    if args.use_lib:
        spline = splines.CatmullRom(pts)
        nsamples = 40 * npts
    else:
        spline = CatmullRomSpline(pts, 0.5)
        nsamples = args.width * 3 * npts
    res = draw_spline(spline, width=args.width, nsamples=nsamples)

    if not args.use_radii or len(args.use_radii) == 0:
        radii = [distrib.sample()]
    else:
        radii = args.use_radii

    for radius in radii:
        dilated = np.copy(res)
        radius = int(radius)
        log_tensor(dilated, 'dilated', logger, level=logging.INFO)
        if radius >= 2.0:
            dilated[:, :, 0] = morphology.erosion(res[:, :, 0], morphology.disk(radius=radius))

        outfile = os.path.join(args.out_dir, 'spline%06d_rad%03d.png' % (i, radius))
        logger.info(f'Saving {outfile}')
        imsave(outfile, dilated)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate random splines.')
    parser.add_argument('--width', action='store', type=int, default=256, help='The side length of the square spline image')
    parser.add_argument('--out_dir', action='store', type=str, required=True)
    parser.add_argument('--pts_min', action='store', type=int, default=4)
    parser.add_argument('--pts_max', action='store', type=int, default=15)
    parser.add_argument('--use_radii', nargs='+', type=int,
                        help='Set to use the same set of radii for every spline, no thickness sampling.')
    parser.add_argument('--samples', action='store', type=int, required=True, help='Number of spline images to generate')
    parser.add_argument('--use_lib', action='store_true', help='Set to use the splines package for generating splines')
    parser.add_argument('--thick_dist', type=str, default='gauss', help='The type of distribution for the spline thickness')
    parser.add_argument('--smart_sampling', action='store_true')

    forger.util.logging.add_log_level_flag(parser)
    args, _ = parser.parse_known_args()
    forger.util.logging.default_log_setup(args.log_level)
    os.makedirs(args.out_dir, exist_ok=True)

    distrib_class = map_flag_to_distrib_class(args.thick_dist)
    parser = distrib_class.modify_cmd_arg(parser)
    args, _ = parser.parse_known_args()
    distrib = distrib_class(args)

    pool_obj = multiprocessing.Pool(multiprocessing.cpu_count())
    samples = range(args.samples)

    pool_obj.map(generate_sample, samples)
