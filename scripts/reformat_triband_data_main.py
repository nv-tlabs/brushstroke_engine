# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import argparse
import os
from skimage.io import imread, imsave
import glob
import logging
import numpy as np

import forger.util.logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='ArtForger user interface.')
    aparser.add_argument('--input_dir', action='store', type=str, required=True)
    aparser.add_argument('--output_dir', action='store', type=str, required=True)
    forger.util.logging.add_log_level_flag(aparser)
    args = aparser.parse_args()

    forger.util.logging.default_log_setup(args.log_level)


    fnames = glob.glob(os.path.join(args.input_dir, "*.png"))
    print(f'Found {len(fnames)} fnames, e.g. {fnames[0]}')

    for idx, fname in enumerate(fnames):
        outname = os.path.join(args.output_dir, os.path.basename(fname))

        if idx % 100 == 0:
            logger.info(f'Processing {idx}/{len(fnames)}: {fname} --> {outname}')

        if os.path.isfile(outname):
            raise RuntimeError(f'Output file exists: {outname}')
        im = imread(fname)
        #forger.util.logging.log_tensor(im, 'im', logger, level=logging.INFO)
        im_out = np.concatenate([im[..., 1:2], im[..., 1:2], im[..., 1:2]], axis=-1)
        #forger.util.logging.log_tensor(im_out, 'im_out', logger, level=logging.INFO)
        imsave(outname, im_out)