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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

def get_pca_vecs(ws_file, num_vecs=10, ws_dim=64):
    ws_orig = np.fromfile(ws_file)
    log_tensor(ws_orig, 'ws', logger)
    ws_orig = ws_orig.reshape((-1, ws_dim))
    log_tensor(ws_orig, 'ws', logger, print_stats=True)

    ws = StandardScaler().fit_transform(ws_orig)
    log_tensor(ws, 'ws scaled', logger, print_stats=True)

    pca = PCA(n_components=num_vecs)
    pca.fit(ws)
    pca_vecs = pca.components_
    log_tensor(pca_vecs, 'pca_vecs', logger, print_stats=True)  # ws_dim x 10

    ws = torch.from_numpy(ws_orig)
    means = ws.mean(dim=0).unsqueeze(0)
    std = torch.std(ws, dim=0).unsqueeze(0)

    # 10 x ws_dim, ws_dim, ws_dim
    return torch.from_numpy(pca_vecs).to(torch.float32), means, std


if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='ArtForger user interface.')
    aparser.add_argument('--gan_checkpoint', action='store', type=str, required=True)
    aparser.add_argument('--encoder_checkpoint', action='store', type=str, required=False)
    aparser.add_argument('--ws', action='store', type=str, default='default')
    aparser.add_argument('--output_dir', action='store', type=str, required=True)
    aparser.add_argument('--nvec', type=int, default=10, help='Number of PCA components to use.')
    aparser.add_argument('--style_seeds', action='store', type=str, default='default',
                         help='If int, will create this many random styles. '
                              'If csv ints, will use these seeds. '
                              'If file, will load seeds from file.')
    aparser.add_argument('--clear_mode', action='store_true')
    aparser.add_argument('--seed', action='store', type=int, default=None)
    aparser.add_argument('--debug', action='store_true')
    forger.util.logging.add_log_level_flag(aparser)
    args = aparser.parse_args()

    forger.util.logging.default_log_setup(args.log_level)
    device = torch.device(0)

    default_subdir = 'viz_pca'
    if args.output_dir == 'default':
        args.output_dir = os.path.join(
            forger.viz.visualize.get_default_eval_directory(args.gan_checkpoint), default_subdir)
        logger.warning(f'Using default output directory: {args.output_dir}')
    os.makedirs(args.output_dir, exist_ok=True)

    if args.ws == 'default':
        args.ws = os.path.join(
            forger.viz.visualize.get_default_eval_directory(args.gan_checkpoint), 'ws.txt')
    pca_vecs, wmeans, wstds = get_pca_vecs(args.ws, num_vecs=args.nvec)
    pca_vecs = pca_vecs.to(device)
    wmeans = wmeans.to(device)
    wstds = wstds.to(device)

    random_state = forger.metrics.util.RandomState(args.seed)
    style_seeds = forger.metrics.util.style_seeds_from_flag(args.style_seeds, args.gan_checkpoint, random_state)
    style_seeds = list(style_seeds) + [-1]  # Mean

    generator = forger.metrics.util.PaintStrokeGenerator.create(
        encoder_checkpoint=args.encoder_checkpoint,
        gan_checkpoint=args.gan_checkpoint,
        device=device,
        batch_size=1,
        random_state=random_state)
    output_resolution = generator.engine.patch_width
    mapping = generator.engine.G.mapping
    synthesis = generator.engine.G.synthesis

    nincr = 10
    # nincr x 1
    weights = torch.cat([
        torch.linspace(-1, 0, steps=nincr)[:-1], torch.linspace(0, 1, steps=nincr)]).unsqueeze(1).to(device) * 5

    images = visualize.load_default_geometry_image(output_resolution).to(torch.float32)[..., 0] / 255.0
    images = images.unsqueeze(0).unsqueeze(1).to(device)
    geom_features = generator.engine.encoder.encode(images)
    geom_features = [x.expand(weights.shape[0], -1, -1, -1) for x in geom_features]

    suffix = ''
    if args.clear_mode:
        suffix = '_clear'
    wlayers = 0
    for idx, seed in enumerate(style_seeds):
        logger.info('Visualizing style {} / {} ({})'.format(idx, len(style_seeds), seed))
        viz_img = np.zeros((output_resolution * pca_vecs.shape[0], output_resolution * weights.shape[0], 4), dtype=np.uint8)

        if seed == -1:
            style_w = wmeans.unsqueeze(1).expand(-1, wlayers, -1)
            seed = 'MEAN'
        else:
            # 1 x zdim
            style_z = generator.get_random_style(seed)
            # 1 x nlayers x wdim
            style_w = mapping(style_z, None)
            wlayers = style_w.shape[1]

        for vec_idx in range(pca_vecs.shape[0]):
            # 1 x wdim
            vec = pca_vecs[vec_idx, ...].unsqueeze(0)
            # nincr x 1 x wdim =
            # nincr x nlayers x wdim = 1 x nlayers x wdim + nincr x 1 x wdim
            ws = style_w + (weights * vec).unsqueeze(1)
            res = synthesis(ws, geom_features, pos_encoding=None, return_debug_data=True)
            imgs = res[0]

            if args.clear_mode:
                uvs = res[1]['uvs']
                alpha = uvs[:, :2, ...].sum(dim=1, keepdims=True)
                imgs = torch.cat([imgs, alpha], dim=1)

            log_tensor(imgs, 'imgs', logger, print_stats=True)
            visualize.fill_image_row(viz_img, output_resolution * vec_idx, 0, (imgs / 2 + 0.5), margin=0)

        imsave(os.path.join(args.output_dir, f'pca_seed{seed}{suffix}.png'), viz_img)

    print(f'Done: {args.output_dir}')
