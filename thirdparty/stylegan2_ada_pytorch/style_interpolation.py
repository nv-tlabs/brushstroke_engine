# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate style mixing image matrix using pretrained network pickle."""

import argparse
import logging
import numpy as np
import os
import PIL
import re
from typing import List, Optional
import torch

import dnnlib
import legacy
from train import CommaSeparatedList
import forger.experimental.autoenc.factory as factory

logger = logging.getLogger(__name__)

# TODO: modify this function for art-forger
def debug_synthesis_forward(syn, ws, geom_feature=None, **block_kwargs):
    block_ws = []
    with torch.autograd.profiler.record_function('split_ws'):
        ws = ws.to(torch.float32)
        w_idx = 0
        for res in syn.block_resolutions:
            block = getattr(syn, f'b{res}')
            block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            w_idx += block.num_conv

    imgs = []
    x = img = None
    for res, cur_ws in zip(syn.block_resolutions, block_ws):
        block = getattr(syn, f'b{res}')
        x, img = block(x, img, cur_ws, **block_kwargs)
        if geom_feature is not None and res in syn.geom_inject_resolutions:
            x = torch.cat([x, geom_feature], dim=1)
        imgs.append(img)
    return x, img, imgs

#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

def generate_style_mix(
    gan_checkpt: str,
    encoder_checkpt: Optional[str],
    geom_image: np.ndarray,
    a_seeds: List[int],
    b_seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str
):
    """Generate images using pretrained network pickle.
    Since this script imports from `forger`,
    the following commands have to be executed at the root of the forger project.

    Examples:

    \b
    # for original StyleGAN2
    python -m thirdparty.stylegan2_ada_pytorch.style_interpolation --outdir=out --rows=85,100,75,458,1500 --cols=55,821,1789,293 \\
        --gan_checkpt=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    python -m thirdparty.stylegan2_ada_pytorch.interpolation \\
        --outdir=out --a=85,100,75,458,1500 --b=55,821,1789,293 \\
        --gan_checkpt=<workspace>/experiment/some_forger_gan_checkpt.pkl \\
        --encoder_checkpt=<workspace>/experiment/geom_encoder_used.pkl \\
        --geom_image=<workspace>/data/geom_image.png
    """
    print('Loading StyleGAN from "%s"...' % gan_checkpt)
    device = torch.device('cuda')
    with dnnlib.util.open_url(gan_checkpt) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

    print('Generating W vectors...')
    all_seeds = list(a_seeds + b_seeds)
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    all_w = G.mapping(torch.from_numpy(all_z).to(device), None)
    w_avg = G.mapping.w_avg
    all_w = w_avg + (all_w - w_avg) * truncation_psi
    w_dict = {seed: w for seed, w in zip(all_seeds, list(all_w))}

    forger = encoder_checkpt is not None
    if forger:
        assert geom_image is not None, "geom_image must be provided when doing style interpolation on forger models"
        print("Loading autoencoder...")
        geom_autoencoder = factory.create_autoencoder_from_checkpoint(encoder_checkpt=encoder_checkpt)
        geom_autoencoder.eval().requires_grad_(False).to(device)

        # Convert geom_image to the desired format TODO: Now that we have the util function
        assert len(geom_image.shape) == 3
        geom_image = np.transpose(geom_image, (2, 0, 1))
        geom_img_arr = torch.from_numpy(geom_image[:1, :, :]).to(torch.float32).unsqueeze(0).to(device) / 255.0

        geom_feature = geom_autoencoder.encoder(geom_img_arr)
        all_geom_feature = geom_feature.repeat(len(all_w), 1, 1, 1)
        print('Generating images...')
        all_images = G.synthesis(all_w, geom_feature=all_geom_feature, noise_mode=noise_mode)
    else:
        print('Generating images...')
        all_images = G.synthesis(all_w, noise_mode=noise_mode)

    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    image_dict = {(seed, seed): image for seed, image in zip(all_seeds, list(all_images))}

    # These are not seeds, just ids
    row_seeds = list(zip(a_seeds, b_seeds))
    col_seeds = None
    debug_imgs = []
    print('Generating style-mixed images...')

    for i in range(len(a_seeds)):
        a_seed = a_seeds[i]
        b_seed = b_seeds[i]

        w_a = w_dict[a_seed]
        w_b = w_dict[b_seed]
        print("w_a shape", w_a.shape)

        col_seeds = list(range(w_a.shape[0] + 1))

        for layer in col_seeds:
            w0 = w_a.clone()
            w1 = w_b.clone()
            if layer > 0:
                w0[0:layer] = w_b[0:layer]
                w1[0:layer] = w_a[0:layer]

            if forger:
                image = G.synthesis(torch.stack([w0, w1]),
                                    geom_feature=geom_feature.repeat(2, 1, 1, 1),
                                    noise_mode=noise_mode)
            else:
                image = G.synthesis(torch.stack([w0, w1]), noise_mode=noise_mode)
            image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            image_dict[((a_seed, b_seed), layer)] = image[0].cpu().numpy()
            image_dict[((b_seed, a_seed), layer)] = image[1].cpu().numpy()

            if layer == 13:
                debug_imgs.append(debug_synthesis_forward(G.synthesis,
                                                          w0.unsqueeze(0),
                                                          geom_feature=geom_feature,
                                                          noise_mode=noise_mode)[-1])



    # print('Saving images...')
    # os.makedirs(outdir, exist_ok=True)
    # for (row_seed, col_seed), image in image_dict.items():
    #     PIL.Image.fromarray(image, 'RGB').save(f'{outdir}/{row_seed}-{col_seed}.png')

    print('Saving image grid...')
    W = G.img_resolution
    H = G.img_resolution
    canvas = PIL.Image.new('RGB', (W * (len(col_seeds * 2)), H * (len(row_seeds))), 'black')
    for row_idx, row_seed in enumerate(row_seeds):
        for col_idx, col_seed in enumerate(col_seeds):
            key = (row_seed, col_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (W * col_idx, H * row_idx))
            key2 = ((row_seed[1], row_seed[0]), col_seed)
            canvas.paste(PIL.Image.fromarray(image_dict[key2], 'RGB'), (W * (col_idx + len(col_seeds)), H * row_idx))

    canvas.save(f'{outdir}/grid.png')

    print('Generating debug pyramid for one pair')
    # Somehow debug pyramid doesn't work here, because 'None' is returned
    debug_canvas = PIL.Image.new('RGB', (W * (len(debug_imgs[0])), H * (len(row_seeds))), 'black')
    for row_idx, row_seed in enumerate(row_seeds):
        for col_idx in range(len(debug_imgs[0])):
            if debug_imgs[row_idx][col_idx] is None:
                continue
            debug_canvas.paste(
                PIL.Image.fromarray(
                    (debug_imgs[row_idx][col_idx].permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)[0].cpu().numpy(),
                    'RGB'),
                (W * col_idx, H * row_idx))

    debug_canvas.save(f'{outdir}/debug_pyramid_grid_style13.png')

    print('Generating W vectors and images for z interpolation...')
    z_a = all_z[0:len(a_seeds), :]  # A x 512
    z_b = all_z[len(a_seeds):, :]   # A x 512  --> A x 14 x 512

    interp = (np.arange(0, 14) / 13).reshape((1, -1, 1))
    z_interp = np.expand_dims(z_a, 1) * (1 - interp) + np.expand_dims(z_b, 1) * interp

    canvas_interp = PIL.Image.new('RGB', (W * interp.size, H * (len(row_seeds))), 'black')
    geom_feature_14 = geom_feature.repeat(14, 1, 1, 1)
    for r in range(len(a_seeds)):
        w_interp = G.mapping(torch.from_numpy(z_interp[r, ...]).to(device), None)
        w_avg = G.mapping.w_avg
        w_interp = w_avg + (w_interp - w_avg) * truncation_psi

        if forger:
            images_interp = G.synthesis(w_interp, geom_feature=geom_feature_14, noise_mode=noise_mode)
        else:
            images_interp = G.synthesis(w_interp, noise_mode=noise_mode)

        images_interp = (images_interp.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()

        for c in range(images_interp.shape[0]):
            canvas_interp.paste(PIL.Image.fromarray(images_interp[c, ...], 'RGB'), (W * c, H * r))

    canvas_interp.save(f'{outdir}/grid_zinterp.png')



#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Style interpolation")
    parser.add_argument('--gan_checkpt', type=str, required=True,
                        help='The .pkl file to load the Generator from.')
    parser.add_argument('--encoder_checkpt', type=str, default=None,
                        help='The .pkl that contains the autoencoder for encoding the geometry data in Forger.'
                             'If provided, then gan_checkpt is assumed to be a forger model.')
    parser.add_argument('--geom_image', type=str,
                        help='The path to the geometry image')
    parser.add_argument('--a', type=CommaSeparatedList.convert, required=True,
                        help='Random seeds to use for left image')
    parser.add_argument('--b', type=CommaSeparatedList.convert, required=True,
                        help='Random seeds to use for right image')
    parser.add_argument('--trunc', type=float, default=1.0,
                        help='truncation psi')
    parser.add_argument('--noise_mode', choices=('const', 'random', 'none'), type=str, default='const',
                        help='Noise mode')
    parser.add_argument('--outdir', type=str, required=True)

    args = parser.parse_args()

    geom_image = None
    if args.encoder_checkpt is not None:
        print("--encoder_checkpt set, running on Forger model")
        if args.geom_image is None:
            raise RuntimeError("geom image not provided for Art-forger model")
        geom_image = np.asarray(PIL.Image.open(args.geom_image))[:, :, :3]

    a_seeds = [int(s) for s in args.a]
    b_seeds = [int(s) for s in args.b]

    generate_style_mix(
        gan_checkpt=args.gan_checkpt,
        encoder_checkpt=args.encoder_checkpt,
        geom_image=geom_image,
        a_seeds=a_seeds,
        b_seeds=b_seeds,
        truncation_psi=args.trunc,
        noise_mode=args.noise_mode,
        outdir=args.outdir
    ) # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
