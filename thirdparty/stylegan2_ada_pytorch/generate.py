# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import argparse
import os
import re
from typing import List, Optional

import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy

from forger.experimental.autoenc.factory import create_autoencoder_from_checkpoint

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

def generate_images(
    gan_checkpt: str,
    geom_image: Optional[str],
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    encoder_checkpt: Optional[str],
    save_uvs: bool,
    class_idx: Optional[int],
    projected_w: Optional[str]
):
    """Generate images using pretrained network pickle.
    Since this script imports from `forger`,
    the following commands have to be executed at the root of the forger project.

    Examples for original StyleGAN2:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python -m thirdparty.stylegan2_ada_pytorch.generate --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --gan_checkpt=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python -m thirdparty.stylegan2_ada_pytorch.generate --outdir=out --trunc=0.7 --seeds=600-605 \\
        --gan_checkpt=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    Examples for Forger model:
    python -m thirdparty.stylegan2_ada_pytorch.generate --outdir=out --seeds=600-605 \\
        --gan_checkpt=<workspace>/experiment/some_forger_gan_checkpt.pkl \\
        --encoder_checkpt=<workspace>/experiment/geom_encoder_used.pkl \\
        --geom_image=<workspace>/data/geom_image.png
    """

    print('Loading StyleGAN from "%s"...' % gan_checkpt)
    device = torch.device('cuda')
    with dnnlib.util.open_url(gan_checkpt) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    geom_feature = None
    if encoder_checkpt is not None:
        if geom_image is None:
            raise RuntimeError("To run a Forger model, --geom_image must be provided.")
        print('Loading geometry image from "%s"...' % geom_image)
        geom_pil = PIL.Image.open(geom_image).convert('RGB')
        # Also save a copy to outdir
        geom_pil.save(f'{outdir}/geom.png')
        w, h = geom_pil.size
        s = min(w, h)
        geom_pil = geom_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        geom_pil = geom_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        geom_u8 = np.transpose(np.array(geom_pil, dtype=np.uint8), (2, 0, 1))
        geom_tensor = torch.from_numpy(geom_u8[:1, :, :]).to(torch.float32).unsqueeze(0).to(device)
        geom_tensor /= 255.0

        print('Loading autoencoder model from "%s"...' % encoder_checkpt)
        autoencoder = create_autoencoder_from_checkpoint(encoder_checkpt)
        encoder = autoencoder.encoder.eval().requires_grad_(False).to(device)
        geom_feature = encoder(geom_tensor)
        del autoencoder

    os.makedirs(outdir, exist_ok=True)
    save_uvs = save_uvs and encoder_checkpt is not None

    # Synthesize the result of a W projection.
    if projected_w is not None:
        if seeds is not None:
            print ('warn: --seeds is ignored when using --projected-w')
        print(f'Generating images from projected W "{projected_w}"')
        ws = np.load(projected_w)['w']
        ws = torch.tensor(ws, device=device) # pylint: disable=not-callable
        assert ws.shape[1:] == (G.num_ws, G.w_dim)
        for idx, w in enumerate(ws):
            if geom_feature is None:
                img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
            else:
                img, triad_data = G.synthesis(w.unsqueeze(0),
                                              geom_feature=geom_feature,
                                              return_debug_data=True,
                                              noise_mode=noise_mode)

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/proj{idx:02d}.png')
            if save_uvs:
                np.savez(f'{outdir}/uvs_{idx:02d}.npz', uvs=triad_data['uvs'].cpu().numpy())
        return

    if seeds is None:
        raise UserWarning('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise UserWarning('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        if geom_feature is None:
            img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        else:
            img, triad_data = G(z,
                                    label,
                                    geom_feature=geom_feature,
                                    return_debug_data=True,
                                    truncation_psi=truncation_psi,
                                    noise_mode=noise_mode)

        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed_{seed:04d}.png')
        if save_uvs:
            np.savez(f'{outdir}/uvs_seed_{seed:04d}.npz', uvs=triad_data['uvs'].cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a single image from pretrained model')
    parser.add_argument('--gan_checkpt', type=str, required=True,
                        help='The .pkl file to load the Generator from.')
    parser.add_argument('--encoder_checkpt', type=str, default=None,
                        help='The .pkl that contains the autoencoder for encoding the geometry data in Forger.'
                             'If provided, then gan_checkpt is assumed to be a forger model.')
    parser.add_argument('--trunc', type=float, default=1.0,
                        help='Truncation psi')
    parser.add_argument('--geom_image', type=str, default=None,
                        help='The path to the geometry image. Required if using a forger model.')
    parser.add_argument('--seeds', type=num_range)
    parser.add_argument('--noise_mode', type=str, choices=('const', 'random', 'none'), default='const',
                        help='')
    parser.add_argument('--class_idx', type=int, default= None,
                        help='Class label (unconditional if not specified)')
    parser.add_argument('--outdir', type=str, required=True,
                        help='The directory where output data will be stored.')
    parser.add_argument('--save_uvs', action='store_true',
                        help='If set, the uvs from the triad data would be saved as a .npz file. '
                             'The flag will be ignored if using a StyleGAN model')
    parser.add_argument('--projected_w', type=str, default=None,
                        help='Projection result file')
    args = parser.parse_args()

    if (args.encoder_checkpt is None) != (args.geom_image is None):
        raise RuntimeError("Must provide both encoder_checkpt and geom_image")

    os.makedirs(args.outdir, exist_ok=True)
    if args.encoder_checkpt:
        print('Using geometry encoder for Forger model')
        assert args.geom_image is not None

    generate_images(
        gan_checkpt=args.gan_checkpt,
        encoder_checkpt=args.encoder_checkpt,
        geom_image=args.geom_image,
        save_uvs=args.save_uvs,
        seeds=args.seeds,
        truncation_psi=args.trunc,
        noise_mode=args.noise_mode,
        outdir=args.outdir,
        class_idx=args.class_idx,
        projected_w=args.projected_w,
    ) # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
