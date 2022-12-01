# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import argparse
import copy
import os
from time import perf_counter

import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

import experiment.util.latent as latent
from forger.experimental.autoenc.factory import create_autoencoder_from_checkpoint


def geom_feature_from_style_data(geom_encoder, style_data):
    """
    Copied from branch cy/geom_phase, loss_modified.py
    """
    phase_gray_style = torch.mean((style_data + 1.0) * 0.5, dim=1).unsqueeze(1)
    temp_min = torch.amin(phase_gray_style, dim=(2, 3), keepdim=True)
    temp_max = torch.amax(phase_gray_style, dim=(2, 3), keepdim=True)
    temp_min = temp_min.expand(style_data.shape[0], 1, style_data.shape[2], style_data.shape[3])
    temp_max = temp_max.expand(style_data.shape[0], 1, style_data.shape[2], style_data.shape[3])
    # Normalize to range [0.0, 1.0]
    phase_gray_style = (phase_gray_style - temp_min) / (temp_max - temp_min + torch.ones_like(temp_max) * 1e-7)
    phase_geom_feature = geom_encoder(phase_gray_style)
    return phase_geom_feature


def project(
    G,
    target: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    geom_feature               = None,
    num_steps                  = 1000,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute w stats
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    w_avg, w_std = latent.get_w_stats(num_samples=w_avg_samples,
                                      z_dim=G.z_dim,
                                      mapping_network=G.mapping,
                                      device=device)
    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        if geom_feature is None:
            synth_images = G.synthesis(ws, noise_mode='const')
        else:
            synth_images = G.synthesis(ws, geom_feature=geom_feature, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255/2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])

#----------------------------------------------------------------------------

def run_projection(
    gan_checkpt: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    geom_checkpt = None,
    geom_image = None,
):
    """Project given image to the latent space of pretrained network pickle.
    Since this script imports from `forger`,
    the following commands have to be executed at the root of the forger project.

    Examples:

    \b
    # For projecting using the original StyleGAN2 model -
    python -m thirdparty.stylegan2_ada_pytorch.projector --outdir=out --target=~/mytargetimg.png \\
        --gan_checkpt=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

    \b
    # For projecting using the Forger model
     python -m thirdparty.stylegan2_ada_pytorch.projector \\
        --gan_checkpt=<workspace>/experiment/some_forger_gan_checkpt.pkl \\
        --encoder_checkpt=<workspace>/experiment/geom_encoder_used.pkl \\
        --geom_image=<workspace>/data/geom_image.png
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading StyleGAN model from "%s"...' % gan_checkpt)
    device = torch.device('cuda')
    with dnnlib.util.open_url(gan_checkpt) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load style image that's used as target.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    geom_feature = None
    if geom_checkpt is not None:
        if geom_image is None:
            raise UserWarning("geom_checkpt and geom_image should both be provided for art-forger model")

        print('Loading geometry image from "%s"...' % geom_image)
        geom_pil = PIL.Image.open(geom_image).convert('RGB')
        w, h = geom_pil.size
        s = min(w, h)
        geom_pil = geom_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        geom_pil = geom_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
        geom_u8 = np.transpose(np.array(geom_pil, dtype=np.uint8), (2, 0, 1))
        geom_tensor = torch.from_numpy(geom_u8[:1, :, :]).to(torch.float32).unsqueeze(0).to(device)
        geom_tensor /= 255.0

        print('Loading autoencoder model from "%s"...' % geom_checkpt)
        autoencoder = create_autoencoder_from_checkpoint(geom_checkpt)
        encoder = autoencoder.encoder.eval().requires_grad_(False).to(device)

        geom_feature = encoder(geom_tensor)
        del autoencoder

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        geom_feature=geom_feature,
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            if geom_feature is not None:
                synth_image = G.synthesis(projected_w.unsqueeze(0), geom_feature=geom_feature, noise_mode='const')
            else:
                synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    if geom_feature is not None:
        synth_image = G.synthesis(projected_w.unsqueeze(0), geom_feature=geom_feature, noise_mode='const')
    else:
        synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gan_checkpt', type=str, required=True,
                        help='The .pkl file to load the Generator from.')
    parser.add_argument('--encoder_checkpt', type=str, default=None,
                        help='The .pkl that contains the autoencoder for encoding the geometry data in Forger.'
                             'If provided, then gan_checkpt is assumed to be a forger model.')
    parser.add_argument('--geom_image', type=str, default=None,
                        help='The path to the geometry image. Required if using a forger model.')
    parser.add_argument('--target_image', type=str, required=True,
                        help='The path to the target image')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--seed', type=int, default=303)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--outdir', type=str, required=True)
    args = parser.parse_args()

    run_projection(
        gan_checkpt=args.gan_checkpt,
        target_fname=args.target_image,
        outdir= args.outdir,
        save_video=args.save_video,
        seed=args.seed,
        num_steps=args.num_steps,
        geom_checkpt = args.encoder_checkpt,
        geom_image=args.geom_image
    ) # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
