# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import argparse
import logging
import os
import pickle
from time import perf_counter

import imageio
import numpy as np
import random
import torch
import torch.nn.functional as F
import torchvision
from skimage.io import imsave, imread

import thirdparty.stylegan2_ada_pytorch.experiment.util.latent as latent
from thirdparty.stylegan2_ada_pytorch.training.dataset import ImageFolderDataset

import forger.metrics.geom_metric
import forger.ui.brush
import forger.util.logging
import forger.viz.visualize
from forger.util.img_proc import RandomPatchGenerator

from forger.viz.visualize import output_encoder_diagnostics
from forger.util.torch_data import get_image_data_iterator_from_dataset
import forger.ui.library
import forger.train.losses

logger = logging.getLogger(__name__)


def make_viz(raw, raw_prev, geo):
    # Input geometry
    geo = torchvision.utils.make_grid(geo.expand(-1, 3, -1, -1), nrow=1, padding=0).permute(2, 1, 0) * 2 - 1

    # Previous result and masks
    prev = torchvision.utils.make_grid(raw['fake_orig'].detach(), nrow=1, padding=0).permute(2, 1, 0)
    prev_debug = forger.viz.visualize.visualize_raw_data(raw_prev).permute(2, 1, 0)

    # Current result and masks
    current = torchvision.utils.make_grid(raw['fake_img'].detach(), nrow=1, padding=0).permute(2, 1, 0)
    current_debug = forger.viz.visualize.visualize_raw_data(raw).permute(2, 1, 0)

    res = torch.cat([geo, prev_debug, prev, current_debug, current], dim=0)

    res = ((res / 2.0 + 0.5) * 255).clip(0, 255).to(torch.uint8).detach().cpu().numpy()
    return res

def run_one_clarity_opt(
        engine,
        geom_set_iterator,
        geom_input_channel,
        geom_truth_channel,
        device,
        w_plus,
        w_start,
        w_std,
        losses,
        num_steps                  = 1000,
        initial_learning_rate      = 0.1,
        initial_noise_factor       = 0.05,
        lr_rampdown_length         = 0.25,
        lr_rampup_length           = 0.05,
        noise_ramp_length          = 0.75,
        regularize_noise_weight    = 10,  #1e5,
        output_video               = None,
        output_prefix              = None
):
    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in engine.G.synthesis.named_buffers() if 'noise_const' in name}

    if w_plus and w_start.shape[1] == 1:
        w_init = np.concatenate([w_start for _ in range(engine.G.mapping.num_ws)], axis=1)
    elif not w_plus and w_start.shape[1] > 1:
        w_init = torch.mean(w_start, dim=1, keepdim=True)
    else:
        w_init = w_start.clone()
    w_opt = torch.tensor(w_init, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable
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

        # Get geometry
        with torch.no_grad():
            geom, _ = next(geom_set_iterator)
            geom = geom.to(device).to(torch.float32) / 255.0
            geom_input = geom[:, geom_input_channel:geom_input_channel + 1, :, :]
            geom_feature = engine.encoder.encode(geom_input)

        # Get geometry images from original w
        target_images, raw_prev = engine.G.synthesis(
            w_start.repeat([geom.shape[0], 1 if w_plus else engine.G.mapping.num_ws, 1]),
            geom_feature=geom_feature, noise_mode='const',
            return_debug_data=True)

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([geom.shape[0], 1 if w_plus else engine.G.mapping.num_ws, 1])
        synth_images, raw = engine.G.synthesis(ws, geom_feature=geom_feature, noise_mode='const',
                                               return_debug_data=True)
        raw['fake_orig'] = target_images
        raw['fake_img'] = synth_images

        dist, loss_details = losses.compute(raw, geom[:, geom_truth_channel:geom_truth_channel + 1, :, :])

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        if step % 100 == 0:
            logger.info(
                ('Step %d: %0.4f Loss  =  ' % (step, loss)) +
                ' + '.join(['%s %0.4f' % (k, v) for k, v in loss_details.items()] +
                           ['reg_w * reg_l %0.4f * %0.4f = %0.4f' %
                            (regularize_noise_weight, reg_loss, reg_loss * regularize_noise_weight)]))

            if output_video is not None:
                output_video.append_data(
                    ((torchvision.utils.make_grid(
                        synth_images.detach().cpu(), nrow=synth_images.shape[0], padding=0) / 2.0 + 0.5) * 255). \
                        clip(0, 255).to(torch.uint8).permute(1, 2, 0).numpy())

        if step % 1000 == 0:
            imsave(output_prefix + '_it%06d.png' % step, make_viz(raw, raw_prev, geom_input))

        if step == num_steps - 1 and output_prefix is not None:
            imsave(output_prefix + '_finalres.png', make_viz(raw, raw_prev, geom_input))

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_opt.detach()


def run_clarity_opt(
        geom_data,
        geom_input_channel,
        geom_truth_channel,
        engine,
        library,
        brush_id,
        losses,
        batch_size,
        outdir: str,
        out_library_fname: str,
        save_video: bool,
        seed: int,
        num_steps: int,
        w_plus: bool,
        w_avg_samples = 10000
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    viz_dir = os.path.join(outdir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)

    G = engine.G
    img_resolution = engine.G.img_resolution
    device = engine.device

    geom_set = ImageFolderDataset(path=geom_data, use_labels=False, max_size=None, xflip=False,
                                  resolution=img_resolution, resize_mode='crop')
    geom_iterator = get_image_data_iterator_from_dataset(geom_set, batch_size=batch_size)
    output_encoder_diagnostics(next(geom_iterator)[0], engine.encoder, device, outdir,
                               geom_input_channel=geom_input_channel)

    # Compute w stats
    logger.info(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    w_avg, w_std = latent.get_w_stats(num_samples=w_avg_samples,
                                      z_dim=G.z_dim,
                                      mapping_network=G.mapping,
                                      device=device)

    # Optimize clarity of the style
    brush_ids = [brush_id] if brush_id is not None else library.get_style_ids()
    for bid in brush_ids:
        opts = forger.ui.brush.GanBrushOptions()
        library.set_style(bid, opts)

        if opts.style_z is not None:
            w_start = engine.G.mapping(opts.style_z, None)
        else:
            w_start = opts.style_ws

        w_start = w_start.to(device)
        video = None
        file_prefix = f'{outdir}/{bid}_opt'
        viz_file_prefix = f'{viz_dir}/{bid}_opt'
        if save_video:
            video_ofile = f'{viz_file_prefix}.mp4'
            video = imageio.get_writer(video_ofile, mode='I', fps=10, codec='libx264', bitrate='16M')

        start_time = perf_counter()
        projected_w = run_one_clarity_opt(
            engine,
            geom_iterator,
            geom_input_channel,
            geom_truth_channel,
            device,
            w_plus,
            w_start,
            w_std,
            losses,
            num_steps=num_steps,
            output_video=video,
            output_prefix=viz_file_prefix)
        logger.info(f'Elapsed: {(perf_counter()-start_time):.1f} s')

        if video is not None:
            video.close()

        projected_w = projected_w.detach().cpu()
        np.savez(os.path.join(outdir, f'{file_prefix}_optimized.npz'), w=projected_w.numpy())

        fname_all_pkl = out_library_fname
        all_data = {}
        if os.path.isfile(fname_all_pkl):
            with open(fname_all_pkl, 'rb') as f:
                all_data = pickle.load(f)
        if bid in all_data:
            logger.warning(f'All pickle already has projection for {bid}, overwriting entry in: {fname_all_pkl}')
        all_data[bid] = projected_w
        with open(fname_all_pkl, 'wb') as f:
            pickle.dump(all_data, f)
            logger.info(f'Added w entry for {bid} to {fname_all_pkl}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gan_checkpoint', action='store', type=str, required=True)
    parser.add_argument('--encoder_checkpoint', action='store', type=str, required=False)
    parser.add_argument('--brush_id', action='store', type=str, default=None)
    parser.add_argument('--library', action='store', type=str, required=True)
    parser.add_argument('--out_library', action='store', type=str, required=True,
                        help='Set to output file or else to "default"')
    parser.add_argument('--output_dir', action='store', type=str, required=True)

    parser.add_argument('--geom_data', help='Training geometry data (directory or zip)', type=str, required=True)
    parser.add_argument('--geom_input_channel', help='Channel to use for geometry conditioning.', type=int, default=1)
    parser.add_argument('--geom_truth_channel', help='Channel to use for geometry loss.', type=int, default=2)
    parser.add_argument('--batch_size', help='Batch size for optimization.', type=int, default=20)
    parser.add_argument('--losses', default='0.5*iou_inv(uvs)+0.5*iou(u)+50*lpips(fake_orig)+50*l1(fake_orig)',
                        type=str)

    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    #parser.add_argument('--w_plus', action='store_true')
    parser.add_argument('--num_steps', type=int, default=1000)

    forger.util.logging.add_log_level_flag(parser)
    args = parser.parse_args()

    forger.util.logging.default_log_setup(args.log_level)
    device = torch.device(0)

    os.makedirs(args.output_dir, exist_ok=True)

    engine = forger.ui.brush.PaintEngineFactory.create(
        encoder_checkpoint=args.encoder_checkpoint,
        gan_checkpoint=args.gan_checkpoint,
        device=device)

    library = forger.ui.library.BrushLibrary.from_file(args.library)

    losses = forger.train.losses.ForgerLosses.create_from_string(args.losses)

    if args.out_library == 'default':
        args.out_library = os.path.join(os.path.dirname(args.library),
                                        'OPT_' + os.path.basename(args.library))
    logger.info(f'Writing out to library: {args.out_library}')

    run_clarity_opt(
        geom_data=args.geom_data,
        geom_input_channel=args.geom_input_channel,
        geom_truth_channel=args.geom_input_channel,
        engine=engine,
        library=library,
        brush_id=args.brush_id,
        losses=losses,
        batch_size=args.batch_size,
        outdir=args.output_dir,
        out_library_fname=args.out_library,
        save_video=args.save_video,
        seed=args.seed,
        num_steps=args.num_steps,
        w_plus=True
    )
