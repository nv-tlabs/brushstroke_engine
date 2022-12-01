# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

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

import forger.metrics.geom_metric
import forger.ui.brush
import forger.util.logging
from forger.util.logging import log_tensor
import forger.viz.visualize
from forger.util.img_proc import RandomPatchGenerator


logger = logging.getLogger(__name__)


def project(
        G,
        target: torch.Tensor,  # [B, C, H, W] and dynamic range [-1..1], W & H must match G output resolution
        geom,
        geom_feature,
        device,
        w_plus,
        num_steps                  = 1000,
        w_avg_samples              = 10000,
        initial_learning_rate      = 0.1,
        initial_noise_factor       = 0.05,
        lr_rampdown_length         = 0.25,
        lr_rampup_length           = 0.05,
        noise_ramp_length          = 0.75,
        regularize_noise_weight    = 10,  #1e5,
        output_video               = None,
        optimize_noise             = True,
        norm_positions = None,
        with_composite = False,
        l1_fg_weight = 0,
        bg_weight = 0,
        resume_from = None,
        min_lpips_improvement = 0.0001,
        target_bg = None,
        w_std = None,
        w_avg = None
):
    assert target.shape[1:] == (G.img_channels, G.img_resolution, G.img_resolution)
    loss_weights = {'lpips': 1.0,
                    'reg': regularize_noise_weight,
                    'l1': l1_fg_weight,
                    'bg': bg_weight}

    l1_crit = torch.nn.L1Loss()
    fg, bg = forger.metrics.geom_metric.get_conservative_fg_bg(geom.to(device))
    bg_color = compute_masked_color(target, bg)
    fg = fg.expand(-1, 3, -1, -1)

    noise_mode = 'const' #('const' if optimize_noise else 'random')

    logger.info(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    if w_std is None or w_avg is None:
        w_avg, w_std = latent.get_w_stats(num_samples=w_avg_samples,
                                          z_dim=G.z_dim,
                                          mapping_network=G.mapping,
                                          device=device)
    w_start = w_avg
    del w_avg
    # Allow optimization into W+
    if w_plus:
        w_start = np.concatenate([w_start for _ in range(G.mapping.num_ws)], axis=1)

    # Compute w stats
    if resume_from is not None and 'w' in resume_from:
        logger.info('Resuming from W')
        if w_start.shape != resume_from['w'].shape:
            w_start = np.concatenate([resume_from['w'] for _ in range(G.mapping.num_ws)], axis=1).to(device)
        else:
            w_start = resume_from['w'].to(device)

    w_opt = torch.tensor(w_start, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable

    # Setup noise inputs.
    noise_bufs = {}
    if optimize_noise:
        noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

        if resume_from is not None and 'noise' in resume_from:
            logger.info('Resuming from noise')
            start_noise = resume_from['noise']
            for k, v in noise_bufs.items():
                v[:] = start_noise[k][:]

    prev_lpips_best = None
    lpips_best = None
    w_best = w_opt.detach().cpu()
    noise_best = dict([(k, v.detach().cpu()) for k, v in noise_bufs.items()])
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
        ws = (w_opt + w_noise).repeat([target.shape[0], 1 if w_plus else G.mapping.num_ws, 1])
        synth_images, raw = G.synthesis(ws, geom_feature=geom_feature, noise_mode=noise_mode,
                                        norm_noise_positions=norm_positions, noise_buffers=noise_bufs,
                                        return_debug_data=True)
        if with_composite:
            # if step == 1:  #HACK
            #     imsave('/tmp/pr_render.png', ((synth_images[0, ...] / 2 + 0.5) * 255).to(torch.uint8).permute(1,2,0).detach().cpu().numpy())

            if target_bg is None:
                synth_images = composite_with_bg_color(raw, bg_color)
            else:
                synth_images = composite_with_bg_image(raw, target_bg)
            # synth_images_white = composite_with_bg(raw)  # white bg

            #HACK
            # if step == 1:
            #     imsave('/tmp/pr_bg.png', (bg[0, 0, ...].to(torch.float32) * 255).to(torch.uint8).detach().cpu().numpy())
            #     imsave('/tmp/pr_fg.png', (fg[0, 0, ...].to(torch.float32) * 255).to(torch.uint8).detach().cpu().numpy())
            #     imsave('/tmp/pr_render_avebg.png',
            #            ((synth_images[0, ...] / 2 + 0.5) * 255).to(torch.uint8).permute(1,2,0).detach().cpu().numpy())
            #     imsave('/tmp/pr_render_whitebg.png',
            #            ((synth_images_white[0, ...] / 2 + 0.5) * 255).to(torch.uint8).permute(1,2,0).detach().cpu().numpy())

        losses = {'lpips': forger.metrics.geom_metric.lpips_batched_vgg(target, synth_images).mean()}
        if lpips_best is None or losses['lpips'] < lpips_best:
            lpips_best = losses['lpips']
            w_best = w_opt.detach().cpu()
            noise_best = dict([(k, v.detach().cpu()) for k, v in noise_bufs.items()])

        if l1_fg_weight > 0:
            losses['l1'] = l1_crit(target[fg], synth_images[fg])

        if bg_weight > 0:
            losses['bg'] = (1 - raw['uvs'][:, 2:, ...][bg]).mean()

        del raw

        # Noise regularization.
        losses['reg'] = 0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                losses['reg'] += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                losses['reg'] += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        loss = 0
        for k, v in losses.items():
            loss = loss + loss_weights[k] * v

        if step % 100 == 0:
            logger.info('Step %d: %s (best lpips %0.4f)' %
                        (step,
                         ' + '.join(['%s %0.4f * weight %0.4f' % (k, v, loss_weights[k]) for k, v in losses.items()]),
                         lpips_best))

            if output_video is not None:
                output_video.append_data(
                    ((torchvision.utils.make_grid(
                        synth_images.detach().cpu(), nrow=synth_images.shape[0], padding=0) / 2.0 + 0.5) * 255).\
                        clip(0, 255).to(torch.uint8).permute(1, 2, 0).numpy())

            if prev_lpips_best is None:
                prev_lpips_best = lpips_best
            else:
                if prev_lpips_best - lpips_best < min_lpips_improvement:
                    logger.info(
                        'Not enough LPIPS improvement since prior log %0.5f --> %0.5f, stopping after %d steps' %
                        (prev_lpips_best, lpips_best, step))
                    break
                prev_lpips_best = lpips_best

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    result = {'w': w_best,
              'noise': noise_best,
              'bg': bg_color.detach().squeeze().cpu(),
              'step': step}
    return result


def composite_with_bg_color(raw, bg_color=None):
    if bg_color is None:
        bg_color = torch.ones((1, 3, 1, 1), dtype=torch.float32, device=raw['colors'].device)

    bg_uvs_idx = 2
    alpha = 1 - raw['uvs'][:, bg_uvs_idx, ...].unsqueeze(1)

    stroke = forger.viz.visualize.compose_stroke(raw['uvs'], raw['colors'])

    return stroke * alpha + bg_color.reshape(1, 3, 1, 1) * (1 - alpha)


def composite_with_bg_image(raw, bg_patches):
    bg_uvs_idx = 2
    alpha = 1 - raw['uvs'][:, bg_uvs_idx, ...].unsqueeze(1)

    stroke = forger.viz.visualize.compose_stroke(raw['uvs'], raw['colors'])

    return stroke * alpha + bg_patches * (1 - alpha)


def compute_masked_color(images, masks):
    bg_color = torch.stack([images[:, i:i+1, ...][masks].mean() for i in range(images.shape[1])]).to(images.device)
    return bg_color.reshape(1, -1, 1, 1)


def load_target_sparse(target_fname, target_bg_fname, geom_fname, width, crop_n=10, patch_range_min=0.2, patch_range_max=0.5):
    # Load style image for the target.
    target = imread(target_fname)[..., :3].astype(np.float32) / 255.0
    min_dim = min(target.shape[0], target.shape[1])

    # Load geometry image for the target.
    geom = imread(geom_fname)[..., 1:2]
    assert target.shape[0:2] == geom.shape[0:2]

    zeros = np.where(geom == 0)
    yx = list(zip(list(zeros[0]), list(zeros[1])))
    random.shuffle(yx)

    geom = geom.astype(np.float32) / 255.0

    patch_generator = RandomPatchGenerator(width, patch_range=(patch_range_min, patch_range_max))
    patches = []
    ntries = 0
    while len(patches) < crop_n or ntries > crop_n * 10:
        pcenter = yx[ntries]
        ntries = ntries + 1
        rwidth = patch_generator.get_random_patch_size(target)
        half_rwidth = rwidth // 2

        start_row = max(0, pcenter[0] - half_rwidth)
        start_col = max(0, pcenter[1] - half_rwidth)

        res_patch = geom[start_row:start_row + rwidth, start_col:start_col + rwidth, 0]
        nzeros = np.sum(res_patch < 0.1)
        if nzeros > rwidth * rwidth * 0.05:
            #imsave('/tmp/actpatch/patch_%02d.png' % len(patches), (res_patch * 255).astype(np.uint8))
            patches.append([start_col, start_row, rwidth, rwidth])

    target_patches = [patch_generator.fixed_patch(target, *x, resize=True) for x in patches]
    geom_patches = [patch_generator.fixed_patch(geom, *x, resize=True) for x in patches]
    positions = torch.tensor([x[:2] for x in patches])
    positions = positions * width / (patch_range_min * min_dim)
    log_tensor(positions, 'positions', logger, print_stats=True)

    target_patches = torch.from_numpy(np.stack(target_patches)).permute(0, 3, 1, 2)
    geom_patches = torch.from_numpy(np.stack(geom_patches)).permute(0, 3, 1, 2)

    target_bg_patches = None
    if target_bg_fname is not None:
        target_bg = imread(target_bg_fname)[..., :3].astype(np.float32) / 255.0
        target_bg = target * geom + target_bg * (1 - geom)
        target_bg_patches = [patch_generator.fixed_patch(target_bg, *x, resize=True) for x in patches]
        target_bg_patches = torch.from_numpy(np.stack(target_bg_patches)).permute(0, 3, 1, 2)

    return target_patches, target_bg_patches, geom_patches, positions


def load_target(target_fname, geom_fname, width, crop_n=10, patch_range_min=0.2, patch_range_max=0.5, overfit_one=False):
    # Load style image for the target.
    target = imread(target_fname)[..., :3].astype(np.float32) / 255.0
    min_dim = min(target.shape[0], target.shape[1])

    # Load geometry image for the target.
    geom = imread(geom_fname).astype(np.float32) / 255.0
    assert target.shape[0:2] == geom.shape[0:2]

    if target.shape[0] == width and target.shape[1] == width:
        # Just flip a few times
        raise NotImplementedError()
    else:
        patch_generator = RandomPatchGenerator(width, patch_range=(patch_range_min, patch_range_max))

        if overfit_one:
            patch = patch_generator.random_patch(target, return_ind=True)[:-1]
            halfncrop = crop_n // 2

            pheight = patch[2]
            up_space = min(patch[1], pheight // 2)
            down_space = min(target.shape[0] - patch[1] - pheight, pheight // 2)

            patches = []
            for i in range(halfncrop // 2, 1, -1):
                patches.append([patch[0], patch[1] - int(up_space / halfncrop * 2 * i), patch[2], patch[3]])
            patches.append(patch)

            for i in range(1, halfncrop // 2):
                patches.append([patch[0], patch[1] + int(down_space / halfncrop * 2 * i), patch[2], patch[3]])

            patches = [x + [patch_generator.fixed_patch(target, *x, resize=True)] for x in patches]

        else:
            patches = [list(patch_generator.random_patch(target, return_ind=True)) for _ in range(crop_n)]
        target_patches = [x[-1] for x in patches]
        geom_patches = [patch_generator.fixed_patch(geom, *x[:-1], resize=True) for x in patches]
        positions = torch.tensor([x[:2] for x in patches])
        positions = positions * width / (patch_range_min * min_dim)
        log_tensor(positions, 'positions', logger, print_stats=True)

    target_patches = torch.from_numpy(np.stack(target_patches)).permute(0, 3, 1, 2)
    geom_patches = torch.from_numpy(np.stack(geom_patches)).permute(0, 3, 1, 2)[:, 1:2, ...]

    # Add flip augmentation
    if overfit_one:
        target_patches = torch.cat([target_patches, target_patches.permute(0, 1, 3, 2)], dim=0)
        geom_patches = torch.cat([geom_patches, geom_patches.permute(0, 1, 3, 2)], dim=0)
        positions = torch.cat([positions, positions], dim=0)

    forger.util.logging.log_tensor(geom_patches, 'geom_patches', logger, print_stats=True)

    return target_patches, geom_patches, positions  # image range [0...1]


def make_viz(style_patches, geom_patches, res_img, res_raw, bg_color=None, bg_images=None):
    def _add_alpha(img):
        return torch.cat([img, torch.ones_like(img[0:1,...])], dim=0)

    geom_viz = torchvision.utils.make_grid(geom_patches, nrow=geom_patches.shape[0], padding=0).cpu() * 2 - 1
    targets_viz = torchvision.utils.make_grid(style_patches, nrow=style_patches.shape[0], padding=0).cpu()
    results_viz = torchvision.utils.make_grid(res_img.detach().cpu(), nrow=geom_patches.shape[0], padding=0)
    rows = [_add_alpha(geom_viz), _add_alpha(targets_viz)]
    legend = ["geom", "target"]

    if bg_images is not None:
        bg_res_img = composite_with_bg_image(res_raw, bg_images)
        rows.append(
            _add_alpha(torchvision.utils.make_grid(bg_res_img.detach().cpu(), nrow=geom_patches.shape[0], padding=0)))
        legend.extend(['opt', 'opt (raw)'])
    elif bg_color is not None:
        bg_res_img = composite_with_bg_color(res_raw, bg_color)
        rows.append(_add_alpha(torchvision.utils.make_grid(bg_res_img.detach().cpu(), nrow=geom_patches.shape[0], padding=0)))
        legend.extend(['opt', 'opt (raw)'])
    else:
        legend.extend(['opt'])
    rows.append(_add_alpha(results_viz))

    rows.append(torchvision.utils.make_grid(
        forger.viz.visualize.compose_stroke_with_canvas(res_raw, "clear_stroke").detach().cpu(),
        nrow=geom_patches.shape[0], padding=0))
    legend.append('opt (clear)')

    legend_img = torch.cat([forger.viz.visualize.torch_image_with_text(txt, img_resolution) for txt in legend], dim=1)

    results_viz = ((torch.cat([_add_alpha(legend_img), torch.cat(rows, dim=1)],
         dim=2) / 2 + 0.5) * 255).clip(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
    return results_viz


def run_projection(
        engine,
        style_patches,
        geom_patches,
        positions,
        fname_prefix: str,
        fname_suffix: str,
        outdir: str,
        save_video: bool,
        num_steps: int,
        w_plus: bool,
        optimize_noise: bool,
        with_positions: bool,
        with_composite: bool,
        l1_fg_weight,
        bg_weight,
        resume_from=None,
        style_bg_patches=None
):
    noise_mode = 'const'  #('const' if optimize_noise else 'random')

    G = engine.G
    G.requires_grad_(False)
    img_resolution = engine.G.img_resolution
    device = engine.device

    positions = positions.to(device)
    style_patches = style_patches.to(device) * 2 - 1
    geom_patches = geom_patches.to(device)
    geom_feature = engine.encoder.encode(geom_patches)

    if style_bg_patches is not None:
        style_bg_patches = style_bg_patches.to(device) * 2 - 1

    norm_positions = None
    if with_positions:
        norm_positions = (positions % img_resolution) / (img_resolution - 1)

    video = None
    if save_video:
        video_ofile = f'{outdir}/{fname_prefix}_projviz_{fname_suffix}.mp4'
        video = imageio.get_writer(video_ofile, mode='I', fps=10, codec='libx264', bitrate='16M')

    # Optimize projection.
    start_time = perf_counter()
    result = project(
        G,
        norm_positions=norm_positions,
        target=style_patches,  # pylint: disable=not-callable
        geom=geom_patches,
        geom_feature=geom_feature,
        num_steps=num_steps,
        device=device,
        output_video=video,
        w_plus=w_plus,
        optimize_noise=optimize_noise,
        with_composite=with_composite,
        l1_fg_weight=l1_fg_weight,
        bg_weight=bg_weight,
        resume_from=resume_from,
        target_bg=style_bg_patches
    )
    logger.info(f'Elapsed: {(perf_counter()-start_time):.1f} s')

    res_img, res_raw = G.synthesis(
        result['w'].to(device).expand(style_patches.shape[0], -1 if w_plus else G.mapping.num_ws, -1),
        geom_feature=geom_feature, noise_mode=noise_mode,
        norm_noise_positions=norm_positions, return_debug_data=True)
    results_viz = make_viz(style_patches, geom_patches, res_img, res_raw,
                           bg_color=result['bg'].to(device) if ('bg' in result and with_composite) else None,
                           bg_images=style_bg_patches)
    imsave(os.path.join(outdir, f'{fname_prefix}_projviz_{fname_suffix}.png'), results_viz)


    np.savez(os.path.join(outdir, f'{fname_prefix}_projected_{fname_suffix}.npz'),
             **result)

    # HACK



    # np.savez('/tmp/ws2.npz', w=projected_w.detach().cpu().numpy(),
    #          geom0=geom_feature[0].detach().cpu().numpy(),
    #          geom1=geom_feature[1].detach().cpu().numpy())
    #
    # npz = np.load('/tmp/ws2.npz')
    # ws = torch.from_numpy(npz['w']).to(device).expand(style_patches.shape[0], -1, -1)
    # geom_feature2 = [torch.from_numpy(npz['geom0']).to(device), torch.from_numpy(npz['geom1']).to(device)]
    # debug_vis = torchvision.utils.make_grid(
    #     G.synthesis(ws.expand(style_patches.shape[0], -1 if w_plus else G.mapping.num_ws, -1),
    #                 geom_feature=geom_feature2, noise_mode=noise_mode).detach().cpu(),
    #     nrow=geom_patches.shape[0], padding=0)
    # imsave('/tmp/debugviz2.png', ((debug_vis / 2 + 0.5) * 255).clip(0, 255).to(torch.uint8).permute(1, 2, 0).numpy())

    # END OF HACK

    if video is not None:
        video.close()

    return result


def parse_patch_range(arg_val):
    res = [float(x) for x in arg_val.split(',')]
    assert len(res) == 2
    assert 0 < res[0] <= 1
    assert 0 < res[1] <= 1
    assert res[0] <= res[1]
    return res[0], res[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gan_checkpoint', action='store', type=str, required=True)
    parser.add_argument('--encoder_checkpoint', action='store', type=str, required=False)
    parser.add_argument('--output_dir', action='store', type=str, required=True)
    parser.add_argument('--geom_image', type=str, default=None,
                        help='The path to the geometry image. Required if using a forger model.')
    parser.add_argument('--output_style_id', action='store', default=None,
                        help='If not set, derived from target_image filename.')
    parser.add_argument('--target_image', type=str, required=True,
                        help='The path to the target image')
    parser.add_argument('--target_bg_image', type=str, required=False,
                        help='Target image to use as background in with_composite mode')
    parser.add_argument('--save_video', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--w_plus', action='store_true')
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--no_noise', action='store_true')
    parser.add_argument('--num_crops', action='store', type=int, default=10)
    parser.add_argument('--patch_scale_range', action='store', type=str, default='0.2,0.5',
                        help='CSV min and max scale of patches, e.g. 0.2,0.5')
    parser.add_argument('--with_positions', action='store_true')
    parser.add_argument('--overfit_one', action='store_true')
    parser.add_argument('--skip_existing', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--with_composite', action='store_true')
    parser.add_argument('--l1_fg_weight', action='store', type=float, default=0)
    parser.add_argument('--bg_weight', action='store', type=float, default=0)
    forger.util.logging.add_log_level_flag(parser)
    args = parser.parse_args()

    if args.output_style_id is not None:
        fname_prefix = args.output_style_id
    else:
        fname_prefix = '.'.join(os.path.basename(args.target_image).split('.')[:-1])
    fname_suffix = 'wplus' if args.w_plus else 'w'
    fname_suffix = fname_suffix + '_' + ('fixednoise' if args.no_noise else 'optnoise')
    fname_suffix = fname_suffix + f'_ncrop{args.num_crops}'
    fname_all_pkl = os.path.join(args.output_dir, f'ALL_projected_{fname_suffix}.pkl')

    logdir = os.path.join(args.output_dir, 'logs')
    os.makedirs(logdir, exist_ok=True)
    forger.util.logging.default_log_setup(
        args.log_level, filename=os.path.join(logdir, f'{fname_prefix}_LOG_{fname_suffix}.txt'))
    device = torch.device(0)

    resume_val = None
    if os.path.isfile(fname_all_pkl):
        if args.skip_existing or args.resume:
            with open(fname_all_pkl, 'rb') as f:
                resume_val = pickle.load(f).get(fname_prefix)
            if args.skip_existing and resume_val is not None:
                logger.info(f'All pickle already has projection for {fname_prefix}, skipping: {fname_all_pkl}')
                exit(0)
            if args.resume and resume_val is None:
                logger.info(f'HACK skipping if cannot resume for {fname_prefix}, skipping: {fname_all_pkl}')
                exit(0)
    if not args.resume:
        resume_val = None

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    engine = forger.ui.brush.PaintEngineFactory.create(
        encoder_checkpoint=args.encoder_checkpoint,
        gan_checkpoint=args.gan_checkpoint,
        device=device)

    img_resolution = engine.G.img_resolution
    pmin, pmax = parse_patch_range(args.patch_scale_range)
    if args.target_bg_image is None:
        style_bg_patches = None
        style_patches, geom_patches, positions = load_target(
            args.target_image, args.geom_image, img_resolution, crop_n=args.num_crops,
            patch_range_min=pmin, patch_range_max=pmax, overfit_one=args.overfit_one)
    else:
        style_patches, style_bg_patches, geom_patches, positions = load_target_sparse(
            args.target_image, args.target_bg_image, args.geom_image, img_resolution, crop_n=args.num_crops,
            patch_range_min=pmin, patch_range_max=pmax)

    result = run_projection(
        engine=engine,
        style_patches=style_patches,
        style_bg_patches=style_bg_patches,
        geom_patches=geom_patches,
        positions=positions,
        fname_prefix=fname_prefix,
        fname_suffix=fname_suffix,
        outdir=args.output_dir,
        save_video=args.save_video,
        num_steps=args.num_steps,
        w_plus=args.w_plus,
        optimize_noise=(not args.no_noise),
        with_positions=args.with_positions,
        with_composite=args.with_composite,
        l1_fg_weight=args.l1_fg_weight,
        bg_weight=args.bg_weight,
        resume_from=resume_val
    )

    all_data = {}
    if os.path.isfile(fname_all_pkl):
        with open(fname_all_pkl, 'rb') as f:
            all_data = pickle.load(f)
    if fname_prefix in all_data:
        logger.warning(f'All pickle already has projection for {fname_prefix}, overwriting entry in: {fname_all_pkl}')
    all_data[fname_prefix] = result

    with open(fname_all_pkl, 'wb') as f:
        pickle.dump(all_data, f)
        logger.info(f'Added w entry for {fname_prefix} to {fname_all_pkl}')

