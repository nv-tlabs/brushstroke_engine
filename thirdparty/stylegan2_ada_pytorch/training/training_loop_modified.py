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
import time
import copy
import json
import pickle


import psutil
import PIL.Image
import numpy as np
from skimage.io import imsave
import torch
import torchvision
import torchvision.transforms.functional
import wandb
import dnnlib
import forger.experimental.autoenc as autoenc
from torch_utils import misc
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix

import legacy
from metrics import metric_main
from metrics.frechet_inception_distance import forger_compute_fid

# Forger imports
import forger.experimental.autoenc.factory as factory
import forger.util.logging
import forger.metrics.util
import forger.ui.library
import forger.metrics.metric_main
import forger.train.stitching
import forger.viz.visualize
from forger.viz.visualize import compose_stroke, compose_stroke_with_canvas, save_image_grid, output_encoder_diagnostics
from forger.util.torch_data import get_image_data_iterator_from_dataset

logger = logging.Logger(__name__)


def training_loop(
        run_dir,                       # Output directory.
        rank,                          # Rank of the current process in [0, num_gpus[.
        num_gpus,                      # Number of GPUs participating in the training.
        image_snapshot_ticks,          # How often to save image snapshots? None = disable.
        network_snapshot_ticks,        # How often to save network snapshots? None = disable.
        random_seed,                   # Global random seed.

        # Metrics
        metrics,                       # Metrics to evaluate during training, supports "fid" and "forger"
        num_fid_items,                 # Parameter of the fid metric
        num_forgermetric_styles,       # Parameter of the forger metrics

        # Data
        style_set_kwargs,              # Options for the style set.
        geom_set_kwargs,               # Options for the geometry set
        data_loader_kwargs,            # Options for torch.utils.data.DataLoader.
        geom_input_channel,
        geom_truth_channel,

        # Networks
        output_resolution,             # If not none, will cut patches from style and geom sets
        geom_encoder_checkpt,          # Path to autoencoder for geometry.
        G_kwargs,                      # Options for generator network.
        D_kwargs,                      # Options for discriminator network.
        G_opt_kwargs,                  # Options for generator optimizer.
        D_opt_kwargs,                  # Options for discriminator optimizer.
        geom_inject_resolutions,       # list, 0 - enc bottleneck, 1 - one layer up in the decoder, 2 - two layers up...

        # Base config
        batch_size,                    # Total batch size for one training iteration. Can be larger than batch_gpu * num_gpus.
        batch_gpu,                     # Number of samples processed at a time by one GPU.
        ema_kimg,                      # Half-life of the exponential moving average (EMA) of generator weights.
        ema_rampup,                    # EMA ramp-up coefficient.
        total_kimg,                    # Total length of the training, measured in thousands of real images.

        # Augmentation
        augment_kwargs,                # Options for augmentation pipeline. None = disable.
        ada_target,                    # ADA target value. None = fixed p.

        # Loss
        loss_kwargs,                   # Options for loss function.

        # Geometry training
        geom_interval,                 # How often to update G based on geometry loss?
        geom_lr,                       # Learning rate for geometry phase (needed?)
        geom_warmstart_kimg,
        geom_phase_mode,
        geom_warmstart_mode,
        geom_warmstart_start_kimg,
        exit_after_warmstart,
        freeze_geom_linear_layer_in_nongeom_phases,

        # Stotching
        stitch_interval,

        resume_pkl,                    # Network pickle to resume training from.
        cudnn_bench,                   # Enable torch.backends.cudnn.benchmark?
        wandb_project,                 # The name of the wandb project or None
        wandb_group,                   # The name of the wandb experiment group or None
        argparse_args,                 # A argparse.Namespace object, to be used for initializing the wandb experiment

        geom_metric_set_kwargs = None, # If set will use a separate dataset for FID eval
        G_reg_interval     = 4,        # How often to perform regularization for G? None = disable lazy regularization.
        D_reg_interval     = 16,       # How often to perform regularization for D? None = disable lazy regularization.
        ada_interval       = 4,        # How often to perform ADA adjustment?
        ada_kimg           = 500,      # ADA adjustment speed, measured in how many kimg it takes for p to increase/decrease by one unit.
        augment_p          = 0,        # Initial value of augmentation probability.
        kimg_per_tick      = 4,        # Progress snapshot interval.
        abort_fn           = None,     # Callback function for determining whether to abort training. Must return consistent results across ranks.
        progress_fn        = None,     # Callback function for updating training progress. Called for all ranks.
):
    # Initialize.
    start_time = time.time()
    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.benchmark = cudnn_bench    # Improves training speed.
    conv2d_gradfix.enabled = True                       # Improves training speed.
    grid_sample_gradfix.enabled = True                  # Avoids errors with the augmentation pipe.

    # Load training set.
    if rank == 0:
        print('Loading training set...')
    style_set_kwargs.resolution = output_resolution
    style_set_kwargs.resize_mode = 'crop'
    style_set = dnnlib.util.construct_class_by_name(**style_set_kwargs)  # subclass of training.dataset.Dataset
    style_set_sampler = misc.InfiniteSampler(dataset=style_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    style_set_iterator = iter(torch.utils.data.DataLoader(dataset=style_set, sampler=style_set_sampler, batch_size=batch_size//num_gpus, **data_loader_kwargs))

    geom_set = dnnlib.util.construct_class_by_name(**geom_set_kwargs)  # subclass of training.dataset.Dataset
    geom_set_sampler = misc.InfiniteSampler(dataset=geom_set, rank=rank, num_replicas=num_gpus, seed=random_seed)
    geom_set_iterator = iter(torch.utils.data.DataLoader(dataset=geom_set, sampler=geom_set_sampler, batch_size=batch_size // num_gpus, **data_loader_kwargs))

    # This one is auto-cropped
    if geom_metric_set_kwargs is None:
        geom_metric_set = dnnlib.util.construct_class_by_name(**geom_set_kwargs, resolution=output_resolution, resize_mode='crop')
    else:
        geom_metric_set = dnnlib.util.construct_class_by_name(**geom_metric_set_kwargs, resolution=output_resolution, resize_mode='crop')
    geom_metric_set_sampler = misc.InfiniteSampler(dataset=geom_metric_set, rank=rank, num_replicas=num_gpus, seed=random_seed)


    if output_resolution is None:
        output_resolution = style_set.resolution
    assert output_resolution == style_set.resolution

    crop_transform = None
    if geom_set.resolution != output_resolution:
        crop_transform = torchvision.transforms.RandomCrop(output_resolution)

    if rank == 0:
        print('==Style Data==')
        style_set.print_info()
        print('==Geometry Data==')
        geom_set.print_info()
        if geom_set != geom_metric_set:
            print('==Geometry Data (for FID)==')
            geom_metric_set.print_info()

    # Autoencoder (of type BaseGeoEncoder) and diagnostics
    geom_autoencoder = autoenc.factory.create_autoencoder_from_checkpoint(geom_encoder_checkpt)
    geom_autoencoder.eval().requires_grad_(False).to(device)
    geom_autoencoder.set_default_encode_resolutions(geom_inject_resolutions)

    # Construct networks in StyleGAN.
    if rank == 0:
        print('Constructing networks...')

    # Update the geometry encoder-relevant parameters
    # TODO: why doesn't logging work here?
    G_kwargs.synthesis_kwargs.geom_feature_channels = [geom_autoencoder.feature_channels(x) for x in geom_inject_resolutions]
    print(f'Embedding channels of the geometry encoder: {G_kwargs.synthesis_kwargs.geom_feature_channels}')
    G_kwargs.synthesis_kwargs.geom_feature_resolutions = [geom_autoencoder.featuremap_resolution(output_resolution, x) for x in geom_inject_resolutions]
    print(f'Geometry feature resolutions: {G_kwargs.synthesis_kwargs.geom_feature_resolutions}')

    common_kwargs = dict(c_dim=style_set.label_dim, img_resolution=output_resolution, img_channels=style_set.num_channels)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()

    # Resume from existing pickle.
    if (resume_pkl is not None) and (rank == 0):
        print(f'Resuming from "{resume_pkl}"')
        with dnnlib.util.open_url(resume_pkl) as f:
            resume_data = legacy.load_network_pkl(f)
        for name, module in [('G', G), ('D', D), ('G_ema', G_ema)]:
            misc.copy_params_and_buffers(resume_data[name], module, require_all=False)

    # Important! This line with DataLoader must be after Style GAN initialization -- there is some bizarre side effect
    # on the random state that cause the StyleGAN initialization to skew pink if this line is before;
    # with this line after the initialization skews blue and controllability is much better.
    # TODO: quantify and draw conclusions about what matters for initialization
    # TODO: debug the bizarre effect of this and file a bug, if appropriate
    geom_metric_set_iterator = iter(
        torch.utils.data.DataLoader(dataset=geom_metric_set, sampler=geom_metric_set_sampler,
                                    batch_size=batch_size // num_gpus,
                                    **data_loader_kwargs))
    viz_dir = os.path.join(run_dir, 'viz')
    if rank == 0:
        logger.info("Creating visualization directory...")
        os.makedirs(viz_dir, exist_ok=True)
        with torch.no_grad():
            logger.info("Outputting encoder diagnostic images...")
            output_encoder_diagnostics(next(geom_metric_set_iterator)[0], geom_autoencoder, device, viz_dir,
                                       geom_input_channel=geom_input_channel)
    misc.reclaim_cuda_memory()

    # Print network summary tables.
    if rank == 0:
        z = torch.empty([batch_gpu, G.z_dim], device=device)
        c = torch.empty([batch_gpu, G.c_dim], device=device)
        # TODO
        geom_feature = [torch.empty([batch_gpu, G_kwargs.synthesis_kwargs.geom_feature_channels[i],
                                    G_kwargs.synthesis_kwargs.geom_feature_resolutions[i],
                                    G_kwargs.synthesis_kwargs.geom_feature_resolutions[i]
                                    ], device=device) for i in range(len(geom_inject_resolutions))]
        img = misc.print_module_summary(G, [z, c, geom_feature])  # Generator is defaulted to return color triad data as well
        misc.print_module_summary(D, [img, c])

    # Setup augmentation.
    if rank == 0:
        print('Setting up augmentation...')
    augment_pipe = None
    ada_stats = None
    if (augment_kwargs is not None) and (augment_p > 0 or ada_target is not None):
        augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
        augment_pipe.p.copy_(torch.as_tensor(augment_p))
        if ada_target is not None:
            ada_stats = training_stats.Collector(regex='Loss/signs/real')

    # Distribute across GPUs.
    if rank == 0:
        print(f'Distributing across {num_gpus} GPUs...')
    ddp_modules = dict()

    for name, module in [('G', G), ('D', D), ('geom_encoder', geom_autoencoder), (None, G_ema), ('augment_pipe', augment_pipe)]:
        if (num_gpus > 1) and (module is not None) and len(list(module.parameters())) != 0:
            module.requires_grad_(True)
            # TODO: what does this do?
            module = torch.nn.parallel.DistributedDataParallel(module, device_ids=[device], broadcast_buffers=False)
            module.requires_grad_(False)
        if name is not None:
            ddp_modules[name] = module

    # Setup training phases.
    if rank == 0:
        print('Setting up training phases...')

    stitcher = forger.train.stitching.RandomStitcher()
    forger_loss = dnnlib.util.construct_class_by_name(device=device, stitcher=stitcher, **ddp_modules, **loss_kwargs)  # subclass of training.loss.Loss
    G_orig = None
    if forger_loss.requires_frozen_generator():
        G_orig = copy.deepcopy(G).eval()

    phases = []

    def _get_prep_module(_mname):
        if freeze_geom_linear_layer_in_nongeom_phases and _mname == 'G':
            return lambda module: module.set_trainable_layers('all_but_linear')
        return None

    for name, module, opt_kwargs, reg_interval in [('G', G, G_opt_kwargs, G_reg_interval), ('D', D, D_opt_kwargs, D_reg_interval)]:
        if reg_interval is None:
            opt = dnnlib.util.construct_class_by_name(params=module.parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'all', module=module, opt=opt, interval=1)]
        else:  # Lazy regularization.
            mb_ratio = reg_interval / (reg_interval + 1)
            opt_kwargs = dnnlib.EasyDict(opt_kwargs)
            opt_kwargs.lr = opt_kwargs.lr * mb_ratio
            opt_kwargs.betas = [beta ** mb_ratio for beta in opt_kwargs.betas]
            opt = dnnlib.util.construct_class_by_name(module.parameters(), **opt_kwargs)  # subclass of torch.optim.Optimizer
            phases += [dnnlib.EasyDict(name=name+'main', module=module, opt=opt, interval=1,
                                       prep_module=_get_prep_module(name))]
            phases += [dnnlib.EasyDict(name=name+'reg', module=module, opt=opt, interval=reg_interval,
                                       prep_module=_get_prep_module(name))]
            # Add stitching phase with the same optimizer as G
            if name == 'G' and stitch_interval > 0:
                phases += [dnnlib.EasyDict(name='Gstitch', module=module, opt=opt, interval=stitch_interval,
                                           prep_module=_get_prep_module(name))]

    # TODO: Why is there a separate optimizer? Can we use regular G optimizer?
    geom_opt = dnnlib.util.construct_class_by_name(class_name='torch.optim.Adam', params=G.parameters(), lr=geom_lr,
                                                  betas=[0, 0.99], eps=1e-8)
    if geom_interval > 0:
        logger.debug(f"Geom interval is {geom_interval}")
        phases += [dnnlib.EasyDict(name='Ggeom', module=G, opt=geom_opt, interval=geom_interval,
                                   prep_module=lambda module:module.set_trainable_layers(geom_phase_mode))]

    warmstart_phases = []
    if geom_warmstart_kimg > 0:
        warmstart_phases = [dnnlib.EasyDict(name='Ggeom-warm', module=G, opt=geom_opt, interval=1,
                                            prep_module=lambda module:module.set_trainable_layers(geom_warmstart_mode))]

    for phase in phases + warmstart_phases:
        phase.start_event = None
        phase.end_event = None
        if rank == 0:
            phase.start_event = torch.cuda.Event(enable_timing=True)
            phase.end_event = torch.cuda.Event(enable_timing=True)

    # Export sample images and initialize visualization auxiliary variables
    visualizer = None
    if rank == 0 and image_snapshot_ticks is not None:
        visualizer = forger.viz.visualize.TrainingVisualizer(device, batch_gpu, width=output_resolution)
        visualizer.init(viz_dir, style_set, geom_metric_set, geom_autoencoder, G.z_dim, geom_input_channel=geom_input_channel)
        visualizer.do_visualize(viz_dir, G_ema, 'init')
        misc.reclaim_cuda_memory()

    # Initialize logs and wandb.
    if rank == 0:
        if wandb_project is not None:
            assert isinstance(argparse_args, argparse.Namespace)
            logger.info("Initializing wandb...")
            # Initialize wandb directory
            wandb_dir = os.path.join(run_dir, 'wandb')
            os.makedirs(wandb_dir, exist_ok=True)
            wandb.init(
                project=wandb_project,
                group=wandb_group,
                name=os.path.basename(run_dir),
                config=argparse_args,
                resume=False,
                dir=wandb_dir,
                sync_tensorboard=True
            )
    logger.info('Initializing logs...')
    stats_collector = training_stats.Collector(regex='.*')
    stats_metrics = dict()
    stats_jsonl = None
    stats_tfevents = None
    if rank == 0:
        stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'wt')
        try:
            import torch.utils.tensorboard as tensorboard
            stats_tfevents = tensorboard.SummaryWriter(run_dir)
        except ImportError as err:
            print('Skipping tfevents export:', err)

    # Train.
    if rank == 0:
        print(f'Training for {total_kimg} kimg...')
        print()
    cur_nimg = 0
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    batch_idx = 0

    if progress_fn is not None:
        progress_fn(0, total_kimg)
    while True:
        is_geom_warmstart = (cur_nimg > geom_warmstart_start_kimg * 1000) and \
                            (cur_nimg - geom_warmstart_start_kimg * 1000 < geom_warmstart_kimg * 1000)
        is_last_geom_warmstart = (
                is_geom_warmstart and
                (cur_nimg + batch_size - geom_warmstart_start_kimg * 1000) >= geom_warmstart_kimg * 1000)
        if is_last_geom_warmstart:
            print(f'Finishing up geometry warm-starting... (iteration {batch_idx})')
        if is_geom_warmstart:
            current_phases = warmstart_phases
        else:
            current_phases = phases

        # Fetch training data.
        with torch.autograd.profiler.record_function('style_data_fetch'):
            # Fetch style data
            phase_real_style_img, phase_real_style_c = next(style_set_iterator)
            # phase_real_style_img is within the range [-1, 1]
            phase_real_style_img = (phase_real_style_img.to(device).to(torch.float32) / 127.5 - 1).split(batch_gpu)
            phase_real_style_c = phase_real_style_c.to(device).split(batch_gpu)
            all_gen_z = torch.randn([len(phases) * batch_size, G.z_dim], device=device)
            all_gen_z = [phase_gen_z.split(batch_gpu) for phase_gen_z in all_gen_z.split(batch_size)]
            all_gen_c = [style_set.get_label(np.random.randint(len(style_set))) for _ in range(len(phases) * batch_size)]
            all_gen_c = torch.from_numpy(np.stack(all_gen_c)).pin_memory().to(device)
            all_gen_c = [phase_gen_c.split(batch_gpu) for phase_gen_c in all_gen_c.split(batch_size)]

        with torch.autograd.profiler.record_function('geometry_data_fetch'):
            # Fetch geometry data (Convert to [0.0, 1.0])
            phase_geom, _ = next(geom_set_iterator)  # This is not necessarily same size as needed
            phase_geom = phase_geom.to(device).to(torch.float32) / 255.0
            phase_real_geom_img_FULL = phase_geom[:, geom_input_channel:geom_input_channel + 1, :, :]

            # Save crop parameters in order do optionally do double crop and stitching
            if crop_transform is not None:
                crop_params = crop_transform.get_params(phase_real_geom_img_FULL, (output_resolution, output_resolution))

            phase_real_geom_img = torchvision.transforms.functional.crop(phase_real_geom_img_FULL, *crop_params)
            phase_real_geom_img = phase_real_geom_img.split(batch_gpu)
            phase_real_geom_img_FULL = phase_real_geom_img_FULL.split(batch_gpu)

            if geom_input_channel == geom_truth_channel:
                phase_real_geom_truth = phase_real_geom_img
            else:
                phase_real_geom_truth = torchvision.transforms.functional.crop(
                    phase_geom[:, geom_truth_channel:geom_truth_channel + 1, :, :], *crop_params)
                phase_real_geom_truth = phase_real_geom_truth.split(batch_gpu)
            del phase_geom

            with torch.no_grad():
                phase_geom_feature = [geom_autoencoder.encode(_x) for _x in phase_real_geom_img]

        # Execute training phases.
        for phase, phase_gen_z, phase_gen_c in zip(current_phases, all_gen_z, all_gen_c):
            if batch_idx % phase.interval != 0:
                continue

            if phase.name == 'Gstitch':
                # Get 2nd crops
                crop_params2 = stitcher.gen_overlapping_square_crop(
                    phase_real_geom_img_FULL[0].shape[-1], crop_params)
                phase_real_geom_img2 = [
                    torchvision.transforms.functional.crop(phase_real_geom_img_FULL[i], *crop_params2)
                    for i in range(len(phase_real_geom_img_FULL))]

                with torch.no_grad():
                    phase_geom_feature2 = [geom_autoencoder.encode(_x) for _x in phase_real_geom_img2]

            # Initialize gradient accumulation.
            if phase.start_event is not None:
                phase.start_event.record(torch.cuda.current_stream(device))
            phase.opt.zero_grad(set_to_none=True)
            phase.module.requires_grad_(True)

            if phase.get('prep_module') is not None:
                phase.get('prep_module')(phase.module)  # E.g. freezes layers etc

            # Accumulate gradients over multiple rounds.
            for round_idx, (
                    real_style,
                    real_c,
                    real_geom,
                    real_geom_truth,
                    geom_feature,
                    gen_z,
                    gen_c
            ) in enumerate(zip(
                phase_real_style_img,
                phase_real_style_c,
                phase_real_geom_img,
                phase_real_geom_truth,
                phase_geom_feature,
                phase_gen_z,
                phase_gen_c
            )):
                sync = (round_idx == batch_size // (batch_gpu * num_gpus) - 1)
                gain = phase.interval

                if phase.name == 'Gstitch':
                    forger_loss.accumulate_gradients_stitch(
                        geom_feature1=geom_feature,
                        geom_feature2=phase_geom_feature2[round_idx],
                        crop1=crop_params,
                        crop2=crop_params2,
                        gen_z=gen_z,
                        gen_c=gen_c,
                        gain=1)
                else:
                    # TODO: I hate this loss class which is so much more than loss
                    forger_loss.accumulate_gradients(
                        phase=phase.name,
                        real_style=real_style,
                        real_c=real_c,
                        real_geom=real_geom_truth,
                        geom_feature=geom_feature,
                        gen_z=gen_z,
                        gen_c=gen_c,
                        sync=sync,
                        gain=gain,
                        G_orig=G_orig
                    )

                misc.reclaim_cuda_memory()
                # print(f'PHASE ---- {phase.name} -- {round_idx}')
                # misc.print_memory_diagnostics(device)

            # Update weights.
            phase.module.requires_grad_(False)
            with torch.autograd.profiler.record_function(phase.name + '_opt'):
                for param_name, param in phase.module.named_parameters():
                    if param.grad is not None:
                        #_tmean = torch.mean(param.grad)
                        misc.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
                        #print('{} ({}) grad: {} (orig {})'.format(param_name, torch.mean(param.data), torch.mean(param.grad), _tmean))
                    # else:
                    #     print('{} grad: None'.format(param_name))
                phase.opt.step()
            if phase.end_event is not None:
                phase.end_event.record(torch.cuda.current_stream(device))

        # Update G_ema.
        with torch.autograd.profiler.record_function('Gema'):
            ema_nimg = ema_kimg * 1000
            if ema_rampup is not None:
                ema_nimg = min(ema_nimg, cur_nimg * ema_rampup)
            ema_beta = 0.5 ** (batch_size / max(ema_nimg, 1e-8))
            for p_ema, p in zip(G_ema.parameters(), G.parameters()):
                p_ema.copy_(p.lerp(p_ema, ema_beta))
            for b_ema, b in zip(G_ema.buffers(), G.buffers()):
                b_ema.copy_(b)

        # Update state.
        cur_nimg += batch_size
        batch_idx += 1

        # Execute ADA heuristic.
        if not is_geom_warmstart and (ada_stats is not None) and (batch_idx % ada_interval == 0):
            ada_stats.update()
            adjust = np.sign(ada_stats['Loss/signs/real'] - ada_target) * (batch_size * ada_interval) / (ada_kimg * 1000)
            augment_pipe.p.copy_((augment_pipe.p + adjust).max(misc.constant(0, device=device)))

        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000) or (is_last_geom_warmstart and exit_after_warmstart)
        if (not done) and (not is_last_geom_warmstart) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in stats_collector.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<8.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        fields += [f"augment {training_stats.report0('Progress/augment', float(augment_pipe.p.cpu()) if augment_pipe is not None else 0):.3f}"]
        training_stats.report0('Timing/total_hours', (tick_end_time - start_time) / (60 * 60))
        training_stats.report0('Timing/total_days', (tick_end_time - start_time) / (24 * 60 * 60))
        if rank == 0:
            print(' '.join(fields))

        # Check for abort.
        if (not done) and (abort_fn is not None) and abort_fn():
            done = True
            if rank == 0:
                print()
                print('Aborting...')

        # Save image snapshot.
        if ((rank == 0) and (image_snapshot_ticks is not None) and
                (done or is_last_geom_warmstart or
                 (cur_tick > 0 and (cur_tick % image_snapshot_ticks == 0)))):
            visualizer.do_visualize(
                viz_dir, G_ema, f'warmstarted' if is_last_geom_warmstart else f'{cur_nimg//1000:06d}')
            misc.reclaim_cuda_memory()

        # Save network snapshot.
        snapshot_pkl = None
        snapshot_data = None
        if (network_snapshot_ticks is not None) and (done or is_last_geom_warmstart or (cur_tick % network_snapshot_ticks == 0)):
            snapshot_data = dict(training_set_kwargs=dict(style_set_kwargs))
            for name, module in [('G', G), ('D', D), ('G_ema', G_ema), ('augment_pipe', augment_pipe)]:
                if module is not None:
                    if num_gpus > 1:
                        misc.check_ddp_consistency(module, ignore_regex=r'.*\.w_avg')
                    module = copy.deepcopy(module).eval().requires_grad_(False).cpu()
                snapshot_data[name] = module
                del module  # conserve memory
            snapshot_data['args'] = argparse_args  # Store all the flags in the snapshot
            snapshot_data['encoder'] = torch.load(geom_encoder_checkpt)  # Also copy encoder for easy model sharing later
            snapshot_pkl = os.path.join(
                run_dir,
                'network-snapshot-' + ('warmstarted.pkl' if is_last_geom_warmstart else f'{cur_nimg//1000:06d}.pkl'))
            if rank == 0:
                with open(snapshot_pkl, 'wb') as f:
                    pickle.dump(snapshot_data, f)

        # Evaluate metrics.
        if snapshot_data is not None:
            del snapshot_data  # snapshot ticks is tied to metric ticks, which makes sense, but we don't need this data

            if (len(metrics) > 0) and rank == 0:  # TODO: support multi-GPU computation?
                print('Evaluating metrics...' + ('after warmstart' if is_last_geom_warmstart else ''))

                eval_batch_size = 16   # Note: must match data loader batch size
                random_state = forger.metrics.util.RandomState(0)  # TODO: same seed ok?
                stroke_generator = forger.metrics.util.PaintStrokeGenerator.create(
                    encoder_checkpoint=geom_encoder_checkpt,
                    gan_checkpoint=snapshot_pkl,
                    device=device,
                    batch_size=eval_batch_size,
                    random_state=random_state)
                stroke_generator.set_geometry_source_from_iterator(
                    get_image_data_iterator_from_dataset(geom_metric_set, eval_batch_size), eval_batch_size,
                    geom_input_channel=geom_input_channel)

                stats_metrics = {}
                for m in metrics:
                    if m == "fid":
                        print('Computing forger FID...')
                        stats_metrics.update(
                            forger_compute_fid(stroke_generator, style_set_kwargs,
                                               device=device, num_gpus=num_gpus, rank=rank,
                                               num_items=num_fid_items))
                    elif m == "forger":
                        print('Computing forger losses...')
                        nbatches_per_style = 1
                        style_library = forger.ui.library.RandomBrushLibrary(
                            num_forgermetric_styles, G.z_dim, random_state)
                        stats_metrics.update(
                            forger.metrics.metric_main.paint_engine_metric_loop(
                                stroke_generator, style_library, nbatches_per_style,
                                geom_set_iterator,
                                stitcher,
                                geom_input_channel))

                forger.metrics.metric_main.summary_losses_to_file(
                    os.path.join(run_dir, 'summary_metrics.txt'), stats_metrics,
                    step=int(cur_nimg / 1e3), do_print=True)

        # Collect statistics.
        for phase in phases:
            value = []
            try:
                if (phase.start_event is not None) and (phase.end_event is not None):
                    phase.end_event.synchronize()
                    value = phase.start_event.elapsed_time(phase.end_event)
                training_stats.report0('Timing/' + phase.name, value)
            except Exception as e:  # should only happen if not all phases ran on the first iteration
                pass
        stats_collector.update()
        stats_dict = stats_collector.as_dict()

        # Update logs.
        timestamp = time.time()
        if stats_jsonl is not None:
            fields = dict(stats_dict, timestamp=timestamp)
            stats_jsonl.write(json.dumps(fields) + '\n')
            stats_jsonl.flush()
        if stats_tfevents is not None:
            global_step = int(cur_nimg / 1e3)
            walltime = timestamp - start_time
            for name, value in stats_dict.items():
                stats_tfevents.add_scalar(name, value.mean, global_step=global_step, walltime=walltime)
            for name, value in stats_metrics.items():
                stats_tfevents.add_scalar(f'Metrics/{name}', value, global_step=global_step, walltime=walltime)
            stats_tfevents.flush()
        if progress_fn is not None:
            progress_fn(cur_nimg // 1000, total_kimg)

        # Skip updating state if we just performed maintenance because of warmstart
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    if rank == 0:
        print()
        print('Exiting...')

#----------------------------------------------------------------------------
