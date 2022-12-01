# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Train a GAN using the techniques described in the paper
"Training Generative Adversarial Networks with Limited Data"."""
import argparse
import logging
import os
import re
import json
import tempfile
import torch
import torch.distributed
import dnnlib
import dnnlib.util
from typing import Optional

from training import training_loop_modified
from metrics import metric_main
from torch_utils import training_stats
from torch_utils import custom_ops

import forger.util.logging
import forger.train.losses

logger = logging.Logger(__name__)

#----------------------------------------------------------------------------

class UserError(Exception):
    pass


def get_train_set_kwargs(data_str, needs_labels, subset=None, xflip=False, random_seed=0):
    parts = data_str.split(':')
    if len(parts) == 2:
        name = parts[0]
        path = parts[1]
    else:
        name = os.path.basename(data_str)
        path = data_str

    kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset',
                             path=path, name=name, use_labels=False, max_size=None, xflip=xflip)
    # TODO: restore if needed
    # training_set = dnnlib.util.construct_class_by_name(**kwargs)
    # kwargs.resolution = training_set.resolution  # be explicit about resolution
    # kwargs.use_labels = training_set.has_labels  # be explicit about labels
    # kwargs.max_size = len(training_set)  # be explicit about dataset size
    # del training_set

    if needs_labels:
        assert kwargs.use_labels, '--cond=True requires labels specified in dataset.json'
    else:
        kwargs.use_labels = False

    if subset is not None:
        assert isinstance(subset, int)
        if not 1 <= subset <= kwargs.max_size:
            raise UserError(f'subset ({subset}) must be between 1 and {kwargs.max_size}')
        if subset < kwargs.max_size:
            kwargs.max_size = subset
            kwargs.random_seed = random_seed

    return kwargs


def lr_description(g_lrate, d_lrate, default_lr=0.0001):
    EPS = default_lr / 100

    def _eq(a, b):
        return abs(a - b) < EPS

    if _eq(g_lrate, d_lrate):
        if not _eq(g_lrate, default_lr):
            return '-lr{}'.format(g_lrate)
        else:
            return ''  # Avoid verbose descriptions

    return '-glr{}-dlr{}'.format(g_lrate, d_lrate)


def zw_description(z_dim, w_dim, color_w_channels):
    if z_dim == w_dim:
        res = '-zw{}'.format(z_dim)
    else:
        res = '-z{}-w{}'.format(z_dim, w_dim)

    if color_w_channels == 0:
        return res
    return res + '-wrgb{}'.format(color_w_channels)


def setup_training_loop_kwargs(
        gpus: int,
        snap: int,   # Snapshot interval
        image_snap: Optional[int], # Overrides snap for images if not None
        seed: int,   # Random seed

        # Metrics.
        metrics, # List of metric names: [], ['fid50k_full'] (default), ...
        num_fid_items: int,  # Only applies to forger training
        num_forgermetric_styles: int,  # Only applies to forger training

        # Style dataset
        data: str, # Style dataset (required): <path>
        cond: bool, # Train conditional model based on dataset labels: <bool>, default = False
        subset: Optional[int], # Train with only N images: <int>, default = all
        mirror: bool,  # Augment dataset with x-flips: <bool>, default = False

        # Geom dataset
        geom_data: str, # Geometry dataset: <path>
        geom_subset: Optional[int],
        geom_input_channel: int,
        geom_truth_channel: int,
        geom_metric_data: Optional[str],  # If set, is used for FID computation

        # Base config
        cfg: str, # Base config: 'auto' (default), 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'
        gamma: Optional[float], # Override R1 gamma: <float>
        kimg: Optional[int], # Override training duration: <int>
        batch: Optional[int], # Override batch size: <int>
        nmap_layers: Optional[int],  # Override number of Z->W mapping layers

        # Geometry encoder
        enc_checkpt: str, # The path to the .pth file of the autoencoder to use
        geom_inject_resolutions: list,  # Resolutions of geo encoding to inject (see BaseGeoEncoder.feature_channels)

        # Discriminator augmentation.
        aug: str, # Augmentation mode: 'ada' (default), 'noaug', 'fixed'
        p: Optional[float], # Specify p for 'fixed' (required): <float>
        target: Optional[float], # Override ADA target for 'ada': <float>, default = depends on aug
        # Augmentation pipeline: 'blit', 'geom', 'color', 'filter', 'noise', 'cutout', 'bg', 'bgc' (default), ..., 'bgcfnc'
        augpipe: Optional[str],

        # Transfer learning.
        resume: Optional[str], # Load previous network: 'noresume' (default), 'ffhq256', 'ffhq512', 'ffhq1024', 'celebahq256', 'lsundog256', <file>, <url>
        freezed: Optional[int], # Freeze-D: <int>, default = 0 discriminator layers

        # Performance options (not included in desc).
        fp32: bool, # Disable mixed-precision training: <bool>, default = False
        nhwc: bool, # Use NHWC memory format with FP16: <bool>, default = False
        nobench: bool, # Disable cuDNN benchmarking: <bool>, default = False
        workers: Optional[int], # Override number of DataLoader workers: <int>, default = 3

        # Generator
        output_resolution: int,  # Output resolution if different from style set
        z_dim: int, # The dimension of the z space
        w_dim: int, # The dimension of the w space
        color_w_channels: int, # Number of channels to use for color estimation (0 for all)
        channel_max: int, # The maximum number of channels per layer in the StyleGAN

        color_format: str, # The ToRGB alyer to use
        synthesis_arch: str, # The architecture of the SynthesisNetwork in StyleGAN
        g_lrate: float, # Learning rate of the StyleGAN generator
        enable_geom_linear_layer: bool, # Enable linear layer after geometry injection
        freeze_geom_linear_layer_in_nongeom_phases: bool,
        positional_encoding: str,  # Type of positional encoding
        posenc_inject_resolutions,
        posenc_featuremap_mode: str,  # fixed, varying
        posenc_injection_mode: str,  # add, concat

        # Discriminator
        d_lrate: float, # Learning rate of the StyleGAN discriminator
        d_arch: str, # Architecture fo discrminator, choices are ['orig', 'skip', 'resnet']
        patch_D: bool,# The flag set to use patch discriminator. To be used.
        geom_mode_D: str, # The mode of geometry injection when updating the discriminator
        geom_mode_G: str, #

        # Custom training options
        geom_phase_mode,    # Which layers to train during geometry phase
        geom_phase_losses,  # Format string interpretable by forger.train.ForgerLosses
        main_phase_losses,  # Format string interpretable by forger.train.ForgerLosses
        geom_interval,   # The interval of updating G based on geometry loss
        partial_loss_with_triband_input,  # Whether to ignore gray input

        # Stitching
        stitch_interval,
        stitch_phase_losses,

        # Warmstarting
        geom_warmstart_losses,  # Format string interpretable by forger.train.ForgerLosses
        geom_warmstart_mode,    # Which layers to warmstart
        geom_warmstart_kimg,  # Number of iterations to run just the geometry phase
        geom_warmstart_start_kimg,
        exit_after_warmstart,

        geom_lrate        = None, # The learning rate for the geometry phase

        # wandb
        wandb_project   = None, # THe name of the wandb project
        wandb_group     = None, # The name of the wandb experiment group
        **extra_kwargs
):
    args = dnnlib.EasyDict()

    # ------------------------------------------
    # General options: gpus, snap, seed, metrics
    # ------------------------------------------

    if not (gpus >= 1 and gpus & (gpus - 1) == 0):
        raise UserError('--gpus must be a power of two')
    args.num_gpus = gpus

    if snap < 1:
        # TODO: what about no snap?
        raise UserError('--snap must be at least 1')
    args.image_snapshot_ticks = image_snap if (image_snap is not None) else snap
    if args.image_snapshot_ticks <= 0:
        args.image_snapshot_ticks = None
    args.network_snapshot_ticks = snap

    args.random_seed = seed

    valid_metrics = ['fid', 'forger']
    if metrics is None:
        metrics = valid_metrics

    assert len(set(metrics).difference(set(valid_metrics))) == 0, f'Metrics must be in: {valid_metrics}'
    args.metrics = metrics
    args.num_fid_items = num_fid_items
    args.num_forgermetric_styles = num_forgermetric_styles

    # -----------------------------------
    # Dataset: data, cond, subset, mirror
    # -----------------------------------
    args.style_set_kwargs = get_train_set_kwargs(data, cond, subset, mirror, seed)
    args.geom_set_kwargs = get_train_set_kwargs(geom_data, False, geom_subset, mirror, seed)
    if geom_metric_data is not None:
        args.geom_metric_set_kwargs = get_train_set_kwargs(geom_metric_data, False, geom_subset, mirror, seed)
    args.geom_input_channel = geom_input_channel
    args.geom_truth_channel = geom_truth_channel
    args.data_loader_kwargs = dnnlib.EasyDict(
        pin_memory=True, num_workers=3 if workers is None else workers,
        prefetch_factor=2, persistent_workers=True)
    desc_style = args.style_set_kwargs.name

    if cond:
        desc_style += '-cond'

    if subset is not None:
        desc_style += f'-subset{subset}'

    if geom_subset is not None:
        desc_style += f'-gsubset{geom_subset}'

    # ------------------------------------
    # Color Format
    # ------------------------------------
    if color_format == 'orig':
        raise RuntimeError('Must call train_orig.py for orig color format')
    args.color_format = color_format
    desc_style += f'-{color_format}'

    # ------------------------------------
    # Base config: cfg, gamma, kimg, batch
    # ------------------------------------
    desc_style += f'-{cfg}'

    cfg_specs = {
        'auto':      dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=8, fmaps=0.5, lrate=-1, gamma=1, ema=10, ramp=0.05, map=4),
        'auto-orig': dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # Populated dynamically based on resolution and GPU count.
        'stylegan2': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # Uses mixed-precision, unlike the original StyleGAN2.
        'paper256':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':  dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024': dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':     dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=1,   lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
    }

    assert cfg in cfg_specs
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto-orig':
        assert gpus >= 1, 'Must set GPU number in auto mode'
        spec.ref_gpus = gpus
        res = args.style_set_kwargs.resolution
        spec.mb = max(min(gpus * min(4096 // res, 32), 64), gpus) # keep gpu memory consumption at bay
        spec.mbstd = min(spec.mb // gpus, 4) # other hyperparams behave more predictably if mbstd group size remains fixed
        spec.fmaps = 1 if res >= 512 else 0.5
        spec.lrate = 0.002 if res >= 1024 else 0.0025
        spec.gamma = 0.0002 * (res ** 2) / spec.mb # heuristic formula
        spec.ema = spec.mb * 10 / 32
    elif cfg == 'auto':
        assert gpus >= 1, 'Must set GPU number in auto mode'
        assert batch is not None, 'Must set batch in auto mode'
        spec.ref_gpus = gpus
        spec.mb = batch

    if gpus > 0:
        desc_style += f'{gpus:d}'
        spec.ref_gpus = gpus

    if gamma is not None:
        assert isinstance(gamma, float) and gamma >= 0, '--gamma must be non-negative'
        desc_style += f'-gamma{gamma:g}'
        spec.gamma = gamma

    if kimg is not None:
        assert isinstance(kimg, int) and kimg >= 1, '--kimg must be at least 1'
        spec.kimg = kimg

    if batch is not None:
        assert isinstance(batch, int)
        if not (batch >= 1 and batch % gpus == 0):
            raise UserError('--batch must be at least 1 and divisible by --gpus')
        spec.mb = batch
        # desc_style += f'-batch{batch}'

    if nmap_layers is not None:
        desc_style += f'map{nmap_layers:d}'
        spec.map = nmap_layers

    args.total_kimg = spec.kimg
    args.batch_size = spec.mb
    args.batch_gpu = spec.mb // spec.ref_gpus
    args.ema_kimg = spec.ema
    args.ema_rampup = spec.ramp

    # ------------------------------------
    # Networks
    # ------------------------------------
    args.geom_encoder_checkpt = enc_checkpt
    args.G_kwargs = dnnlib.EasyDict(class_name='training.networks_modified.Generator',
                                    z_dim=z_dim, w_dim=w_dim,
                                    mapping_kwargs=dnnlib.EasyDict(),
                                    synthesis_kwargs=dnnlib.EasyDict())
    args.D_kwargs = dnnlib.EasyDict(class_name='training.networks.Discriminator',
                                    block_kwargs=dnnlib.EasyDict(),
                                    channel_max=channel_max,
                                    mapping_kwargs=dnnlib.EasyDict(),
                                    epilogue_kwargs=dnnlib.EasyDict())

    args.output_resolution = output_resolution
    args.G_kwargs.synthesis_kwargs.channel_base = args.D_kwargs.channel_base = int(spec.fmaps * 32768)
    args.G_kwargs.synthesis_kwargs.channel_max = args.D_kwargs.channel_max = channel_max
    args.G_kwargs.z_dim = z_dim
    args.G_kwargs.w_dim = w_dim
    args.G_kwargs.mapping_kwargs.num_layers = spec.map
    args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 4 # enable mixed-precision training
    args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow
    args.G_kwargs.synthesis_kwargs.architecture = synthesis_arch
    args.G_kwargs.synthesis_kwargs.color_format = color_format
    args.G_kwargs.synthesis_kwargs.color_w_channels = color_w_channels
    args.G_kwargs.synthesis_kwargs.enable_geom_linear = enable_geom_linear_layer
    args.G_kwargs.positional_kwargs = dnnlib.EasyDict()
    args.G_kwargs.positional_kwargs.positional_encoding = positional_encoding
    args.G_kwargs.positional_kwargs.posenc_inject_resolutions = [] if positional_encoding is None else posenc_inject_resolutions
    args.G_kwargs.positional_kwargs.posenc_featuremap_mode = posenc_featuremap_mode
    args.G_kwargs.positional_kwargs.posenc_injection_mode = posenc_injection_mode

    args.geom_inject_resolutions = geom_inject_resolutions
    args.freeze_geom_linear_layer_in_nongeom_phases = freeze_geom_linear_layer_in_nongeom_phases

    args.D_kwargs.epilogue_kwargs.mbstd_group_size = spec.mbstd
    args.D_kwargs.architecture = d_arch

    args.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam',
                                        lr=g_lrate, betas=[0, 0.99], eps=1e-8)
    args.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam',
                                        lr=d_lrate, betas=[0, 0.99], eps=1e-8)
    desc_style += lr_description(g_lrate, d_lrate)
    desc_style += zw_description(z_dim, w_dim, color_w_channels)

    # ------------------------------------
    # Losses
    # ------------------------------------
    args.loss_kwargs = dnnlib.EasyDict(class_name='training.loss_modified.ForgerLoss', r1_gamma=spec.gamma)
    args.loss_kwargs.geom_mode_D = geom_mode_D
    args.loss_kwargs.geom_mode_G = geom_mode_G
    args.loss_kwargs.color_format = color_format
    args.loss_kwargs.stitch_phase_losses = stitch_phase_losses
    args.loss_kwargs.geom_phase_losses = geom_phase_losses
    args.loss_kwargs.geom_warmstart_losses = geom_warmstart_losses
    args.loss_kwargs.main_phase_losses = main_phase_losses
    args.loss_kwargs.partial_loss_with_triband_input = partial_loss_with_triband_input
    args.geom_interval = geom_interval
    args.stitch_interval = stitch_interval
    args.geom_phase_mode = geom_phase_mode.split(',')
    args.geom_warmstart_mode = geom_warmstart_mode.split(',')
    args.geom_lr = g_lrate if geom_lrate is None else geom_lrate
    args.geom_warmstart_kimg = geom_warmstart_kimg
    args.geom_warmstart_start_kimg = geom_warmstart_start_kimg
    args.exit_after_warmstart = exit_after_warmstart
    if geom_interval > 0:
        assert len(geom_phase_losses) > 0, 'Must specify --geom_phase_losses to enable geom interval'
    if args.geom_warmstart_kimg > 0:
        assert (geom_warmstart_losses is not None and len(geom_warmstart_losses) > 0) or len(geom_phase_losses) > 0, \
            'Must provide geom_phase or geom_warmstart losses to enable geom warmstart'
    else:
        args.loss_kwargs.geom_warmstart_losses = None

    if cfg == 'cifar':
        args.loss_kwargs.pl_weight = 0  # disable path length regularization
        args.loss_kwargs.style_mixing_prob = 0  # disable style mixing
        args.D_kwargs.architecture = 'orig'  # disable residual skip connections

    # ---------------------------------------------------
    # Discriminator augmentation: aug, p, target, augpipe
    # ---------------------------------------------------

    args.ada_target = None
    if aug == 'ada':
        args.ada_target = 0.6
    elif aug == 'noaug':
        pass
    elif aug == 'fixed':
        if p is None:
            raise UserError(f'--aug={aug} requires specifying --p')
    else:
        raise UserError(f'--aug={aug} not supported')

    if aug != 'ada':
        # Only add description if using unusual setting
        desc_style += f'-{aug}'

    if p is not None:
        if aug != 'fixed':
            raise UserError('--p can only be specified with --aug=fixed')
        if not 0 <= p <= 1:
            raise UserError('--p must be between 0 and 1')
        desc_style += f'-p{p:g}'
        args.augment_p = p

    if target is not None:
        if aug != 'ada':
            raise UserError('--target can only be specified with --aug=ada')
        if not 0 <= target <= 1:
            raise UserError('--target must be between 0 and 1')
        desc_style += f'-target{target:g}'
        args.ada_target = target

    assert augpipe is None or isinstance(augpipe, str)
    if augpipe is None:
        augpipe = 'bgc'
    else:
        if aug == 'noaug':
            raise UserError('--augpipe cannot be specified with --aug=noaug')

    if augpipe != 'bgc':
        desc_style += f'-{augpipe}'

    augpipe_specs = {
        'blit':   dict(xflip=1, rotate90=1, xint=1),
        'geom':   dict(scale=1, rotate=1, aniso=1, xfrac=1),
        'color':  dict(brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'filter': dict(imgfilter=1),
        'noise':  dict(noise=1),
        'cutout': dict(cutout=1),
        'bg':     dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1),
        'bgc':    dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1),
        'bgcf':   dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1),
        'bgcfn':  dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1),
        'bgcfnc': dict(xflip=1, rotate90=1, xint=1, scale=1, rotate=1, aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1, imgfilter=1, noise=1, cutout=1),
    }
    assert augpipe in augpipe_specs

    if aug != 'noaug':
        args.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', **augpipe_specs[augpipe])

    # ----------------------------------
    # Transfer learning: resume, freezed
    # ----------------------------------

    resume_specs = {
        'ffhq256':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl',
        'ffhq512':     'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res512-mirror-stylegan2-noaug.pkl',
        'ffhq1024':    'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res1024-mirror-stylegan2-noaug.pkl',
        'celebahq256': 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/celebahq-res256-mirror-paper256-kimg100000-ada-target0.5.pkl',
        'lsundog256':  'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/lsundog-res256-paper256-kimg100000-noaug.pkl',
    }

    assert resume is None or isinstance(resume, str)
    if resume is None:
        args.resume_pkl = None
    elif resume in resume_specs:
        desc_style += f'-resume{resume}'
        args.resume_pkl = resume_specs[resume] # predefined url
    else:
        desc_style += '-resumecustom'
        args.resume_pkl = resume # custom path or url

    if resume is not None:
        args.ada_kimg = 100 # make ADA react faster at the beginning
        args.ema_rampup = None # disable EMA rampup

    if freezed is not None:
        assert isinstance(freezed, int)
        if not freezed >= 0:
            raise UserError('--freezed must be non-negative')
        if freezed > 0:
            desc_style += f'-freezed{freezed:d}'
        args.D_kwargs.block_kwargs.freeze_layers = freezed

    # -------------------------------------------------
    # Performance options: fp32, nhwc, nobench, workers
    # -------------------------------------------------

    if fp32:
        args.G_kwargs.synthesis_kwargs.num_fp16_res = args.D_kwargs.num_fp16_res = 0
        args.G_kwargs.synthesis_kwargs.conv_clamp = args.D_kwargs.conv_clamp = None

    if nhwc:
        args.G_kwargs.synthesis_kwargs.fp16_channels_last = args.D_kwargs.block_kwargs.fp16_channels_last = True


    args.cudnn_bench = not nobench

    args.wandb_project = wandb_project
    args.wandb_group = wandb_group

    return desc_style, args

# ----------------------------------------------------------------------------

def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(file_name=os.path.join(args.run_dir, 'log.txt'), file_mode='a', should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, '.torch_distributed_init'))
        if os.name == 'nt':
            init_method = 'file:///' + init_file.replace('\\', '/')
            torch.distributed.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=args.num_gpus)
        else:
            init_method = f'file://{init_file}'
            torch.distributed.init_process_group(backend='nccl', init_method=init_method, rank=rank, world_size=args.num_gpus)

    # Init torch_utils.
    sync_device = torch.device('cuda', rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0:
        custom_ops.verbosity = 'none'

    # Execute training loop.
    color_format = args.color_format
    delattr(args, 'color_format')

    logger.info("Running with color format: ", color_format)
    training_loop_modified.training_loop(rank=rank, **args)


# ----------------------------------------------------------------------------


class CommaSeparatedList:
    name = 'list'

    @staticmethod
    def convert(value):
        if value is None or value.lower() == 'none' or value == '':
            return []
        return value.split(',')

class IntCommaSeparatedList:
    name = 'int_list'

    @staticmethod
    def convert(value):
        if value is None or value.lower() == 'none' or value == '':
            return []
        return [int(x) for x in value.split(',') if len(x) > 0]

# ----------------------------------------------------------------------------

def main(**config_kwargs):
    """
    The codebase is based on StyleGAN2. For training the original StyleGAN2, please refer to
    the docstring in https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/train.py

    Currently, Art Forger can only be trained on 1 GPU.
    Caution:

    Below is an example shell script to train the Art Forger model:

    \b
    # Train with custom dataset using 1 GPU.

    python3 -m thirdparty.stylegan2_ada_pytorch.train --gpus 1 --batch 8 \
    --outdir /home/user/Documents/experiment/art-forger/forger_inplace/viz_grid \
    --data /home/user/Documents/data/73467/proc0_256_aug0.zip \
    --kimg 1500 --color_triads 1 \
    --geom_inject \
    --geom_loss_types mask_bce,mask_iou \
    --geom_loss_weights 0.001,0.002 \
    --synthesis_arch orig \
    --geom_mode_D "enc" \
    --channel_max 256 \
    --geom_data="/home/aliceli/workspace/shared/Data/Splines/spline_patches_1K.zip" \
    --enc_checkpt="/home/aliceli/workspace/shared/Experiments/cy/spline_encoder/ngc/conv_l3_emb8_cf_2_neg0.01/checkpts/conv_run1_it28000_e11.pt" \
    --wandb_group "debug"
    """
    # TODO: Modify the argument parsing logic such that user can only set parameters for the chosen color_format
    dnnlib.util.Logger(should_flush=True)
    pser = argparse.ArgumentParser()
    pser.add_argument('--outdir', help='Where to save the results', type=str, required=True)
    pser.add_argument('--name_prefix', help='Custom name for the run', type=str, default=None)

    pser.add_argument('--gpus', help='Number of GPUs to use [default: 1]', type=int, default=1)
    pser.add_argument('--snap', help='Snapshot interval.', type=int, default=50)
    pser.add_argument('--image_snap', help='Overrides snap for images', type=int, default=None)
    pser.add_argument('--seed', help='Random seed [default: 0]', type=int, default=0)
    pser.add_argument('-n', '--dry-run', help='Print training options and exit', action='store_true')

    # Metrics
    pser.add_argument('--metrics', help='Comma-separated list or "none", supports "fid" and "forger".',
                      type=CommaSeparatedList.convert)
    pser.add_argument('--num_fid_items', type=int, default=50000,
                      help='Number of items to use for FID stats computation. Only applies to forger training. ')
    pser.add_argument('--num_forgermetric_styles', type=int, default=200,
                      help='Number of styles to use for computing forger metrics.')

    # Style dataset
    pser.add_argument('--data', help='Training style data (directory or zip)', type=str, required=True)
    pser.add_argument('--cond', help='Train conditional model based on dataset labels.', action='store_true')
    pser.add_argument('--subset', help='Train with only N images [default: all]', type=int, default=None)
    pser.add_argument('--mirror', help='Enable dataset x-flips [default: false]', action='store_true')

    # Geometry dataset
    pser.add_argument('--geom_data', help='Training geometry data (directory or zip)', type=str, required=True)
    pser.add_argument('--geom_subset', help='Train with only N images [default: all]', type=int, default=None)
    pser.add_argument('--geom_input_channel', help='Channel to use for geometry conditioning.', type=int, default=1)
    pser.add_argument('--geom_truth_channel', help='Channel to use for geometry loss.', type=int, default=2)
    pser.add_argument('--geom_metric_data', help='Geometry data (dir or zip) for evaluation; not used for training.',
                      type=str, default=None)

    # Base config
    pser.add_argument('--cfg', help='Base config [default: auto]', default='auto', const='auto',
                      choices=('auto', 'stylegan2', 'paper256', 'paper512', 'paper1024', 'cifar'), nargs='?')
    pser.add_argument('--gamma', help='Override R1 gamma', type=float, default=None)
    pser.add_argument('--kimg', help='Override training duration', type=int, default=None)
    pser.add_argument('--batch', help='Override batch size', type=int, default=None)
    pser.add_argument('--nmap_layers', help='Number of Z to W mapping layers', type=int, default=None)

    # Geometry encoder
    pser.add_argument('--enc_checkpt', help='The path of the pretrained geometry autoencoder',
                      type=str, required=True)
    pser.add_argument('--geom_inject_resolutions', help='Resolutions to inject geometry at, CSV list '
                      'defined as follows: 0 - encoder bottleneck output, 1 - one layer up in the decoder, 2 - two up, etc',
                      type=IntCommaSeparatedList.convert, default=[0])

    # Augmentation
    pser.add_argument('--aug', help='Augmentation mode [default: ada]', default='ada', const='ada',
                      choices=('noaug', 'ada', 'fixed'), nargs='?')
    pser.add_argument('--p', help='Augmentation probability for --aug=fixed', type=float, default=None)
    pser.add_argument('--target', help='ADA target value for --aug=ada', type=float)
    pser.add_argument('--augpipe', type=str, default=None,
                      help='Augmentation pipeline, one of: blit, geom, color, filter, noise, cutout, bg, bgc, bgcf, bgcfn, bgcfnc')

    # Freezing/resuming
    pser.add_argument('--resume', help='Resume training [default: noresume]', type=str, default=None)
    pser.add_argument('--freezed', help='Freeze-D [default: 0 layers]', type=int, default=None)

    # Performance options
    pser.add_argument('--fp32', help='Disable mixed-precision training', action='store_true')
    pser.add_argument('--nhwc', help='Use NHWC memory format with FP16', action='store_true')
    pser.add_argument('--nobench', help='Disable cuDNN benchmarking', action='store_true')
    pser.add_argument('--workers', help='Override number of DataLoader workers', type=int, default=None)

    # Generator
    pser.add_argument('--output_resolution', type=int, default=None,
                      help='Must be smaller than style and geom size; if set, will subsample patches during training.')
    pser.add_argument('--z_dim', type=int, default=512)
    pser.add_argument('--w_dim', type=int, default=512)
    pser.add_argument('--color_w_channels', type=int, default=0,
                      help='Set to >0 to use only this many channels of W for color estimation.')
    pser.add_argument('--channel_max', type=int, default=512)
    pser.add_argument('--color_format', help='Run a network with the chosen ToRGB Layer',
                      type=str, default="triad", choices=("triad", "canvas"))
    pser.add_argument('--synthesis_arch', help='Type of generator architecture', default='orig', const='orig',
                      choices=('orig', 'skip', 'resnet'), nargs='?')
    pser.add_argument('--g_lrate', help='Generator learning rate', type=float, default=0.0001)
    pser.add_argument('--enable_geom_linear_layer', help='Adds linear layer for injected geometry.', action='store_true')
    pser.add_argument('--freeze_geom_linear_layer_in_nongeom_phases', action='store_true')

    # Positional encoding
    pser.add_argument('--positional_encoding', type=str, default=None,
                      help='Use positional encoding in the generator, supported: grid, sine:16, sine:20, etc.')
    pser.add_argument('--posenc_inject_resolutions', help='Resolutions to inject positional encoding at, CSV list '
                      'defined as the generator level, 0 - const, 1 - 8x8, etc.',
                      type=IntCommaSeparatedList.convert, default=[0])
    pser.add_argument('--posenc_featuremap_mode', help='How to generate positional encoding for the feature map, fixed for all '
                      'positions or varying as grid', type=str, default='fixed', choices=('fixed', 'varying'))
    pser.add_argument('--posenc_injection_mode', help='How to inject positional encoding, by adding or concat; '
                      'if add note that number of channels must be correct or code will crash.',
                      type=str, default='cat', choices=('add', 'cat'))

    # Discriminator
    pser.add_argument('--d_lrate', help='Discriminator learning rate', type=float, default=0.0001)
    pser.add_argument('--d_arch', help='Architecture of Discriminator', type=str,
                      default='resnet', choices=('orig', 'resnet', 'skip'))
    pser.add_argument('--patch_D', help='Whether or not to use patch GAN', action='store_true')
    pser.add_argument('--geom_mode_D', type=str, default="orig", choices=("orig", "zero", "rand", "enc"),
                      help='If set to "orig", a real geometry feature will be used to update the Discriminator. '
                           'If set to "zero", a zero tensor will be used.'
                           'If set to "rand", a random tensor will be used.'
                           'If set to "enc", get the geometry feature by encoding the real image',
                      )
    pser.add_argument('--geom_mode_G', type=str, default='orig',  choices=("orig", "zero", "rand", "enc"),
                      help='If set to "orig", a real geometry feature will be used to update the Discriminator.'
                           'See --geom_mode_D for details of choices ')

    # Geometry loss
    pser.add_argument('--geom_phase_losses', type=str, default='',
                      help='Human readable sum of weighted losses formatted as: ' +
                      'loss_component+loss_component where: ' +
                           forger.train.losses.ForgerLossItemFactory.get_format_info_string())
    pser.add_argument('--geom_phase_mode', type=str, default='all',
                      help='CSV layers to update during geom phase, can be all, rgb, last_and_rgb, linear')
    pser.add_argument('--main_phase_losses', type=str, default='',
                      help='Human readable sum of weighted losses formatted as: ' +
                           'loss_component+loss_component where: ' +
                           forger.train.losses.ForgerLossItemFactory.get_format_info_string())
    pser.add_argument('--geom_interval', type=int, default=-1,
                      help='The interval of updating Generator based on geometry loss')
    pser.add_argument('--geom_lrate', type=float, default=None,
                      help='The learning rate of G optimizer duing geometry phase. If None, then yhe same as G lr')
    pser.add_argument('--partial_loss_with_triband_input', action='store_true',
                      help='If set, will only compute loss for white and black pixels in the truth; ignore gray.')

    # Stitching
    pser.add_argument('--stitch_interval', type=int, default=-1,
                      help='The interval of updating Generator with stitching opration in the loop.')
    pser.add_argument('--stitch_phase_losses', type=str, default='',
                      help='Human readable sum of weighted losses formatted as: ' +
                           'loss_component+loss_component where: ' +
                           forger.train.losses.ForgerLossItemFactory.get_format_info_string())

    # Geometry Warmstart
    # pser.add_argument('--geom_freeze_linear_after_warmstart', action='store_true')
    # pser.add_argument('--geom_warmstart_linear_only', action='store_true')
    pser.add_argument('--geom_warmstart_mode', type=str, default='all',
                      help='CSV list of layers to warmstart, can be all, rgb, last_and_rgb, linear')
    pser.add_argument('--geom_warmstart_losses', type=str, default=None,
                      help='If not set will use geom_phase_losses. Human readable sum of weighted losses formatted as: ' +
                      'loss_component+loss_component where: ' +
                           forger.train.losses.ForgerLossItemFactory.get_format_info_string())
    pser.add_argument('--geom_warmstart_kimg', type=int, default=0,
                      help='Will only run geom_phase_losses for this many iterations to pre-condition network.')
    pser.add_argument('--geom_warmstart_start_kimg', type=int, default=0,
                      help='When to start warmstarting geometry.')
    pser.add_argument('--exit_after_warmstart', action='store_true')
    #pser.add_argument('--geometry_finetune_iterations', type=int, default=0


    # Observability
    pser.add_argument('--wandb_project', help='The name of the wandb project', type=str, default=None)
    pser.add_argument('--wandb_group', help='The name of the wandb experiment group', type=str, default='debug')
    pser.add_argument('--cache_dir', help='Directory for FID stats and other.', type=str, default=None)
    forger.util.logging.add_log_level_flag(pser)
    argparse_args = pser.parse_args()
    forger.util.logging.default_log_setup(argparse_args.log_level)

    dnnlib.util.set_cache_dir(argparse_args.cache_dir)
    outdir = argparse_args.outdir
    dry_run = argparse_args.dry_run
    # Delete these two variables as they are not needed outside this function
    delattr(argparse_args, 'outdir')
    delattr(argparse_args, 'dry_run')

    # Setup training options.
    # TODO: this needs a clean up following switch to ArgumentParser with default values
    run_desc, args = setup_training_loop_kwargs(**argparse_args.__dict__)

    if argparse_args.name_prefix is not None:
        run_desc = argparse_args.name_prefix + '-' + run_desc

    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{run_desc}')
    assert not os.path.exists(args.run_dir)

    # Print options.
    print()
    print('Training options:')
    print(json.dumps(args, indent=2))
    print()
    print(f'Output directory:   {args.run_dir}')
    print(f'Training style data:      {args.style_set_kwargs.path}')
    print(f'Training geometry data:   {args.geom_set_kwargs.path}')
    print(f'Training duration:  {args.total_kimg} kimg')
    print(f'Number of GPUs:     {args.num_gpus}')
    #print(f'Number of images:   {args.style_set_kwargs.max_size}')
    #print(f'Image resolution:   {args.style_set_kwargs.resolution}')
    #print(f'Conditional model:  {args.style_set_kwargs.use_labels}')
    #print(f'Dataset x-flips:    {args.style_set_kwargs.xflip}')
    print()

    # Dry run?
    if dry_run:
        print('Dry run; exiting.')
        return

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(args.run_dir)
    with open(os.path.join(args.run_dir, 'training_options.json'), 'wt') as f:
        json.dump(args, f, indent=2)

    # Launch processes.
    args.argparse_args = argparse_args  # Used in wandb.init later
    print('Launching processes...')
    torch.multiprocessing.set_start_method('spawn')
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
