# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import os
import argparse
import forger.util.logging
import torch

from thirdparty.stylegan2_ada_pytorch.metrics.metric_utils import compute_feature_stats_for_dataset, MetricOptions
from thirdparty.stylegan2_ada_pytorch.metrics.frechet_inception_distance import default_feature_detector, compute_fid_from_feature_stats
from thirdparty.stylegan2_ada_pytorch.train import get_train_set_kwargs


if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='ArtForger user interface.')
    aparser.add_argument('--style_dataset', action='store', type=str,
                         default='style1:/mnt/ssd2/Data/Images/Scribbles_proc/datasets/styles1.zip')
    aparser.add_argument('--generated_images', action='store', type=str, required=True)
    aparser.add_argument('--output_file', action='store', type=str, required=True)
    aparser.add_argument('--batch_size', action='store', type=int, default=64)
    aparser.add_argument('--num_fid_items', action='store', type=int, default=50000)
    aparser.add_argument('--resolution', action='store', type=int, default=128)
    aparser.add_argument('--debug', action='store_true')
    forger.util.logging.add_log_level_flag(aparser)
    args = aparser.parse_args()

    device = torch.device('cuda')

    detector_url, detector_kwargs = default_feature_detector()
    data_loader_kwargs = dict(pin_memory=False, num_workers=1)

    style_set_kwargs = get_train_set_kwargs(args.style_dataset, False)
    style_set_kwargs.resolution = args.resolution
    style_set_kwargs.resize_mode = 'crop'
    opts_style = MetricOptions(dataset_kwargs=style_set_kwargs, num_gpus=1, rank=0, device=device, cache=True)
    mu_real, sigma_real = compute_feature_stats_for_dataset(
            opts=opts_style, data_loader_kwargs=data_loader_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=args.num_fid_items).get_mean_cov()
    del style_set_kwargs
    del opts_style

    gen_set_kwargs = get_train_set_kwargs(args.generated_images, False)
    gen_set_kwargs.resolution = args.resolution
    gen_set_kwargs.resize_mode = 'crop'
    opts_gen = MetricOptions(dataset_kwargs=gen_set_kwargs, num_gpus=1, rank=0, device=device, cache=False)
    mu_gen, sigma_gen = compute_feature_stats_for_dataset(
        opts=opts_gen, data_loader_kwargs=data_loader_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=args.num_fid_items).get_mean_cov()

    result = compute_fid_from_feature_stats(mu_real, sigma_real, mu_gen, sigma_gen)

    metric_name = 'fid'
    if args.num_fid_items < 1000:
        metric_name = '%s_toy%d' % (metric_name, args.num_fid_items)
    else:
        metric_name = '%s%dk' % (metric_name, args.num_fid_items // 1000)

    with open(args.output_file, 'a') as f:
        f.write(f'{metric_name} {result} {args.generated_images} {args.style_dataset}')

    print(f'Output {metric_name} {result} to {args.output_file}')
