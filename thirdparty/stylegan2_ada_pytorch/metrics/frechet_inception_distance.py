# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils


def default_feature_detector():
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True)  # Return raw features before the softmax layer.
    return detector_url, detector_kwargs

def compute_fid(opts, max_real, num_gen):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen).get_mean_cov()

    # TODO: what is the point of running all the computation on all nodes?
    if opts.rank != 0:
        return float('nan')

    return compute_fid_from_feature_stats(mu_real, sigma_real, mu_gen, sigma_gen)


def compute_fid_from_feature_stats(mu_real, sigma_real, mu_gen, sigma_gen):
    m = np.square(mu_gen - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

# FORGER CUSTOM ----------------------------------------------------------------------------


def forger_compute_fid(generator, style_dataset_kwargs, device, num_gpus, rank, num_items):
    detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
    detector_kwargs = dict(return_features=True)  # Return raw features before the softmax layer.

    opts = metric_utils.MetricOptions(dataset_kwargs=style_dataset_kwargs,
                                      num_gpus=num_gpus, rank=rank, device=device, cache=True)
    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=num_items).get_mean_cov()

    stats, stats_bc = metric_utils.forger_compute_feature_stats_for_generator(
        generator, device, num_gpus, rank,
        detector_url, detector_kwargs,
        batch_size=generator.batch_size * 4,  # debug_dir='/tmp/fid_sanity',
        capture_mean_cov=True, max_items=num_items)

    metric_name = 'fid'
    if num_items < 1000:
        metric_name = '%s_toy%d' % (metric_name, num_items)
    else:
        metric_name = '%s%dk' % (metric_name, num_items // 1000)

    result = {metric_name + '_full': compute_fid_from_feature_stats(mu_real, sigma_real, *stats.get_mean_cov())}

    if stats_bc is not None:
        result[metric_name + '_fadecanvas'] = compute_fid_from_feature_stats(
            mu_real, sigma_real, *stats_bc.get_mean_cov())

    return result


