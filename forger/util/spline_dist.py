# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
from abc import ABC
from bisect import bisect_left
import numpy as np
import random
from typing import List

"""
Experimenting with different distribution of thickness
for the splines
"""


def map_flag_to_distrib_class(flag: str):
    if flag == 'gauss':
        return GaussianMix
    elif flag == 'rand':
        return RandomInt
    else:
        raise RuntimeError('Unrecognized thickness distribution')


class Distribution:
    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError


class GaussianMix(Distribution, ABC):
    def __init__(self, args=None):
        super(GaussianMix, self).__init__()
        self.centers = [float(c) for c in args.centers.split(',')]
        self.sigmas = [float(s) for s in args.sigmas.split(',')]
        probabilities = [float(p) for p in args.prob.split(',')]

        if len(self.centers) != len(self.sigmas):
            raise RuntimeError("The number of centers is different from that of sigmas")
        if len(probabilities) < len(self.centers):
            raise RuntimeError("")
        elif len(probabilities) > len(self.centers):
            raise RuntimeWarning("Provided ")

        self.n_gaussian = len(self.centers)
        prob_sum = sum(probabilities[:self.n_gaussian])

        # Cumulative distribution function.
        self.cdf = [0.0] + [p / prob_sum for p in probabilities[:self.n_gaussian]]
        for i in range(2, self.n_gaussian+1):
            self.cdf[i] += self.cdf[i-1]

    @staticmethod
    def modify_cmd_arg(parser):
        parser.add_argument('--centers', type=str, default='6.0,10.0',
                            help='Comma-separated list of 1D gaussian centers')
        parser.add_argument('--sigmas', type=str, default='2.0,3.0',
                            help='Comma-separated list of 1D gaussian sigma')
        parser.add_argument('--prob', type=str, default='6.0,10.0',
                            help='Comma-separated list of float numbers; does not have to be normalized.')
        return parser

    def sample(self):
        r = random.uniform(0.0, 1.0)
        gaussian_index = bisect_left(self.cdf, r) - 1
        res = -1.0

        # Keep drawing until res can be rounded to a non-negative number
        while res < -0.5:
            res = np.random.normal(self.centers[gaussian_index], self.sigmas[gaussian_index])
        return np.round(res)


class RandomInt(Distribution):
    """
    A wrapper around
    """
    def __init__(self, args):
        super(RandomInt, self).__init__()
        self.min_thickness = args.min_thickness
        self.max_thickness = args.max_thickness

    @staticmethod
    def modify_cmd_arg(parser):
        parser.add_argument('--min_thickness', action='store', type=int, default=1)
        parser.add_argument('--max_thickness', action='store', type=int, default=30)
        return parser

    def sample(self):
        return random.randint(self.min_thickness, self.max_thickness)
