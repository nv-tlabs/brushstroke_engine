# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from skimage.io import imread
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot colors of an image.')
    parser.add_argument(
        '--image', action='store', type=str, required=True,
        help='Image path.')
    parser.add_argument(
        '--samples', action='store', type=int, default=5000,
        help='Number of samples to plot.')
    args = parser.parse_args()

    img = imread(args.image)
    img = img.reshape((-1, img.shape[2] if len(img.shape) > 2 else 1))
    samples = np.random.choice(img.shape[0], (args.samples,))
    unique_samples, counts = np.unique(samples, axis=0, return_counts=True)

    min_marker_size = 5.0
    colors = img[unique_samples, :].astype(np.float32) / 255.0
    sizes = counts.astype(np.float32) * min_marker_size

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(colors[:, 0], colors[:, 1], colors[:, 2],
               c=colors, cmap=None, edgecolors='none', depthshade=0, s=sizes)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_zlim(0, 1.0)
    ax.set_xlabel('R')
    ax.set_ylabel('G')
    ax.set_zlabel('B')
    plt.show()