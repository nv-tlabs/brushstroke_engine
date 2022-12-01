# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import bisect
import logging
import math
import numpy as np
from skimage.draw import line
import random

logger = logging.getLogger(__name__)


class CatmullRomSpline:
    """Rough implementation of multi-node Centripetal Catmull-Rom splines."""
    # TODO: needs to be cleaned up
    # TODO: matrix formulation would be more efficient
    # TODO: need rough arc length parameterization or way to cut up spline into windows

    def __init__(self, ctr_pts, alpha=0.5):
        """

        :param ctr_pts: N x 2 tensor of control points, at least 4
        """
        assert ctr_pts.shape[1] == 2
        assert ctr_pts.shape[0] >= 4
        self.pts = ctr_pts
        self.alpha = alpha
        self.ts = np.concatenate(
            [np.zeros((1), dtype=np.float32),
             np.power(np.linalg.norm(self.pts[1:, :] - self.pts[:-1, :], axis=1),
                      self.alpha)], axis=0)
        self.ts = list(np.cumsum(self.ts))

    def sample_t(self, t_global):
        # t_global 0...1
        idx = bisect.bisect_left(self.ts, t_global) - 2
        if idx > self.pts.shape[0] - 4:
            logger.warning('idx {} out of bounds for target {}'.format(idx, t_global))
            idx = self.pts.shape[0] - 4
        #t_global = t_global * (self.pts.shape[0] - 3)
        #idx = self.pts.shape[0] - math.floor(t_global)
        #t_local = t_global - idx
        return self.sample_t_one(t_global, idx)

    def sample_t_one(self, t, idx):
        """
        Samples itself based on time.
        :param t:
        :return:
        """
        #t0 = 0
        #t1 = math.pow(np.linalg.norm(self.pts[idx + 1, :] - self.pts[idx, :]), self.alpha)
        #t2 = t1 + math.pow(np.linalg.norm(self.pts[idx + 2, :] - self.pts[idx + 1, :]), self.alpha)
        #t3 = t2 + math.pow(np.linalg.norm(self.pts[idx + 3, :] - self.pts[idx + 2, :]), self.alpha)
        t0 = self.ts[idx]
        t1 = self.ts[idx + 1]
        t2 = self.ts[idx + 2]
        t3 = self.ts[idx + 3]

        #t_range = t2 - t1
        #t = (t_local * t_range + t1)
        logger.debug('T 1, 2, 3 = {}, {}, {} --> {}'.format(t1, t2, t3, t))

        A1 = (t1 - t) / (t1 - t0) * self.pts[idx, :] + (t - t0) / (t1 - t0) * self.pts[idx + 1, :]
        A2 = (t2 - t) / (t2 - t1) * self.pts[idx + 1, :] + (t - t1) / (t2 - t1) * self.pts[idx + 2, :]
        A3 = (t3 - t) / (t3 - t2) * self.pts[idx + 2, :] + (t - t2) / (t3 - t2) * self.pts[idx + 3, :]
        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
        C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2


       # p(t) = ...
       # t(s) = ..?..
       # ds/dt


        #tvec = np.array([t ** 3, t ** 2, t, 1], dtype=self.pts.dtype).reshape((1, 4)) * self.tension  # Fix
        #res = np.matmul(np.matmul(tvec, self.mat), self.pts)
        res = C
        return res



    def sample_s(self, nsamples):
        """
        Uniform arc length sampling of the entire spline.
        :param nsamples:
        :return:
        """
        pass


def sample_control_pts2(npts):
    quadrants = np.zeros((4, 4), dtype=np.int)
    res = np.zeros((npts, 2), dtype=np.float32)

    def _sample_next():
        indices = np.argwhere(quadrants == 0)
        if indices.shape[0] == 0:
            return np.random.rand(1, 2) * 2.2 - 1.1
        else:
            idx = indices[random.randint(0, indices.shape[0] - 1), :]  # 0...3
            x = idx[0] / 4 * 2 - 1 + random.random() * 0.5
            y = idx[1] / 4 * 2 - 1 + random.random() * 0.5
            quadrants[idx[1], idx[0]] += 1
            return [x, y]

    for i in range(0, npts):
        res[i, :] = _sample_next()

    return res


def sample_control_pts(npts, radius_mean=0.8, radius_sigma=0.3):
    res = np.zeros((npts, 2), dtype=np.float32)
    res[0, :] = np.random.rand(1, 2) * 2.0 - 1.0

    def _sample_next(prev):
        radius = np.random.normal(loc=radius_mean, scale=radius_sigma)
        theta = np.random.random(1)[0] * 2 * math.pi
        delta = [math.cos(theta) * radius, math.sin(theta) * radius]

        return [min(1.0, max(-1.0, prev[0])) + delta[0],
                min(1.0, max(-1.0, prev[0])) + delta[1]]

    for i in range(1, npts):
        res[i, :] = _sample_next(res[i - 1, :])

    return res


def normalize_coord(x, width, clamp=True):
    """ Normalizes coord in [-1, 1] to be in [0, width-1] range."""
    tmp = round((x + 1.0) / 2.0 * width)
    if not clamp:
        return tmp
    return max(0, min(width - 1, tmp))


# TODO: refactor the draw_spline functions
def draw_spline(spline, width, nsamples):
    image = np.ones((width, width, 1), dtype=np.uint8) * 255

    if isinstance(spline, CatmullRomSpline):
        return draw_spline_custom(spline, image, nsamples)
    else:
        return draw_spline_lib(spline, image, nsamples)


def draw_spline_custom(spline, image: np.ndarray, nsamples):
    width = image.shape[0]

    for i in range(0, nsamples):
        t = spline.ts[1] + i / (nsamples - 1.0) * (spline.ts[-2] - spline.ts[1])  # HACK
        res = spline.sample_t(t).squeeze()
        x = normalize_coord(res[0], width, clamp=False)
        y = normalize_coord(res[1], width, clamp=False)
        if 0 <= x < width and 0 <= y < width:
            image[y, x, :] = 0

    return image

def draw_spline_lib(spline, image: np.ndarray, nsamples):
    """
        :param spline: splines.CatmullRom - The spline object to be drawn to image
        :param image: numpy.ndarray - 3D array of shape H x W x C
        :param nsamples: int - number of sample points to use for generating the spline
        :return: numpy.ndarray - 3D array of shape H x W x C
    """
    assert nsamples is not None
    width = image.shape[0]
    prev_x = prev_y = None
    for i, t in enumerate(np.linspace(spline.grid[0], spline.grid[-1], nsamples)):
        x, y = spline.evaluate(t)
        x = int(normalize_coord(x, width, clamp=False))
        y = int(normalize_coord(y, width, clamp=False))

        if i != 0:
            rows, cols = line(prev_y, prev_x, y, x)
            for i in range(len(rows)):
                if 0 <= cols[i] < width and 0 <= rows[i] < width:
                    image[rows[i], cols[i], 0] = 0
        prev_x, prev_y = x, y
    return image


def draw_spline_debug(spline, nsamples=8, width=256):
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[:, :, :] = 255

    for i in range(0, nsamples):
        t = spline.ts[1] + i / (nsamples - 1.0) * (spline.ts[-2] - spline.ts[1])  # HACK
        res = spline.sample_t(t).squeeze()
        x = normalize_coord(res[0], width, clamp=False)
        y = normalize_coord(res[1], width, clamp=False)
        if 0 <= x < width and 0 <= y < width:
            blue = i / (nsamples - 1.0)
            red = 1 - blue
            image[y, x, 0] = red * 255
            image[y, x, 1] = 0
            image[y, x, 2] = blue * 255

    # Draw control points too
    for p in range(spline.pts.shape[0]):
        x = normalize_coord(spline.pts[p, 0], width)
        y = normalize_coord(spline.pts[p, 1], width)
        blue = p / (spline.pts.shape[0] - 1)
        red = 1 - blue
        image[y-1:y+1, x-1:x+1, 0] = 255 * red
        image[y-1:y+1, x-1:x+1, 1] = 0
        image[y-1:y+1, x-1:x+1, 2] = 255 * blue

    return image
