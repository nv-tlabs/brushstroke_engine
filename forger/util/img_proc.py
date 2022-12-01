# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import logging
import numpy as np
import random
import torch
import torchvision
from scipy.stats import entropy as scipy_entropy
from skimage.transform import resize
from skimage.measure import shannon_entropy
from skimage.filters import threshold_otsu, rank
from skimage.morphology import erosion, dilation, opening, closing
from skimage.morphology import disk
from skimage.transform import rescale

from forger.util.logging import log_tensor

logger = logging.getLogger(__name__)


def ensure_np(img):
    if torch.is_tensor(img):
        if len(img.shape) == 3 and img.shape[0] < img.shape[1]:
            img = img.permute(1, 2, 0)
        img = img.cpu().numpy()
    return img


def ensure_np_uint8(img):
    img = ensure_np(img)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    return img


def to_torch_gray(im):
    if len(im.shape) == 2:
        im = np.expand_dims(im, axis=2)
    return torch.from_numpy(np.mean(im.astype(np.float32), axis=2)).unsqueeze(0) / 255.0


def alpha_to_torch_gray(im):
    assert im.shape[2] == 4
    return 1 - (torch.from_numpy(im[..., 3]).to(torch.float32).unsqueeze(0) / 255.0)


def blur_img(im):
    max_dim = max(im.shape[1], im.shape[2])
    kernel_size = max(3, max_dim // 20)
    if kernel_size % 2 != 1:
        kernel_size += 1
    blur = torchvision.transforms.GaussianBlur(kernel_size, sigma=(max_dim / 100))
    return blur(im)


def torch_shannon_entropy(img):
    _, counts = torch.unique((img * 255).to(torch.uint8), return_counts=True)
    return scipy_entropy(counts.cpu().numpy(), base=2)


def threshold_img(img, to_float=True):
    thresh = threshold_otsu(ensure_np(img))
    res = img > thresh
    if to_float:
        res = res.to(torch.float32)
    return res


def threshold_img_local(img):
    """

    @param img: 1 x H x W torch
    @return:
    """
    if len(img.shape) == 2:
        img = img.unsqueeze(0)
    assert img.shape[0] == 1, 'Can only threshold grayscale'
    min_edge = min(img.shape[1], img.shape[2])

    def _do_thresh(img, min_entropy=None):
        res = torch.zeros((1, 2, img.shape[-2], img.shape[-1]), dtype=torch.float32, device=img.device)
        if min_entropy is not None:
            ent = torch_shannon_entropy(img)
            logger.debug('Patch entropy %0.3f' % ent)
            if ent < min_entropy:
                logger.debug('Skipping thresholding low entropy patch')
                res[0, 0, ...] = 1
                return res
        try:
            # Note this rarely happens when otsu can't be computed
            tmp = threshold_img(img.squeeze(0).squeeze(0), to_float=False)
            res[0, 0, tmp > 0.1] = 1
            res[0, 1, tmp <= 0.1] = 1
        except Exception as e:
            logger.error(f'Failed to threshold patch with error: {e}')
        return res

    img_entropy = torch_shannon_entropy(img)
    logger.debug('Image entropy: %0.3f' % img_entropy)

    def _do_thresh_entr(img):
        return _do_thresh(img, min_entropy=(img_entropy * 0.4))

    res = _do_thresh(img.unsqueeze(0), min_entropy=None)
    pfactors = [0.5, 0.3, 0.2]
    for pfactor in pfactors:
        patch_size = int(min_edge * pfactor)
        stride = patch_size // 2
        res += run_aggregate_patchwise(img.unsqueeze(0), patch_size, stride, _do_thresh_entr, noutput_channels=2)

    res = res.squeeze(0)
    indices = torch.max(res, dim=0).indices
    res = (indices == 0).reshape(img.shape).to(torch.float32)
    return res


def split_into_confidence_regions(gray, binimg, blurred, return_three_channels=False):
    """

    @param gray: 1 x H x W
    @param binimg: 1 x H x W
    @param blurred: 1 x H x W
    @param return_three_channels:
    @return:
    """
    # FG/BG zero by default
    assert gray.shape == binimg.shape == blurred.shape
    assert len(gray.shape) == 3
    neg_quant = -10
    pos_quant = 10
    failed = False
    try:
        neg_quant = torch.quantile(blurred[binimg < 0.1], 0.5)
        pos_quant = torch.quantile(blurred[binimg > 0.9], 0.1)
    except Exception as e:
        logger.debug('Could not get one of the quantiles (zero region?)')
        failed = True

    if return_three_channels:
        if failed:
            shp = [x for x in blurred.shape]
            shp[0] = 3
            res = torch.zeros(shp, dtype=torch.bool)
            if binimg.mean() > 0.99:
                res[0, ...] = 1
            elif binimg.mean() < 0.01:
                res[1, ...] = 1
        else:
            fg = blurred > pos_quant
            bg = blurred < neg_quant
            neither = torch.logical_not(torch.logical_or(fg, bg))
            res = torch.cat([fg, bg, neither], dim=0)
    else:
        res = torch.zeros_like(gray) + 0.5
        if not failed:
            res[blurred < neg_quant] = 0
            res[blurred > pos_quant] = 1
    return res


def run_aggregate_patchwise(images, patch_size, stride, function, noutput_channels=None):
    """

    @param images: B x C x H x W
    @param patch_size: int
    @param stride: int
    @param function: takes B x C x H' x W' and returns B x Cout x H' x W'
    @param noutput_channels: int Cout (or assumes same as image)
    @return:
    """
    assert len(images.shape) == 4, f'Unexpected shape {images.shape}'
    if noutput_channels is None:
        noutput_channels = images.shape[1]

    B, C, H, W = images.shape
    # B x C x npatchesi x npatchesj x patch_size x patch_size
    patches = images.unfold(3, patch_size, stride).unfold(2, patch_size, stride)

    out_shape = [x for x in patches.shape]
    out_shape[1] = noutput_channels
    res = torch.zeros(out_shape, dtype=torch.float32, device=patches.device)
    for i in range(patches.shape[2]):
        for j in range(patches.shape[3]):
            patch = patches[:, :, i, j, ...]
            res[:, :, i, j, ...] = function(patch)

    # reshape output to match F.fold input
    res = res.permute(0, 1, 2, 3, 5, 4)
    res = res.contiguous().view(B, noutput_channels, -1, patch_size * patch_size)  # [B, C, nb_patches_all, kernel_size*kernel_size]
    res = res.permute(0, 1, 3, 2)  # [B, C, kernel_size*kernel_size, nb_patches_all]
    res = res.contiguous().view(B, noutput_channels * patch_size * patch_size, -1)  # [B, C*prod(kernel_size), L] as expected by Fold

    fold = torch.nn.Fold(output_size=(H, W), kernel_size=patch_size, stride=stride)
    res = fold(res)
    return res


def get_rolling_confidence(img_gray_bin_blurred):
    """

        @param img_gray_bin_blurred: 3 x H x W of grayscale image, its binarization and blur of the binarization
        @param patch_size:
        @param stride:
        @return:
        """

    def _run_patch_function(patch):
        return split_into_confidence_regions(patch[0, :1, ...], patch[0, 1:2, ...], patch[0, 2:3, ...],
                                             return_three_channels=True).to(torch.float32).unsqueeze(0)

    min_edge = min(img_gray_bin_blurred.shape[-1], img_gray_bin_blurred.shape[-2])
    img_gray_bin_blurred = img_gray_bin_blurred.unsqueeze(0)
    res = 0
    pfactors = [0.5, 0.2]
    for pfactor in pfactors:
        patch_size = int(min_edge * pfactor)
        stride = patch_size // 4
        res = res + run_aggregate_patchwise(img_gray_bin_blurred, patch_size, stride, _run_patch_function, 3)

    res = res.squeeze(0)
    log_tensor(res, 'confidence', logger)
    indices = torch.max(res, dim=0).indices
    res = torch.stack([indices == 0, indices == 1, indices == 2])  # FG, BG, NEITHER
    return res


def encode_confidence_to_one_channel(conf):
    """

    @param conf: 3 x H x W (R==fg, G=bg, B=neither)
    @return:
    """
    mult = torch.tensor([1.0, 0.0, 0.5], dtype=torch.float32, device=conf.device).reshape((3, 1, 1))
    res = torch.sum(conf.to(torch.float32) * mult, dim=0, keepdim=True)
    return res

# from skimage.io import imsave; imsave('/tmp/conf.png', (res.permute(1,2,0) * 255).to(torch.uint8).numpy())


def resize_square_rgb(img, new_width, nchannels=3):
    if img.shape[0] == new_width and img.shape[1] == new_width:
        return img[:, :, 0:nchannels]
    else:
        return resize(img[:, :, 0:nchannels], (new_width, new_width, img.shape[2]), preserve_range=True)


class RandomPatchGenerator:
    '''
    Returns random patches from an image.
    '''

    def __init__(self, patch_width, patch_range=None, center_bias=False):
        """
        patch_width: output patch width
        patch_range: tuple of min/max in fraction of the original image, or None
                     to cut patches of fixed width patch_width without resizing
        center_bias: to bias patch selection to central pixels
        """
        self.patch_width = patch_width
        self.patch_range = patch_range
        self.center_bias = center_bias
        self.min_entropy = 0.0
        self.max_entropy = float('inf')
        self.num_filter_retries = 0
        self.entropy_channel = None

    def filter_by_entropy(self, min_entropy, max_entropy, channel=None, num_retries=5):
        """
        Args:
            min_entropy: minimum entropy for any accepted patch (to discard background-only)
            num_retries: number of retries to get a patch that meets criteria (to avoid infinite loop)
        """
        self.min_entropy = min_entropy if min_entropy is not None else 0.0
        self.max_entropy = max_entropy if max_entropy is not None else float('inf')
        self.num_filter_retries = num_retries
        self.entropy_channel = channel

    def get_random_pos(self, rwidth, img_width, img_height):
        if not self.center_bias:
            start_row = random.randint(0, img_height - rwidth)
            start_col = random.randint(0, img_width - rwidth)
        else:
            pos = np.random.normal([img_height / 2.0, img_width / 2.0],
                                   [img_height * 0.3, img_width * 0.3]) - rwidth / 2.0
            start_row = int(max(0, min(img_height - rwidth, pos[0])))
            start_col = int(max(0, min(img_width - rwidth, pos[1])))
        return start_row, start_col

    def random_patch(self, img, return_ind=None, resize=True):
        if self.num_filter_retries <= 0:
            pdata = self._random_patch(img, True)
        else:
            for i in range(self.num_filter_retries + 1):
                pdata = self._random_patch(img, True)
                entropy = shannon_entropy(
                    pdata[-1] if self.entropy_channel is None
                    else pdata[-1][..., self.entropy_channel])  # Pick correct channel
                if self.max_entropy >= entropy >= self.min_entropy:
                    break
            if entropy < self.min_entropy or entropy > self.max_entropy:
                logger.warning('Failed to generate patch with entropy within the range {} ~ {}'
                               .format(self.min_entropy, self.max_entropy))
        if resize:
            pdata = list(pdata)
            pdata[-1] = resize_square_rgb(pdata[-1], self.patch_width)

        if return_ind:
            return pdata
        else:
            return pdata[-1]

    def fixed_patch(self, img, start_col, start_row, rwidth, rheight, resize=True):
        patch = img[start_row:start_row + rheight, start_col:start_col + rwidth, :]

        if resize:
            patch = resize_square_rgb(patch, self.patch_width)

        return patch

    def get_random_patch_size(self, img):
        img_height = img.shape[0]
        img_width = img.shape[1]
        rwidth = self.patch_width
        if self.patch_range is not None:
            min_dim = min(img_width, img_height)
            rwidth = random.randint(
                min(min_dim, int(self.patch_range[0] * min_dim)),
                min(min_dim, int(self.patch_range[1] * min_dim)))
        return rwidth

    def _random_patch(self, img, return_ind):
        img_height = img.shape[0]
        img_width = img.shape[1]
        rwidth = self.get_random_patch_size(img)
        start_row, start_col = self.get_random_pos(rwidth, img_width, img_height)

        patch = img[start_row:start_row + rwidth, start_col:start_col + rwidth, :]

        if return_ind:
            return start_col, start_row, rwidth, rwidth, patch
        else:
            return patch