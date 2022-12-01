# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import logging
import lpips
import numpy as np
from torch import nn
import math
import numbers
import torch
from torch.nn import functional as F
import torchvision.transforms
from forger.util.logging import log_tensor

import forger.util.color

logger = logging.getLogger(__name__)

# TODO(aliceli): restore when this is merged
#from thirdparty.stylegan2_ada_pytorch.training.geom_loss import gradientLayer

# Code largely borrowed from stylegan2
#----------------------------------------------------------------------------

_mse_loss = nn.MSELoss()
_l1_loss = nn.L1Loss(reduction='sum')
#_gaussian_gradient_layer = gradientLayer()
_squared_error = nn.MSELoss(reduction='sum')

# TODO: The foreground used here is INCORRECT; please do not use geom_metrics.
#@metric_main.register_metric_func(category='geom')
def mse(triad_data, geom_data):
    foreground = triad_data['uvs'][:, :1, :, :] + triad_data['uvs'][:, 1:2, :, :]
    return _mse_loss(foreground, 1.0 - geom_data)


#@metric_main.register_metric_func(category='geom')
def sad(triad_data, geom_data):
    foreground = triad_data['uvs'][:, :1, :, :] + triad_data['uvs'][:, 1:2, :, :]
    return _l1_loss(foreground, 1.0 - geom_data)


def random_patches(images, patch_width):
    """

    @param images: B x ch x W x W
    @param patch_width: int
    @return: B x ch x pW x pW
    """
    crop = torchvision.transforms.RandomCrop(patch_width)
    return crop(images)


class GaussianSmoothing(nn.Module):
    """
    Source: https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/9
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=2)


def default_gaussian_smoothing(img):
    smoothing = GaussianSmoothing(img.shape[1], 5, 1).to(img.device)
    output = smoothing(img)
    return output


def get_conservative_fg_bg(geom):
    geom = geom
    geom_blur = default_gaussian_smoothing(default_gaussian_smoothing(geom))

    BG_THRESH = 0.999
    FG_THRESH = 0.1
    bg = geom_blur >= BG_THRESH
    fg = geom_blur < FG_THRESH
    return fg, bg


def compute_transparency_metrics(renders, geom):
    """
    @param renders: B x 4 x W x W rendering float32 tensor [0...1] rendered stroke images for the *same style*
    @param geom: B x 1 x W x W geometry guidance float32 tensor [0...1] (0 == FG) geometry guidance

    @param renders:
    @param geom:
    @return:
    """
    alphas = renders[:, 3, ...]
    geom_blur = default_gaussian_smoothing(default_gaussian_smoothing(geom)).squeeze(1)
    geom = geom.squeeze(1)

    BG_THRESH = 0.999
    FG_THRESH = 0.3
    bg_clarity = 1 - torch.mean(alphas[geom_blur > BG_THRESH]).item()
    fg_opacity = torch.median(alphas[geom < FG_THRESH]).item()

    return {'BG_CLARITY_MEAN': bg_clarity,
            'FG_OPACITY_MEDIAN': fg_opacity}


def compute_stitching_metrics(stitching_result, margin):
    """
    See RandomStitcher.generate_with_stitching.

    @param stitching_result:
    @param margin:
    @return:
    """
    def _crop(img):
        if margin == 0:
            return img
        B, C, H, W = img.shape
        return img[:, :, margin:H-margin*2, margin:W-margin*2]

    l1 = nn.L1Loss()

    def _metrics(im1, im2):
        return lpips_batched(im1, im2).mean(), l1(im1, im2)

    lpips1, l11 = _metrics(_crop(stitching_result['fake1']), _crop(stitching_result['fake1_composite']))
    lpips2, l12 = _metrics(_crop(stitching_result['fake2']), _crop(stitching_result['fake2_composite']))
    return {'STITCH_LPIPS': 0.5 * (lpips1 + lpips2),
            'STITCH_L1': 0.5 * (l11 + l12)}


def compute_lpips_across_geo(renders):
    """

    @param renders: B x 4 x W x W rendering float32 tensor [0...1] rendered stroke images for the *same style* and color
    @return:
    """
    # Composite over white canvas
    alpha = renders[:, 3, ...].unsqueeze(1)
    renders_rgb = alpha * renders[:, :3, ...] + (1 - alpha) * 1.0

    renders_proc = renders_rgb * 2 - 1.0
    renders_proc_perm = renders_proc[torch.randperm(renders.shape[0]), ...]
    scores = lpips_batched(renders_proc, renders_proc_perm)
    result = {'LPIPS_ACROSS_GEO': scores.mean().item()}
    return result


def compute_uniform_bg_lpips_metric(renders, geom, patch_width=None, same_style=False, return_debug=False,
                                    key_suffix=None):
    """
    Computes lpips geometry metrics based *only* on final rendered image.

    @param key_suffix: suffix to add to key in the returned dict
    @param return_debug: returns debug img if set
    @param same_style: assumes all renders to be of the same style/color, so will form random pairs
    @param patch_width: scale of patches to compare
    @param renders: B x 4 x W x W rendering float32 tensor [0...1] rendered stroke images for the *same style*
    @param geom: B x 1 x W x W geometry guidance float32 tensor [0...1] (0 == FG) geometry guidance
    @return:
    """
    result_key = 'LPIPS_UNIFORM_BG'
    if key_suffix is not None:
        result_key = '{}_{}'.format(result_key, key_suffix)

    if patch_width is None:
        patch_width = renders.shape[-1] // 4
        if patch_width < 64:
            patch_width = renders.shape[-1] // 2
            if patch_width < 64:
                patch_width = int(0.8 * renders.shape[-1])

    # Composite over white canvas
    alpha = renders[:, 3, ...].unsqueeze(1)
    renders_rgb = alpha * renders[:, :3, ...] + (1 - alpha) * 1.0

    geom = default_gaussian_smoothing(geom)

    # TODO: blur guidance to remove boundaries from the BG regions
    BG_THRESH = 0.99
    bg_mask = (geom > BG_THRESH).to(torch.float32)
    # B x 3 x 1 x 1
    mean_colors = torch.sum(renders_rgb * bg_mask, dim=(2, 3)) / torch.clamp(torch.sum(bg_mask, dim=(2, 3)), min=1.0)
    mean_colors = mean_colors.unsqueeze(-1).unsqueeze(-1)
    # B x 4 x W x W
    renders_rgb = torch.cat([renders_rgb, geom], dim=1)

    # Get 2 random patches from each image
    patches0 = random_patches(renders_rgb, patch_width)
    patches1 = random_patches(renders_rgb, patch_width)
    # Also flip patches1
    patches1 = patches1.permute(0, 1, 3, 2)
    if same_style:  # randomize comparison order as well
        patches1 = patches1[torch.randperm(patches1.shape[0]), ...]

    # Unset the background values using geometry guidance for *both* patches
    bg_mask = torch.logical_and(patches0[:, 3, ...] > BG_THRESH, patches1[:, 3, ...] > BG_THRESH).to(torch.float32).unsqueeze(1)
    patches0_proc = bg_mask * patches0[:, :3, ...] + (1 - bg_mask) * mean_colors
    patches1_proc = bg_mask * patches1[:, :3, ...] + (1 - bg_mask) * mean_colors
    patches0_proc = patches0_proc * 2 - 1.0
    patches1_proc = patches1_proc * 2 - 1.0
    log_tensor(patches0, "patches 0", logger, print_stats=True)
    scores = lpips_batched(patches0_proc, patches1_proc)
    result = {result_key : scores.mean().item()}

    if return_debug:
        def _touint8np(im):
            return (im.permute(1, 2, 0).detach().cpu() * 255).to(torch.uint8).numpy()

        margin = 2
        width = renders.shape[-1]
        row_width = max(3 * patch_width + 4 * margin, width)
        debug_img = np.zeros((renders.shape[0] * (row_width + margin),
                              (width + margin * 7 + patch_width * 6), 4), dtype=np.uint8)
        colors = forger.util.color.colormap_simple(scores)
        colors = (colors.cpu() * 255).to(torch.uint8)
        for b in range(renders.shape[0]):
            cstart = 0
            debug_img[b * width:b * width + width, cstart:cstart+width, :] = _touint8np(renders_rgb[b, ...])
            debug_img[b * width:b * width + width, cstart:cstart + width, 3] = 255
            cstart += width + margin

            rstart = b * (row_width + margin)
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, :3] = _touint8np(patches0[b, :3, ...])
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, 3] = 255
            cstart += (patch_width + margin)
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, :3] = _touint8np(patches1[b, :3, ...])
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, 3] = 255
            rstart = rstart + patch_width + margin
            cstart -= (patch_width + margin)
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, :] = _touint8np(patches0[b, 3, ...].unsqueeze(0))
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, 3] = 255
            cstart += (patch_width + margin)
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, :] = _touint8np(patches1[b, 3, ...].unsqueeze(0))
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, 3] = 255
            rstart = rstart + patch_width + margin
            cstart -= (patch_width + margin)

            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, :3] = _touint8np(patches0_proc[b, ...] / 2.0 + 0.5)
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, 3] = 255
            cstart += (patch_width + margin)
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, :3] = _touint8np(patches1_proc[b, ...] / 2.0 + 0.5)
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, 3] = 255
            cstart += (patch_width + margin)
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, :3] = colors[b].unsqueeze(0).unsqueeze(0).numpy()
            debug_img[rstart:rstart + patch_width, cstart:cstart + patch_width, 3] = 255
        return result, debug_img

    return result, None


    # Compute LPIPS
_lpips_alex = None
def lpips_batched(im1, im2):
    """

    :param im1: torch float32 Bx3xHxW RGB image normalized to [-1,1] (also works unbatched 3xHxW)
    :param im2: torch float32 Bx3xHxW RGB image normalized to [-1,1] (also works unbatched 3xHxW)
    :return:
    """
    global _lpips_alex
    if _lpips_alex is None:
        _lpips_alex = lpips.LPIPS(net='alex').to(torch.device(0))

    return _lpips_alex(im1, im2).squeeze()


_lpips_vgg = None
def lpips_batched_vgg(im1, im2):
    """

    :param im1: torch float32 Bx3xHxW RGB image normalized to [-1,1] (also works unbatched 3xHxW)
    :param im2: torch float32 Bx3xHxW RGB image normalized to [-1,1] (also works unbatched 3xHxW)
    :return:
    """
    global _lpips_vgg
    if _lpips_vgg is None:
        _lpips_vgg = lpips.LPIPS(net='vgg').to(torch.device(0))

    return _lpips_vgg(im1, im2).squeeze()
