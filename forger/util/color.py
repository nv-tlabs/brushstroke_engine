# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import logging
import torch
from forger.util.logging import log_tensor

logger = logging.getLogger(__name__)

# COLOR MAP ------------------------------------------------------------------------------------------------------------

def colormap_simple(values, color_min=None, color_half=None, color_max=None):
    """

    @param values: any shape S0 x S1 ... x Sn tensor with values 0...1
    @param color_min:
    @param color_half:
    @param color_max:
    @return: float tensor S0 x S1 ... x Sn x 3 with colors for every value
    """
    if color_min is None:
        color_min = torch.tensor([82, 171, 245], dtype=torch.float32, device=values.device) / 255
    if color_half is None:
        color_half = torch.tensor([245, 253, 83], dtype=torch.float32, device=values.device) / 255
    if color_max is None:
        color_max = torch.tensor([209, 50, 38], dtype=torch.float32, device=values.device) / 255

    values_min = torch.min(values)
    values_max = torch.max(values)
    values_mean = torch.mean(values)

    shp = values.shape
    color_shp = tuple([1 for _ in range(len(shp))] + [3])
    color_min = color_min.reshape(color_shp)
    color_half = color_half.reshape(color_shp)
    color_max = color_max.reshape(color_shp)

    values_upper = ((values - 0.5) * 2.0).unsqueeze(-1).clamp(min=0.0, max=1.0)
    values_lower = (values * 2.0).unsqueeze(-1).clamp(min=0.0, max=1.0)
    result = (values.unsqueeze(-1) > 0.5).to(torch.float32) * (values_upper * color_max + (1 - values_upper) * color_half) + \
             (values.unsqueeze(-1) <= 0.5).to(torch.float32) * (values_lower * color_half + (1 - values_lower) * color_min)
    return result

# CONVERSIONS ----------------------------------------------------------------------------------------------------------
# Code is a debugged version of: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

# SRGB <--> Linear RGB -----------------------------
def srgb2linear_rgb(srgb_pixels):
    '''
    :param srgb_pixels: has shape [N, 3] with R,G,B in each row
    :return: same shape with linearized RGB
    '''
    linear_mask = (srgb_pixels <= 0.04045).to(torch.float32)
    exponential_mask = (srgb_pixels > 0.04045).to(torch.float32)
    rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
    return rgb_pixels


def linear_rgb2srgb(rgb_pixels):
    '''
    :param rgb_pixels: has shape [N, 3] with R,G,B in each row
    :return: same shape with RGB (for SRGB)
    '''
    linear_mask = (rgb_pixels <= 0.0031308).to(torch.float32)
    exponential_mask = (rgb_pixels > 0.0031308).to(torch.float32)
    srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * exponential_mask
    return srgb_pixels


# Linear RGB <--> XYZ -----------------------------
def linear_rgb2xyz(rgb_pixels):
    '''
    :param rgb_pixels: has shape [N, 3] with R,G,B in each row
    :return: same shape with XYZ
    '''
    rgb_to_xyz = torch.tensor([
        #    X        Y          Z
        [0.412453, 0.212671, 0.019334],  # R
        [0.357580, 0.715160, 0.119193],  # G
        [0.180423, 0.072169, 0.950227],  # B
    ], dtype=torch.float32, device=rgb_pixels.device)
    xyz_pixels = torch.matmul(rgb_pixels, rgb_to_xyz)
    return xyz_pixels


def xyz2linear_rgb(xyz_pixels):
    '''
    :param xyz_pixels: has shape [N, 3] with X,Y,Z in each row
    :return: same shape with RGB
    '''
    xyz_to_rgb = torch.tensor([
        #     r           g          b
        [3.2404542, -0.9692660, 0.0556434],  # x
        [-1.5371385, 1.8760108, -0.2040259],  # y
        [-0.4985314, 0.0415560, 1.0572252],  # z
    ], dtype=xyz_pixels.dtype, device=xyz_pixels.device)
    rgb_pixels = torch.matmul(xyz_pixels, xyz_to_rgb)
    # avoid a slightly negative number messing up the conversion
    torch.clamp(rgb_pixels, 0.0, 1.0)
    return rgb_pixels


# XYZ <--> LAB -------------------------------------
def xyz2lab(xyz_pixels):
    '''
    :param xyz_pixels: has shape [N, 3] with X,Y,Z in each row
    :return: same shape with LAB
    '''
    Xn = 0.95047
    Yn = 1.000
    Zn = 1.08883
    delta = 6.0 / 29.0
    D3 = delta ** 3.0
    D2INV3 = 1.0 / (3 * (delta ** 2))
    XnYnZn = torch.tensor([1.0 / Xn, 1.0 / Yn, 1.0 / Zn],
                          dtype=xyz_pixels.dtype, device=xyz_pixels.device).unsqueeze(0)

    xyz_normalized_pixels = xyz_pixels * XnYnZn

    linear_mask = (xyz_normalized_pixels < D3).to(torch.float32)
    exponential_mask = (xyz_normalized_pixels >= D3).to(torch.float32)

    eps = 1.0e-8  # stabilize cubed root gradient
    fxfyfz_pixels = (xyz_normalized_pixels * D2INV3 + 4.0 / 29) * linear_mask + \
                    (torch.pow(xyz_normalized_pixels + eps, (1.0 / 3.0))) * exponential_mask

    # convert to lab
    fxfyfz_to_lab = torch.tensor([
        #  l       a       b
        [0.0, 500.0, 0.0],  # fx
        [116.0, -500.0, 200.0],  # fy
        [0.0, 0.0, -200.0],  # fz
    ], dtype=xyz_pixels.dtype, device=xyz_pixels.device)
    lab_pixels = torch.matmul(fxfyfz_pixels, fxfyfz_to_lab) + \
                 torch.tensor([-16.0, 0.0, 0.0], dtype=xyz_pixels.dtype, device=xyz_pixels.device)
    return lab_pixels


def lab2xyz(lab_pixels):
    '''
    :param lab_pixels: has shape [N, 3] with L,A,B in each row
    :return: same shape with XYZ
    '''

    # convert to fxfyfz
    lab_to_fxfyfz = torch.tensor([
        #   fx      fy        fz
        [1 / 116.0, 1 / 116.0, 1 / 116.0],  # l
        [1 / 500.0, 0.0, 0.0],  # a
        [0.0, 0.0, -1 / 200.0],  # b
    ], dtype=lab_pixels.dtype, device=lab_pixels.device)
    fxfyfz_pixels = torch.matmul(
        lab_pixels + torch.tensor([16.0, 0.0, 0.0],
                                  dtype=lab_pixels.dtype, device=lab_pixels.device).unsqueeze(0),
        lab_to_fxfyfz)

    # convert to xyz
    epsilon = 6 / 29.0
    linear_mask = (fxfyfz_pixels <= epsilon).to(torch.float32)
    exponential_mask = (fxfyfz_pixels > epsilon).to(torch.float32)
    xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29.0)) * linear_mask + \
                 (fxfyfz_pixels ** 3) * exponential_mask
    # denormalize for D65 white point
    xyz_pixels = xyz_pixels * torch.tensor(
        [0.950456, 1.0, 1.088754],
        dtype=lab_pixels.dtype, device=lab_pixels.device).unsqueeze(0)
    return xyz_pixels


# Top level converters: SRGB <--> LAB ----------------------
def rgb2lab(srgb_pixels):
    '''
    Converts SRGB colors to CieLAB.
    :param srgb_pixels: has shape [N, 3] with R,G,B in each row
    :return: tensor of shape [N, 3], with L,A,B in each row
    '''
    rgb_pixels = srgb2linear_rgb(srgb_pixels)
    xyz_pixels = linear_rgb2xyz(rgb_pixels)
    lab_pixels = xyz2lab(xyz_pixels)
    return lab_pixels


def lab2rgb(lab_pixels):
    '''
    Converts CieLAB colors to SRGB.
    :param lab_pixels: has shape [N, 3], with L,A,B in each row
    :return: tensor of shape [N, 3] with R,G,B in each row
    '''
    xyz_pixels = lab2xyz(lab_pixels)
    rgb_pixels = xyz2linear_rgb(xyz_pixels)
    srgb_pixels = linear_rgb2srgb(rgb_pixels)
    return srgb_pixels


def rgb2lab_anyshape(colors, rgb_dim=-1):
    """
    Converts tensor of any shape to LAB.

    @param colors: torch float32 tensor of any shape with rgb_dim corresponding to 3 RGB values [0...1]
    @param rgb_dim: dimension where RGB values are located
    @return: same shape as colors, with rgb_dim values replaced with L*a*b* values
    """
    shape = colors.shape
    assert shape[rgb_dim] == 3, 'Rgb_dim {} must be 3 in shape {}'.format(rgb_dim, shape)

    if len(shape) == 1:
        return rgb2lab(colors.unsqueeze(0)).squeeze(0)

    tmp_colors = colors
    back_order = None
    if rgb_dim != -1:
        assert rgb_dim > 0, 'Negative indexing not supported'
        assert rgb_dim < len(shape), 'Wrong dim {} for shape {}'.format(rgb_dim, shape)

        order = list(range(0, len(shape)))
        order.remove(rgb_dim)
        order.append(rgb_dim)
        back_order = list(range(0, len(shape) - 1))
        back_order.insert(rgb_dim, len(shape) - 1)
        tmp_colors = colors.permute(*order)
        shape = tmp_colors.shape

    tmp_colors = tmp_colors.reshape(-1, 3)
    tmp_colors = rgb2lab(tmp_colors)
    tmp_colors = tmp_colors.reshape(shape)
    if back_order:
        tmp_colors = tmp_colors.permute(back_order)

    return tmp_colors

# hsv(color_0,v=0.5)
# hsv(color_1,s=0.9)
# rgbtargetloss(canvas,r=0.5,g=0.5,b=0.5)

# TODO(mshugrina): add this test for rgb2lab_anyshape to repo in a runnable state
# p = torch.rand((10, 3, 5, 7))
# p_lab = forger.util.color.rgb2lab_anyshape(p, 1)
# for b in range(p.shape[0]):
#     for r in range(p.shape[2]):
#         for c in range(p.shape[3]):
#             actual = p_lab[b, :, r, c].unsqueeze(0)
#             expected = forger.util.color.rgb2lab(p[b, :, r, c].unsqueeze(0))
#             passed = torch.allclose(actual, expected, atol=0.001)
#             if not passed:
#                 print(actual)
#                 print(expected)
#                 assert passed