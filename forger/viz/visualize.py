# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import argparse
import os
import logging
import numpy as np
import time
# TODO: would be best to standardize image output functions, but this is pooled from many sources
from skimage.io import imsave, imread
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.utils.data
from PIL import Image, ImageDraw, ImageFont

import forger.ui.brush
import forger.util.logging
import forger.metrics.util
import forger.train.stitching
from forger.util.logging import log_tensor
from forger.util.torch_data import get_image_data_iterator, get_image_data_iterator_from_dataset

logger = logging.getLogger(__name__)

DEFAULT_MARGIN = 0

BUNDLED_IMAGES_PATH = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 os.path.pardir, 'images'))

BUNDLED_GEOMETRY_PATH = os.path.join(BUNDLED_IMAGES_PATH, 'spline_patches_curated')

BUNDLED_FONTS_PATH = os.path.realpath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),
                 os.path.pardir, 'resources', 'fonts'))

DEFAULT_COLORS = ['74,97,209', '202,42,29', '228,223,78', '116,226,189', '30,30,30']


def parse_color_list(flag_input):
    """
    Parses command line colors into a tensor.

    @param flag_input: List of CSV rgb values [0...255] of length L
    @return: L x 3 torch float32 tensor [0...1]
    """
    if flag_input is None:
        return None

    if len(flag_input) == 1 and flag_input[0] == 'default':
        logger.info('Using default custom color list')
        flag_input = DEFAULT_COLORS

    result = torch.zeros((len(flag_input), 3), dtype=torch.uint8)
    for i, v in enumerate(flag_input):
        result[i, :] = torch.tensor([int(x) for x in v.strip().split(',')])
    return result.to(torch.float32) / 255


def get_default_eval_directory(gan_checkpoint):
    bname = os.path.basename(gan_checkpoint).replace('network-snapshot-', '').replace('.pkl', '')
    basedir = os.path.dirname(gan_checkpoint)
    return os.path.join(basedir, 'eval', 'model_%s' % bname)


def load_default_stitching_image(width=None):
    crops = torch.tensor([[122, 0, 256, 256], [40, 142, 256, 256], [206, 112, 256, 256],
                          [158, 244, 256, 256], [88, 372, 256, 256], [104, 560, 256, 256],
                          [188, 724, 256, 256], [74, 668, 256, 256], [244, 578, 256, 256],
                          [254, 412, 256, 256], [102, 834, 256, 256],
                          [144, 1150, 256, 256], [144, 1255, 256, 256], [144, 1360, 256, 256],
                          [144, 1465, 256, 256], [144, 1570, 256, 256], [144, 1690, 256, 256],
                          ],
                         dtype=torch.int64)

    img = load_geometry_image(os.path.join(BUNDLED_IMAGES_PATH, 'large_guidance', 'stitching_img.png')).permute(2, 0, 1)
    C, H, W = img.shape
    if H > width * 2:
        img = torchvision.transforms.Resize(width * 2)(img)
        factor = H // (width * 2)
        crops = (crops / factor).to(torch.int64)

    patches = torch.stack([torchvision.transforms.functional.crop(
        img, crops[i][0], crops[i][1], crops[i][2], crops[i][3]) for i in range(crops.shape[0])])

    return img.permute(1, 2, 0), patches, crops


def generate_stitched_image(crops, result_size, patches, margin=0, clear=False):
    """

    @param crops:
    @param result_size:
    @param patches:
    @return:
    """
    B, C, H, W = patches.shape
    result = torch.ones((C, result_size[0], result_size[1]),
                        dtype=patches.dtype, device=patches.device)
    if clear:
        result = result * -1
    for i in range(B):
        crop = crops[i]
        assert crop[2] == H
        assert crop[3] == W
        result[:, crop[0]+margin:crop[0]-margin*2+H, crop[1]+margin:crop[1]+W-margin*2] = \
            patches[i, :, margin:H-margin*2, margin:W-margin*2]
    return result


def load_default_geometry_image(width=None):
    return load_bundled_geometry_image('cross_rad016.png', width=width)


def load_default_curated_geometry_images(width=None):
    fnames = ['cross_rad016.png',
              'curve_rad025.png',
              'end_rad016.png',
              'many_rad009.png',
              'line_rad003.png',
              'curve2_rad001.png']
    return torch.stack([load_bundled_geometry_image(f, width) for f in fnames])


# TODO: replace with bundled.py
def load_bundled_geometry_image(basename, width=None):
    if not os.path.isdir(BUNDLED_GEOMETRY_PATH):
        raise RuntimeError('Expected images at path {}'.format(BUNDLED_GEOMETRY_PATH))

    path = os.path.join(BUNDLED_GEOMETRY_PATH, basename)
    if not os.path.isfile(path):
        raise RuntimeError('Expected image not found: {}'.format(path))

    return load_geometry_image(path, width)


# TODO: replace with bundled.py
def load_geometry_image(path, width=None):
    res = torch.from_numpy(imread(path))
    if width is not None:
        res = torchvision.transforms.Resize(width)(res.permute(2, 0, 1)).permute(1, 2, 0)
    return res


def torch_image_with_text(text, rows, cols=None, font_size=None):
    """
    @param text:
    @return: 3 x W x W float32 [-1..1] torch image
    """
    if cols is None:
        cols = rows
    if font_size is None:
        font_size = max(15, int(min(rows, cols) / 256 * 45))

    return torch.from_numpy(
        write_text_on_image(np.ones((rows, cols, 3), dtype=np.uint8) * 255, text, font_size=font_size)) \
               .to(torch.float32).permute(2, 0, 1) / 255 * 2 - 1


def write_text_on_image(np_uint8_image, text, font_size=35, color='rgb(0, 0, 0)'):
    """
    Takes and returns a numpy uint8 array.
    """
    pilim = Image.fromarray(np_uint8_image)
    font = ImageFont.truetype(os.path.join(BUNDLED_FONTS_PATH, 'OpenSans-Regular.ttf'), size=font_size)
    draw = ImageDraw.Draw(pilim)
    tw, th = draw.textsize(text, font=font)
    iw = np_uint8_image.shape[1]
    ih = np_uint8_image.shape[0]

    start_x = 0
    start_y = 0
    if tw > iw:
        logger.warning('Text width is wider than image')
    else:
        start_x = (iw - tw) // 2

    if th > ih:
        logger.warning('Text height is larger than image height')
    else:
        start_y = (ih - th) // 2

    draw.text((start_x, start_y), text, font=font, fill=color)
    return np.array(pilim)


def fill_image_row(viz_img, rstart, cstart, img_batch, margin=DEFAULT_MARGIN):
    fill_image_row_or_col(viz_img, rstart, cstart, img_batch, margin, fill_row=True)


def fill_image_col(viz_img, rstart, cstart, img_batch, margin=DEFAULT_MARGIN):
    fill_image_row_or_col(viz_img, rstart, cstart, img_batch, margin, fill_row=False)


def fill_image_row_or_col(viz_img, rstart, cstart, img_batch, margin, fill_row=True):
    """
    Fills in a row of a numpy visualization image from a batched image tensor.

    @param viz_img: W' x H' x 4 numpy uint8 array
    @param rstart: row to start at
    @param cstart: col to start at
    @param img_batch: B x {1,3,4} x W x W torch float32 [0...1]
    @return:
    """
    nbatches = img_batch.shape[0]
    nchannels = img_batch.shape[1]
    width = img_batch.shape[-1]

    if nchannels == 1:
        img_batch = torch.cat([img_batch.expand(-1, 3, -1, -1), torch.ones_like(img_batch)], dim=1)
    elif nchannels == 3:
        img_batch = torch.cat([img_batch, torch.ones_like(img_batch[:, :1, ...])], dim=1)
    elif nchannels != 4:
        raise RuntimeError('Unsupported num channels {} in shape {}'.format(nchannels, img_batch.shape))

    img_batch = (img_batch.detach().permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()

    for b in range(nbatches):
        viz_img[rstart:rstart + width, cstart:cstart + width, :] = img_batch[b, ...]
        if fill_row:
            cstart = cstart + width + margin
        else:
            rstart = rstart + width + margin


# Note: this function is from original stylegan-ada-pytorch repo:
# https://github.com/NVlabs/stylegan2-ada-pytorch
def setup_snapshot_image_grid(training_set, crop_transform=None, random_seed=0):
    shape = crop_transform.size if crop_transform is not None else training_set.image_shape[1:]
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // shape[1], 7, 32)
    gh = np.clip(4320 // shape[0], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict()  # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    if crop_transform is not None:
        images = [crop_transform(torch.from_numpy(img)).numpy() for img in images]
    return (gw, gh), np.stack(images), np.stack(labels)


# Note: this function is from original stylegan-ada-pytorch repo:
# https://github.com/NVlabs/stylegan2-ada-pytorch
def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        Image.fromarray(img, 'RGB').save(fname)


def output_encoder_diagnostics(geom_input, geom_autoencoder, device, run_dir, geom_input_channel=1):
    def _make_viz(gt_input, final_output, max_items=10):
        max_items = min(max_items, gt_input.shape[0])
        in_grid = torchvision.utils.make_grid(gt_input[0:max_items, ...].to('cpu'),
                                              nrow=max_items, padding=0)
        out_grid = torchvision.utils.make_grid(final_output[0:max_items, ...].to('cpu'),
                                               nrow=max_items, padding=0)
        grid = torch.cat([in_grid, out_grid], dim=-2)
        return grid

    gray_input = geom_input[:, geom_input_channel:geom_input_channel + 1, :, :]
    gray_input = gray_input.to(torch.float32).to(device) / 255.0
    geom_output = geom_autoencoder.postprocess(geom_autoencoder(geom_autoencoder.preprocess(gray_input))).cpu()

    geom_grid = _make_viz(gt_input=gray_input, final_output=geom_output)
    geom_grid = np.transpose(geom_grid.numpy(), (1, 2, 0))

    plt.imsave(fname=os.path.join(run_dir, "geom_enc_result.jpg"), arr=geom_grid)


def compose_stroke(uvs, colors):
    """
    @param uvs: torch.Tensor of size B x 3 x W x H
    @param colors: torch.Tensor of size B x 3 x 3
    @return
    """
    # (B x 1 x ncolors x W x W) * (B x C x ncolors x 1 x 1)  = B x C x ncolors x W x W
    tmp = uvs.unsqueeze(1) * colors.unsqueeze(-1).unsqueeze(-1)
    return torch.sum(tmp, dim=2)


def compose_stroke_with_canvas(raw_output, mode, bg_uvs_idx=2, primary_color_idx=0,
                               primary_colors=None, blur_weight=0.8):
    """
    Runs composition using raw generator output (e.g. obtained by setting return_debug_data=True)

    @param raw_output:
    @param mode:
    @param bg_uvs_idx:
    @param primary_color_idx:
    @param primary_colors: B x 3
    @return:
    """
    # Note that it would be best to use GanPaintEngine, but the use case for training/eval are somewhat different
    modes = {"clear_stroke", "stroke_over_blurred_canvas", "stroke_over_white_canvas"}
    assert mode in modes

    # B x C x ncolors
    colors = raw_output['colors']
    if primary_colors is not None:
        colors = colors.detach().clone()
        colors[:, :, primary_color_idx] = primary_colors

    stroke = compose_stroke(raw_output['uvs'], colors)

    if 'alpha_fg' in raw_output:
        alpha = raw_output['alpha_fg']
    else:
        # Assume BG is S
        alpha = 1 - raw_output['uvs'][:, bg_uvs_idx, ...].unsqueeze(1)

    if 'canvas' not in raw_output:
        canvas = torch.ones_like(stroke) * 2 - 1.0
    else:
        canvas = raw_output['canvas']

    if mode == "clear_stroke":
        return torch.cat([stroke, alpha * 2 - 1], dim=1)
    elif mode == "stroke_over_blurred_canvas":
        mean_colors = canvas.mean(dim=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
        canvas = canvas * (1 - blur_weight) + mean_colors * blur_weight
        return stroke * alpha + canvas * (1 - alpha)
    elif mode == "stroke_over_white_canvas":
        canvas = torch.ones_like(stroke) * 2 - 1.0
        return stroke * alpha + canvas * (1 - alpha)


def visualize_raw_data(raw_output, return_legend_row=False):
    """

    @param raw_output: dictionary expected to contain
    uvs : B x 3 x W x W torch float tensor [0...1]
    colors : B x C x ncolors torch float tensor [-1..1]
    canvas (optional) : B x 3 x W x W [-1..1]
    alpha_fg (optional) : B x 1 x W x W [0..1]

    @return: [-1..1] 3 x W2 x H2  and optionally [-1..1] 3 x W2 x H3 legend
    """
    W = raw_output['uvs'].shape[-1]
    legend = []
    columns = []
    color_width = W // 3

    # Include UVS
    for idx, name in zip([0, 1, 2], ['U', 'V', 'S']):
        columns.append(
            torchvision.utils.make_grid(
                raw_output['uvs'][:, idx:idx + 1, ...].expand(-1, 3, -1, -1), nrow=1, padding=0) * 2 - 1)
        if return_legend_row:
            legend.append(torch_image_with_text(name, rows=W))

    # Include Colors
    for idx in range(3):
        columns.append(
            torchvision.utils.make_grid(
                raw_output['colors'][..., idx].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, W, color_width),
                nrow=1, padding=0))
    if return_legend_row:
        legend.append(torch_image_with_text('colors', rows=W, cols=(color_width * 3)))

    # Include canvas if present
    if 'canvas' in raw_output:
        columns.append(
            torchvision.utils.make_grid(
                raw_output['canvas'].clip(-1, 1), nrow=1, padding=0))

        # Also include stroke only
        columns.append(
            torchvision.utils.make_grid(
                compose_stroke(raw_output['uvs'], raw_output['colors']), nrow=1, padding=0))

        if return_legend_row:
            legend.append(torch_image_with_text('canvas', rows=W))
            legend.append(torch_image_with_text('stroke', rows=W))

    if 'alpha_fg' in raw_output:
        columns.append(
            torchvision.utils.make_grid(
                raw_output['alpha_fg'].expand(-1, 3, -1, -1) * 2 - 1, nrow=1, padding=0))
        if return_legend_row:
            legend.append(torch_image_with_text('alpha_fg', rows=W))

    columns = torch.cat(columns, dim=-1)
    if not return_legend_row:
        return columns

    legend = torch.cat(legend, dim=-1)
    return columns, legend

    # torch.tile(raw_output['colors'][0, ...].unsqueeze(-2), (1, 256, 10))
    # imsave('/tmp/c.png', ((torchvision.utils.make_grid(raw_output['colors'][..., 0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 256, 50), nrow=1, padding=0) /2 - 0.5) * 255).permute(1,2,0).cpu().to(torch.uint8).numpy())


class TrainingVisualizer(object):
    def __init__(self, device, batch_gpu, width=None):
        self.device = device
        self.batch_gpu = batch_gpu
        self.width = width

        # Random inputs for style grid visualization
        self.grid_size = None
        self.grid_z = None
        self.grid_c = None
        self.grid_geom_feature = None

        # Fixed guidance inputs
        self.fixed_geom_img = None
        self.fixed_geom_feature = None
        self.debug_z = None
        self.curated_geom_feature = None
        self.curated_geom_images = None
        self.curated_primary_colors = None

        # Stitching
        self.stitching_image = None
        self.stitching_patches_features = None
        self.stitching_crops = None
        self.stitching_positions = None
        self.crop_margin = 10

    def init(self, viz_dir, style_set, geom_set, geom_encoder, z_dim, geom_input_channel=1):
        gic = geom_input_channel

        logger.info('Exporting sample style images...')
        self.grid_size, style_samples, labels = setup_snapshot_image_grid(training_set=style_set)
        save_image_grid(style_samples, os.path.join(viz_dir, 'reals_style.png'),
                        drange=[0, 255], grid_size=self.grid_size)

        logger.info('Exporting sample spline images...')
        _, geom_samples, _ = setup_snapshot_image_grid(training_set=geom_set)
        save_image_grid(geom_samples, os.path.join(viz_dir, 'reals_geom.png'),
                        drange=[0, 255], grid_size=self.grid_size)

        logger.info('Initializing grid Zs...')
        self.grid_z = torch.randn([labels.shape[0], z_dim], device=self.device).split(self.batch_gpu)
        self.grid_c = torch.from_numpy(labels).to(self.device).split(self.batch_gpu)

        logger.info('Initializing grid geom...')
        geom_set_iterator = get_image_data_iterator_from_dataset(geom_set, self.batch_gpu, shuffle=True)
        self.grid_geom_feature = [
            geom_encoder.encode(
                next(geom_set_iterator)[0][0:x.shape[0], gic:gic+1, ...].to(torch.float32).to(self.device) / 255.0)
            for x in self.grid_z]

        logger.info('Initializing fixed geom...')
        self.fixed_geom_img = load_default_geometry_image(self.width).to(torch.float32) / 255.0  # W x W x 3
        self.fixed_geom_feature = geom_encoder.encode(
            self.fixed_geom_img[..., 0].to(self.device).unsqueeze(0).unsqueeze(0))
        self.fixed_geom_img = self.fixed_geom_img.permute(2, 0, 1).numpy() * 2.0 - 1.0

        logger.info('Initializing control input and debug Zs...')
        self.debug_z = torch.randn([self.grid_size[1], z_dim], device=self.device).split(self.batch_gpu)
        self.curated_primary_colors = forger.viz.visualize.parse_color_list(['default']) * 2 - 1.0
        self.curated_geom_images = forger.viz.visualize.load_default_curated_geometry_images(self.width).to(torch.float32) / 255.0
        self.curated_geom_feature = geom_encoder.encode(
            self.curated_geom_images[..., 0].to(self.device).unsqueeze(1))
        self.curated_geom_images = self.curated_geom_images.permute(0, 3, 1, 2) * 2.0 - 1.0

        logger.info('Initializing stitching geometry...')
        self.stitching_image, stitching_patches, self.stitching_crops = \
            forger.viz.visualize.load_default_stitching_image(self.width)
        self.stitching_image = self.stitching_image.to(self.device).permute(2, 0, 1).to(torch.float32) / 255.0 * 2 - 1.0
        stitching_patches = stitching_patches.to(torch.float32) / 255.0
        self.stitching_patches_features = geom_encoder.encode(
            stitching_patches[:, :1, ...].to(self.device))
        self.stitching_positions = self.stitching_crops[:, :2].to(torch.int64).to(self.device)

    def visualize_stitching(self, G_ema, fname):
        logger.info('Outputting stitching diagnostics...')

        result_size = [self.stitching_image.shape[1], self.stitching_image.shape[2]]
        image_pos = torch.stack([
             generate_stitched_image(
                 self.stitching_crops, result_size,
                 G_ema(z=self.debug_z[0][i:i+1, ...].expand(self.stitching_patches_features[0].shape[0], -1), c=[],
                       geom_feature=self.stitching_patches_features,
                       positions=self.stitching_positions,
                       noise_mode='const', return_debug_data=False), margin=self.crop_margin)
             for i in range(self.debug_z[0].shape[0])])
        image_nopos = torch.stack([
             generate_stitched_image(
                 self.stitching_crops, result_size,
                 G_ema(z=self.debug_z[0][i*4:i*4+1, ...].expand(self.stitching_patches_features[0].shape[0], -1), c=[],
                       geom_feature=self.stitching_patches_features,
                       positions=None,
                       noise_mode='const', return_debug_data=False), margin=self.crop_margin)
             for i in range(self.debug_z[0].shape[0] // 4)])
        image_pos = torchvision.utils.make_grid(image_pos, nrow=4, padding=0)
        image_nopos = torchvision.utils.make_grid(image_nopos, nrow=1, padding=0)

        legend_height = G_ema.img_resolution // 2
        legend = torch.cat([torch_image_with_text("Random positions", legend_height, image_nopos.shape[-1]),  # [0..1]
                            torch_image_with_text("Correct positions", legend_height, image_pos.shape[-1])], dim=2).\
                     to(image_pos.device) * 2 - 1
        legend2 = torch.cat([self.stitching_image,
                             torch.ones_like(image_pos[:, :self.stitching_image.shape[1], :])], dim=2)
        image = torch.cat([legend, legend2,
                           torch.cat([image_nopos, image_pos], dim=2)], dim=1).cpu()
        image = ((image.permute(1, 2, 0) / 2.0 + 0.5) * 255).cpu().clip(0, 255).to(torch.uint8).numpy()
        imsave(fname, image)

    def do_visualize(self, viz_dir, G_ema, fname_prefix: str):

        def _fname_png(bname):
            return os.path.join(viz_dir, '%s_%s.png' % (fname_prefix, bname))

        def _add_alpha(im):
            return torch.cat([im, torch.ones_like(im[0, ...].unsqueeze(0)) * 2 - 1], dim=0)

        def _color_legend_image(width, color):
            res = torch.ones((3, width, width), dtype=color.dtype)
            res[:, width//4:(width - width//4), width//4:(width - width//4)] = color.unsqueeze(-1).unsqueeze(-1)
            return res

        legend_height = 100

        logger.info('Outputting fakes...')
        images = torch.cat([G_ema(z=z, c=c, geom_feature=gf, noise_mode='const').cpu()
                            for z, c, gf in zip(self.grid_z, self.grid_c, self.grid_geom_feature)]).numpy()
        save_image_grid(images, _fname_png('fakes'), drange=[-1, 1], grid_size=self.grid_size)

        logger.info('Outputting fakes (fixed geo)...')
        images = torch.cat([G_ema(z=z, c=c,
                                  geom_feature=[geof.expand(z.shape[0], -1, -1, -1) for geof in self.fixed_geom_feature],
                                  noise_mode='const').cpu()
                            for z, c in zip(self.grid_z, self.grid_c)]).numpy()
        images[0] = self.fixed_geom_img
        save_image_grid(images, _fname_png('fakes_fixedgeo'), drange=[-1, 1], grid_size=self.grid_size)

        logger.info('Outputting control diagnostics...')
        # list of N tensors B x ... ->
        raw_results = [[G_ema(z=z, c=[],
                              geom_feature=[geof[g_idx, ...].expand(z.shape[0], -1, -1, -1) for geof in self.curated_geom_feature],
                              noise_mode='const', return_debug_data=True)
                        for z in self.debug_z] for g_idx in range(self.curated_geom_feature[0].shape[0])]

        # Full style rendering using only the first curated geometry image
        image = torchvision.utils.make_grid(
                torch.cat([x[0] for x in raw_results[0]], dim=0), nrow=1, padding=0).cpu()
        image = _add_alpha(image)
        images = [image]
        legend_row0 = [torch_image_with_text("Style (full)", legend_height, image.shape[-1])]  # [0..1]
        legend_row1 = [self.curated_geom_images[0, ...]]

        # Clear style rendering using only the first geometry image
        image = torchvision.utils.make_grid(
                    torch.cat([compose_stroke_with_canvas(x[1], mode="clear_stroke")
                               for x in raw_results[0]], dim=0), nrow=1, padding=0).cpu()
        images.append(image)
        legend_row0.append(torch_image_with_text("Style (clear)", legend_height, image.shape[-1]))
        legend_row1.append(self.curated_geom_images[0, ...])

        # If there is canvas, also show blurred canvas (for FID computation debugging)
        if 'canvas' in raw_results[0][0][1]:
            image = torchvision.utils.make_grid(
                torch.cat([compose_stroke_with_canvas(x[1], mode="stroke_over_blurred_canvas")
                           for x in raw_results[0]], dim=0), nrow=1, padding=0).cpu()
            images.append(_add_alpha(image))
            legend_row0.append(torch_image_with_text("Blur canvas", legend_height, image.shape[-1]))
            legend_row1.append(self.curated_geom_images[0, ...])

        # Visualize debug outputs
        raw_output_viz = [visualize_raw_data(x[1], return_legend_row=True)
                          for x in raw_results[0]]
        image = torch.cat([x[0] for x in raw_output_viz], dim=1)
        images.append(_add_alpha(image).cpu())
        legend_row0.append(torch_image_with_text("Raw Outputs", legend_height, image.shape[-1]))
        legend_row1.append(raw_output_viz[0][1])
        del raw_output_viz

        # Visualize geometry control (white canvas)
        image = torch.cat(
            [torchvision.utils.make_grid(
                torch.cat(
                    [compose_stroke_with_canvas(x[1], mode="stroke_over_white_canvas")
                     for x in y], dim=0), nrow=1, padding=0)
                for y in raw_results], dim=-1).cpu()
        images.append(_add_alpha(image))
        legend_row0.append(torch_image_with_text("Geometry control (white canvas)", legend_height, image.shape[-1]))
        legend_row1.append(torchvision.utils.make_grid(
            self.curated_geom_images, nrow=self.curated_geom_images.shape[0], padding=0))

        # Visualize color control for primary color (white canvas)
        image = torch.cat(
            [torchvision.utils.make_grid(
                torch.cat([compose_stroke_with_canvas(
                    x[1],
                    primary_colors=self.curated_primary_colors[i, :].unsqueeze(0).expand(x[1]['colors'].shape[0], -1),
                    mode="stroke_over_white_canvas")
                           for x in raw_results[0]], dim=0), nrow=1, padding=0)
                for i in range(self.curated_primary_colors.shape[0])], dim=-1).cpu()
        images.append(_add_alpha(image))
        legend_row0.append(torch_image_with_text("Color control (white canvas)", legend_height, image.shape[-1]))
        legend_row1.extend([_color_legend_image(raw_results[0][0][0].shape[-1], self.curated_primary_colors[i, :])
                            for i in range(self.curated_primary_colors.shape[0])])

        image = torch.cat(images, dim=2)
        legend_row0 = torch.cat(legend_row0, dim=2)
        legend_row1 = torch.cat(legend_row1, dim=2)

        image = ((torch.cat([_add_alpha(legend_row0), _add_alpha(legend_row1), image], dim=1).
                  permute(1, 2, 0) / 2.0 + 0.5) * 255).clip(0, 255).to(torch.uint8).numpy()

        imsave(_fname_png('debug_control'), image)

        self.visualize_stitching(G_ema, _fname_png('stitching'))


