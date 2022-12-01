# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import copy
#import cv2
import logging
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from skimage.io import imsave, imread

import forger.experimental.autoenc.factory as factory
import thirdparty.stylegan2_ada_pytorch.dnnlib as dnnlib
import thirdparty.stylegan2_ada_pytorch.legacy as legacy
from thirdparty.stylegan2_ada_pytorch.training.networks_modified import Generator

from forger.util.logging import log_tensor

# TODO: this is a circular import which must be fixed
from forger.ui.mapper import StyleUVSMapper
import forger.train.stitching as stitching

logger = logging.getLogger(__name__)


class FeatureCanvas:
    def __init__(self, canvas_height, canvas_width, down_factor):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.down_factor = down_factor
        self.width = int(math.ceil(self.canvas_width / down_factor))
        self.height = int(math.ceil(self.canvas_height / down_factor))

        self.features = None
        self.mask = None

    def _init_canvasses(self, feature_patch):
        """

        @param feature_patch: B x C x H x W
        @return:
        """
        B, C, H, W = feature_patch.shape
        self.features = torch.zeros((1, C, self.height, self.width),
                                    dtype=feature_patch.dtype, device=feature_patch.device)
        self.mask = torch.zeros((self.height, self.width), dtype=torch.bool, device=feature_patch.device)

    def down_area(self, area):
        rows = area.rend - area.rstart
        cols = area.cend - area.cstart
        if rows % self.down_factor != 0 or cols % self.down_factor != 0 or \
            area.rstart % self.down_factor != 0 or area.cstart % self.down_factor != 0:
            pass
            # HACK
            logger.warning(f'Area {area} not divisible by {self.down_factor} in feature canvas')

        return stitching.CropHelper.make_area(
            area.rstart // self.down_factor,
            area.cstart // self.down_factor,
            rows // self.down_factor,
            cols // self.down_factor)

    def get_features(self, area):
        """

        @param area:
        @return: mask, features or None, None
        """
        if self.mask is None:
            return None, None

        return self.mask[..., area.rstart:area.rend, area.cstart:area.cend], \
               self.features[..., area.rstart:area.rend, area.cstart:area.cend]

    def set_features(self, area, feature_patch, update_mask=None):
        if self.features is None:
            self._init_canvasses(feature_patch)

        if update_mask is None:
            self.mask[..., area.rstart:area.rend, area.cstart:area.cend] = True
            self.features[..., area.rstart:area.rend, area.cstart:area.cend] = feature_patch
        else:
            self.mask[..., area.rstart:area.rend, area.cstart:area.cend][update_mask] = True
            update_mask = update_mask.unsqueeze(0).unsqueeze(0).expand(-1, self.features.shape[1], -1, -1)
            self.features[..., area.rstart:area.rend, area.cstart:area.cend][update_mask] = feature_patch[update_mask]


class PaintingHelper:
    """
    Encapsulates rendering/brush utilities. A separate instance is created for every session to server.
    """
    _test_stroke = None

    def __init__(self, paint_engine, style_seed=None, debug_dir=None):
        self.engine = paint_engine
        self.seed_rng = np.random.default_rng(seed=style_seed)
        self.brush_options = GanBrushOptions()
        self.brush_options.set_style(*self.random_brush_style())

        self.debug_dir = debug_dir
        self.render_id = 0
        if self.debug_dir is not None:
            os.makedirs(self.debug_dir, exist_ok=True)

        # Server-side canvas
        self.geom_canvas = None
        self.feature_canvas = None
        self.feature_blending_level = 0  # 0 - disabled, 1 - output resolution, 2 - output_resolution / 2 etc...
        self.feature_blending_margin = 16

    def make_new_canvas(self, rows, cols, feature_blending=None):
        self.geom_canvas = torch.ones((rows, cols), dtype=torch.float32, device=self.engine.device)
        logger.info(f'Requesting new canvas {rows}x{cols}')
        if feature_blending is None:
            self.set_feature_blending(self.feature_blending_level)
        else:
            self.set_feature_blending(feature_blending)

    def set_feature_blending(self, feature_blending_level=0):
        down_factor = 2 ** (feature_blending_level - 1)
        logger.info(f'Setting feature blending level to {feature_blending_level}, down factor {down_factor}')
        self.feature_blending_level = feature_blending_level
        if feature_blending_level > 0:
            self.feature_canvas = FeatureCanvas(self.geom_canvas.shape[-2], self.geom_canvas.shape[-1],
                                                down_factor=down_factor)
        else:
            self.feature_canvas = None

    def set_new_brush(self, seed=None):
        style_z, seed = self.random_brush_style(seed)
        self.brush_options.set_style(style_z, seed)

        # TODO: also return rendered sample
        return seed

    def set_render_mode(self, mode=None):
        self.engine.set_render_mode(mode)

    def generate_style_seed(self):
        return self.seed_rng.integers(low=0, high=10000, size=1)[0]

    def random_brush_style(self, seed=None):
        if seed is None:
            seed = self.generate_style_seed()
        style_z = self.engine.random_style(seed)

        return style_z, seed

    def default_brush_options(self):
        return copy.copy(self.brush_options)  # Shallow is ok for this use case

    def generate_dirty_area_alpha(self, dirty_area, width, margin, crop_margin=0):
        if dirty_area.min_width == width:
            dirty_area = stitching.CropHelper.make_area(
                margin + crop_margin, margin + crop_margin,
                width - margin * 2 - crop_margin * 2,
                width - margin * 2 - crop_margin * 2)

        x = torch.linspace(0, width - 1, steps=width).to(self.engine.device)
        grid_y, grid_x = torch.meshgrid(x, x)

        dist_sq_x = torch.min(torch.pow(grid_x - dirty_area.cstart, 2), torch.pow(grid_x - dirty_area.cend + 1, 2))
        dist_sq_y = torch.min(torch.pow(grid_y - dirty_area.rstart, 2), torch.pow(grid_y - dirty_area.rend + 1, 2))

        # Use distance to corners in the corners
        dist_sq = dist_sq_x + dist_sq_y
        dist_sq[0:dirty_area.rstart, dirty_area.cstart:dirty_area.cend] = \
            dist_sq_y[0:dirty_area.rstart, dirty_area.cstart:dirty_area.cend]
        dist_sq[dirty_area.rend:, dirty_area.cstart:dirty_area.cend] = \
            dist_sq_y[dirty_area.rend:, dirty_area.cstart:dirty_area.cend]
        dist_sq[dirty_area.rstart:dirty_area.rend, 0:dirty_area.cstart] = \
            dist_sq_x[dirty_area.rstart:dirty_area.rend, 0:dirty_area.cstart]
        dist_sq[dirty_area.rstart:dirty_area.rend, dirty_area.cend:] = \
            dist_sq_x[dirty_area.rstart:dirty_area.rend, dirty_area.cend:]
        dist = torch.sqrt(dist_sq)

        result = 1 - dist / margin
        result[result < 0] = 0
        result[dirty_area.rstart:dirty_area.rend, dirty_area.cstart:dirty_area.cend] = 1
        return result.to(self.engine.device)

    # TODO: can also do style blending using this
    def _get_blended_features(self, feature_canvas, dirty_area, gen_area, crop_margin):
        blend_margin = self.feature_blending_margin // feature_canvas.down_factor
        crop_margin = crop_margin // feature_canvas.down_factor
        blending_resolution = int(self.engine.patch_width // (2 ** (self.feature_blending_level - 1)))

        # Compute update area: hasfeatures or dirty; remove margin from features too
        update_mask = torch.zeros((blending_resolution, blending_resolution),
                                  dtype=torch.bool, device=self.engine.device)

        # Convert areas to the feature canvas scale (TODO: must make divisible by 2, 4, etc...)
        dirty_area_sc = feature_canvas.down_area(dirty_area)
        gen_area_sc = feature_canvas.down_area(gen_area)

        # Compute alpha as a function of distance from edge:
        #    within dirty area = 1
        #    gradually expand out from dirty area
        relative_dirty_area = stitching.CropHelper.make_area_relative(dirty_area_sc, gen_area_sc)
        alpha = self.generate_dirty_area_alpha(relative_dirty_area, gen_area_sc.min_width,
                                               margin=blend_margin, crop_margin=crop_margin)
        update_mask[alpha > 0.99] = True

        # Get features
        mask, features = feature_canvas.get_features(gen_area_sc)
        if mask is not None:
            update_mask[torch.logical_and(mask, alpha > 0)] = True  # where have saved features and blending
            alpha[torch.logical_not(mask)] = 1  # where have no saved features = 1
            alpha = 1 - alpha
            features = stitching.BlendedFeatures(features, alpha.unsqueeze(0).unsqueeze(0))
        else:
            features = None

        if crop_margin > 0:
            update_mask[:crop_margin, :] = False
            update_mask[-crop_margin:, :] = False
            update_mask[:, :crop_margin] = False
            update_mask[:, -crop_margin:] = False

        return blending_resolution, features, update_mask

    def get_blended_features(self, dirty_area, gen_area, crop_margin):
        if self.feature_canvas is not None:
            blending_resolution, blended_features, update_mask = self._get_blended_features(
                self.feature_canvas, dirty_area, gen_area, crop_margin)
            if blended_features is not None:
                return [blending_resolution], {blending_resolution: blended_features}, update_mask
            else:
                return [blending_resolution], {}, update_mask
        return [], {}, None

    def update_blended_features(self, blended_resolutions, raw_net_output, gen_area, update_mask=None):
        if self.feature_canvas is not None:
            gen_area_sc = self.feature_canvas.down_area(gen_area)
            self.feature_canvas.set_features(gen_area_sc, raw_net_output['features%d' % blended_resolutions[0]], update_mask)

    def render_stroke(self, stroke_patch, canvas_patch, opts, meta=None):
        H, W, C = stroke_patch.shape
        dirty_area = None
        gen_area = stitching.CropHelper.make_area(0, 0, H, W)
        crop_margin = 0
        if meta is not None:
            x = int(meta.get('x'))
            y = int(meta.get('y'))

            if self.feature_canvas is not None:
                # TODO: better to round
                if (x % self.feature_canvas.down_factor) != 0:
                    x = (x // self.feature_canvas.down_factor) * self.feature_canvas.down_factor
                if (y % self.feature_canvas.down_factor) != 0:
                    y = (y // self.feature_canvas.down_factor) * self.feature_canvas.down_factor
            # if x % 2 != 0:
            #     x = x + 1
            # if y % 2 != 0:
            #     y = y + 1

            # Create a dirty area out of meta
            dirty_area = stitching.CropHelper.make_area(y, x, H, W)
            gen_area = stitching.CropHelper.make_area(y, x, H, W)  # Same by default

            if 'crop_margin' in meta:
                crop_margin = int(meta.get('crop_margin'))

        # Get the geometry patch from canvas and update geometry canvas if needed
        geo_patch = self.engine.prepare_geom_input(stroke_patch).squeeze(0).squeeze(0)
        if W != self.engine.patch_width or H != self.engine.patch_width:
            raise RuntimeError('Not implemented')
            assert self.geom_canvas is not None, 'Must call make_new_canvas before generating on smaller patches'
            assert dirty_area is not None, 'Must provide x,y coordinates for partial geometry input'
            assert W <= self.engine.patch_width
            assert H <= self.engine.patch_width

            # Update geometry with dirty patch
            self.geom_canvas[dirty_area.rstart:dirty_area.rend, dirty_area.cstart:dirty_area.cend] = geo_patch

            # TODO: some subtlety; we don't want to always save end of stroke features
            # Expand dirty patch to allow sufficient context
            dirty_area = stitching.CropHelper.pad_area_bounded(
                dirty_area, margin=self.feature_blending_margin, max_dim=self.engine.patch_width)

            # Determine generator area
            canvas_rows, canvas_cols = self.geom_canvas.shape[-2], self.geom_canvas.shape[-1]
            dirty_area = stitching.CropHelper.clip_area(dirty_area, canvas_rows, canvas_cols)
            gen_area = stitching.CropHelper.expand_area(
                dirty_area, self.engine.patch_width, canvas_rows, canvas_cols)

            # Get geometry in the larger patch
            geo_patch = self.geom_canvas[gen_area.rstart:gen_area.rend, gen_area.cstart:gen_area.cend]
        geo_patch = geo_patch.unsqueeze(0).unsqueeze(0)

        # TODO: Get blended features
        generator_kwargs = {}
        blended_resolutions = []
        feature_update_mask = None
        if self.feature_blending_level > 0:
            assert dirty_area is not None
            blended_resolutions, blended_features, feature_update_mask = \
                self.get_blended_features(dirty_area, gen_area, crop_margin)

            generator_kwargs["blended_features"] = blended_features  # causes weirdness
            generator_kwargs["return_features"] = blended_resolutions

        # Evaluate the generator
        img, raw_net_output, debug_img = self.engine._render_stroke_torch(
            geo_patch, canvas_patch, opts, **generator_kwargs)
        log_tensor(img, 'raw rendered result', logger, print_stats=True)

        # HACK
        save_didactic = debug_img is not None and self.debug_dir is not None and \
                        self.feature_canvas is not None and self.feature_canvas.mask is not None
        if save_didactic:
            did_dir = os.path.join(self.debug_dir, 'did%d' % self.render_id)
            os.makedirs(did_dir, exist_ok=True)
            imsave(did_dir + '/geo.png', (geo_patch.squeeze() * 255).to(torch.uint8).detach().cpu().numpy())
            min_feat = self.feature_canvas.features.min()
            den = self.feature_canvas.features.max() - min_feat
            imsave(did_dir + '/feat_all_mask.png',
                   ((self.feature_canvas.mask).to(torch.uint8) * 255).detach().cpu().numpy())
            imsave(did_dir + '/feat_all.png',
                   ((self.feature_canvas.features[0, 3:6, ...] - min_feat) / den * 255).clip(0, 255).permute(1,2,0).to(torch.uint8).detach().cpu().numpy())
            # imsave(did_dir + '/feat_all_clear.png',
            #        ((self.feature_canvas.features[0, 3:7, ...] - min_feat) / den* 255).clip(0, 255).permute(1,2,0).to(torch.uint8).detach().cpu().numpy())
            g_key = list(generator_kwargs["blended_features"].keys())[0]
            imsave(did_dir + '/feat_in_mask.png',
                   (generator_kwargs["blended_features"][g_key].alpha.squeeze() * 255).to(torch.uint8).detach().cpu().numpy())
            imsave(did_dir + '/feat_in.png',
                   ((generator_kwargs["blended_features"][g_key].features[0, 3:6, ...] - min_feat) / den* 255).clip(0, 255).permute(1,2,0).to(torch.uint8).detach().cpu().numpy())
            # imsave(did_dir + '/feat_in_clear.png',
            #        ((generator_kwargs["blended_features"][g_key].features[0, 3:7, ...] - min_feat) / den* 255).clip(0, 255).permute(1,2,0).to(
            #            torch.uint8).detach().cpu().numpy())
            imsave(did_dir + '/feat_update_mask.png',
                   (feature_update_mask.to(torch.uint8) * 255).detach().cpu().numpy())
            imsave(did_dir + '/feat_update.png',
                   ((raw_net_output['features%d' % g_key][0, 3:6, ...] - min_feat) / den* 255).clip(0, 255).permute(1,2,0).to(torch.uint8).detach().cpu().numpy())
            imsave(did_dir + '/feat_preblend.png',
                   ((raw_net_output['features%d_preblend' % g_key][0, 3:6, ...] - min_feat) / den * 255).clip(0, 255).permute(1, 2, 0).to(
                       torch.uint8).detach().cpu().numpy())
            imsave(did_dir + '/result.png',
                   (img.detach().squeeze(0).permute(1, 2, 0) * 255).cpu().clip(0, 255).to(torch.uint8).numpy())
            print(f'X,Y; {x}, {y}  dirty area: {dirty_area.cstart}, {dirty_area.rstart}')

        # HACK:
        # if "blended_features" in generator_kwargs and 128 in generator_kwargs["blended_features"]:
        #     img = torch.stack([generator_kwargs["blended_features"][128].alpha.squeeze(),
        #                        generator_kwargs["blended_features"][128].alpha.squeeze(),
        #                        generator_kwargs["blended_features"][128].alpha.squeeze(),
        #                        torch.ones_like(generator_kwargs["blended_features"][128].alpha.squeeze())]).unsqueeze(0)
        #     img[:, 1:2, 0, :] = 0
        #     img[:, 1:2, -1, :] = 0
        #     img[:, 1:2, :, 0] = 0
        #     img[:, 1:2, 0, -1] = 0
        #     img[:, 0, 0, :] = 1
        #     img[:, 0, -1, :] = 1
        #     img[:, 0, :, 0] = 1
        #     img[:, 0, 0, -1] = 1

        # Update feature canvas
        # TODO: No, should not update the whole area, but cut out margin
        self.update_blended_features(blended_resolutions, raw_net_output, gen_area, feature_update_mask)

        # Apply margin on the server side to avoid network overhead
        gen_area = stitching.CropHelper.offset_area(gen_area, crop_margin)
        img_area = stitching.CropHelper.offset_area(
            stitching.CropHelper.make_area(0, 0, self.engine.patch_width, self.engine.patch_width), crop_margin)
        if crop_margin > 0:
            img = img[..., img_area.rstart:img_area.rend, img_area.cstart:img_area.cend]
        out_meta = {'x': gen_area.cstart, 'y': gen_area.rstart}

        # Postprocess image for encoding
        img = (img.detach().squeeze(0).permute(1, 2, 0) * 255).cpu().clip(0, 255).to(torch.uint8).numpy()
        log_tensor(img, 'final rendered result', logger, print_stats=True)
        img = np.ascontiguousarray(img)

        if save_didactic:
            imsave(did_dir + '/AFTER_feat_all_mask.png',
                   ((self.feature_canvas.mask).to(torch.uint8) * 255).detach().cpu().numpy())
            imsave(did_dir + '/AFTER_feat_all.png',
                   ((self.feature_canvas.features[0, 3:6, ...] - min_feat) / den* 255).clip(0, 255).permute(1, 2, 0).to(
                       torch.uint8).detach().cpu().numpy())
            # imsave(did_dir + '/AFTER_feat_all_clear.png',
            #        ((self.feature_canvas.features[0, 3:7, ...] - min_feat) / den* 255).clip(0, 255).permute(1, 2, 0).to(
            #            torch.uint8).detach().cpu().numpy())
            imsave(did_dir + '/result_crop.png', img)

        if debug_img is not None and self.debug_dir is not None:
            fpath = os.path.join(self.debug_dir, 'debug_render%02d.png' % (self.render_id % 100))  # save at most 100
            logger.debug('Saving debug image to: %s' % fpath)
            imsave(fpath, debug_img)
            self.render_id = self.render_id + 1

        return img, debug_img, out_meta

    @staticmethod
    def test_stroke():
        # TODO: not thread safe
        if PaintingHelper._test_stroke is None:
            image_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../images")
            spline_path = os.path.join(image_dir, "synthesized_stroke.png")
            PaintingHelper._test_stroke = imread(spline_path)
        return PaintingHelper._test_stroke


class GanBrushOptions(object):
    """
    Encapsulates options for a specific brush the paint engine can render.
    """
    def __init__(self,
                 primary_color=None,
                 secondary_color=None,
                 debug=False):
        self.color0 = primary_color  # [3] or [N x 3]
        self.color1 = secondary_color  # [3] or [N x 3]
        self.canvas_color = None  #torch.ones((3,), dtype=torch.float32)  # [3] or [N x 3]
        self.style_z = None
        self.style_id = None
        self.library_id = ''
        self.style_ws = None
        self.opacity = 1.0
        self.debug = debug
        self.position = None  # y, x  (1 x 2)
        self.custom_args = {}
        self.enable_uvs_mapping = False

    def to(self, device):
        if self.style_z is not None:
            self.style_z = self.style_z.to(device)

        if self.style_ws is not None:
            self.style_ws = self.style_ws.to(device)

        if 'noise_buffers' in self.custom_args:
            for k, v in self.custom_args['noise_buffers'].items():
                if not torch.is_tensor(v):
                    v = torch.from_numpy(v)
                self.custom_args['noise_buffers'][k] = v.to(device)
        return self

    def set_position(self, x, y):
        if type(x) is int:
            self.position = torch.tensor([y, x], dtype=torch.int64).unsqueeze(0)
        else:
            self.position = torch.stack([y, x], dim=1)

    def get_position(self, device):
        if self.position is None:
            return None
        else:
            return self.position.to(device)

    def set_color(self, color_idx, in_color):
        """
        @param color_idx: 0 - primary, 1 - secondary, 2 - canvas
        @param in_color: [3] or [Bx3] numpy or torch uint8 [0..255] or float32 [0..1] tensor
        @return:
        """
        def _prepcolor(x):
            if x is None:
                return None
            color = x
            if not torch.is_tensor(color):
                color = torch.from_numpy(color)
            if color.dtype == torch.uint8:
                color = color.to(torch.float32) / 255
            else:
                color = color.to(torch.float32)
            if len(color.shape) == 1:
                color = color.unsqueeze(0)
            return color
        if color_idx == 0:
            self.color0 = _prepcolor(in_color)
        elif color_idx == 1:
            self.color1 = _prepcolor(in_color)
        elif color_idx == 2:
            self.canvas_color = _prepcolor(in_color)
        else:
            logger.error(f'Wring color idx {color_idx}')

    def set_style(self, style_z, style_id=None):
        self.style_z = style_z
        self.style_id = style_id
        self.style_ws = None

    def set_style_w(self, style_w, style_id=None, custom_args=None):
        self.style_ws = style_w  # TODO: fix if not ws, but w
        self.style_id = style_id
        self.style_z = None
        if custom_args is not None:
            self.custom_args = custom_args
        else:
            self.custom_args = {}

    # TODO: this is poorly designed; figure out better way to deal with batches
    def prepare_style(self, batch_size, device):
        def _prep(x):
            if x is None:
                return None
            if x.shape[0] != batch_size:
                assert x.shape[0] == 1, 'Brush options must either have correct style batch, or batch of 1'
                if len(x.shape) == 2:
                    return x.expand(batch_size, -1).to(device)
                else:
                    return x.expand(batch_size, -1, -1).to(device)
            return x.to(device)
        self.style_z = _prep(self.style_z)
        self.style_ws = _prep(self.style_ws)

    def prepare_colors(self, default_colors):
        """

        @param default_colors: B x 3 x ncolors float torch tensor [0..1]
        @return: B x 3 x ncolors float torch tensor [0..1]
        """
        out_colors = default_colors.clone()
        if self.color0 is not None:
            out_colors[:, :, 0] = self.color0
        if self.color1 is not None:
            out_colors[:, :, 1] = self.color1
        if self.canvas_color is not None:
            out_colors[:, :, 2] = self.canvas_color
        return out_colors


class PaintEngine:
    """
    Base interface for paint engine capable of rendering strokes.
    """
    def __init__(self,
                 device=None):
        self.device = device if device is not None else torch.device('cpu')
        self.patch_width = 0

    def render_stroke(self, stroke_patch, canvas_patch, opts, **generator_kwargs):
        """
        opts: GanBrushOptions
        """
        raise NotImplementedError

    def random_style(self, seed):
        return None

    def summary(self):
        raise NotImplementedError


class PaintEngineFactory(object):
    @staticmethod
    def create(gan_checkpoint, device, encoder_checkpoint=None):
        if gan_checkpoint is None:
            logger.warning('Creating MockPaintEngine')
            return MockPaintEngine(256)
        else:
            gan_dir = os.path.dirname(gan_checkpoint)

            encoder = None
            color_format = 'triad'
            geom_inject_resolutions = [0]
            requires_legacy_geom = False
            try:
                # New pkl format
                with dnnlib.util.open_url(gan_checkpoint) as f:
                    pkl = legacy.load_network_pkl(f)
                    color_format = pkl['args'].color_format
                    if hasattr(pkl['args'], 'geom_inject_resolutions'):
                        geom_inject_resolutions = pkl['args'].geom_inject_resolutions
                    else:
                        requires_legacy_geom = True
                    if 'encoder' in pkl:
                        encoder = factory.create_autoencoder_from_checkpoint(loaded_encoder_checkpt=pkl['encoder'])
            except Exception as e:
                # If the checkpoint is saved in the old format, use .json file instead
                logger.warning(f'Failed to parse pickle: {e}')
                gan_opt_path = os.path.join(gan_dir, 'training_options.json')
                try:
                    with open(gan_opt_path, "r") as f:
                        gan_opt = json.load(f)
                        if 'color_format' in gan_opt['loss_kwargs'].keys():
                            color_format = gan_opt['loss_kwargs']['color_format']
                except Exception:
                    logger.warning(f'Failed to determine color format from {gan_opt_path}; falling back to {color_format}')
            logger.info(f'The color format is {color_format}')

            if encoder is None:
                assert encoder_checkpoint is not None, f'Provide geometry encoder; no encoder in GAN checkpoint {gan_checkpoint}'
                encoder = factory.create_autoencoder_from_checkpoint(encoder_checkpoint)
            encoder.set_default_encode_resolutions(geom_inject_resolutions)

            if color_format == 'triad':
                engine = TriadGanPaintEngine(encoder, gan_checkpoint, device, encoder_checkpoint=encoder_checkpoint)
            elif color_format == 'canvas':
                engine = CanvasPaintEngine(encoder, gan_checkpoint, device, encoder_checkpoint=encoder_checkpoint)
            else:
                raise RuntimeError('Unknown color_format found in training_options.json')

            if requires_legacy_geom:
                engine.legacy_requires_non_list_geom()

            return engine


class GanPaintEngine(PaintEngine):
    """
    The base class that renders the strokes with a deep model.
    Every model assumes a geometry encoder and a gan model. The low-level compositing
    is handled by subclasses.
    """
    def __init__(self,
                 encoder,
                 gan_checkpoint,
                 device,
                 encoder_checkpoint=''):
        super().__init__(device)
        self.encoder_checkpoint = encoder_checkpoint
        self.gan_checkpoint = gan_checkpoint
        self.render_modes = {'clear',  # Use only stroke (UVS weight map) and estimate alpha
                             'full'    # Complete GAN output (opaque)
                             }
        self.render_mode = 'clear'
        self.encoder = encoder
        self.encoder.eval().requires_grad_(False).to(self.device)

        logger.info("Loading StyleGAN2 model at " + gan_checkpoint)
        # Using the functions from the StyleGAN code base TODO: Try to migrate to torch.load
        with dnnlib.util.open_url(gan_checkpoint) as fp:
            self.G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(self.device)
            self.G = Generator.create_from_unpickled(self.G)  # Ensure has latest features
        self.patch_width = self.G.img_resolution

        # TODO(mshugrina): can this be None?
        self.style_c = torch.from_numpy(np.zeros((0, self.G.c_dim))).to(self.device)

        # Debug image default parameters
        self.debug_cfg = {
            'margin': 5,
            'color_width': 20
        }
        self.legacy_geom = False

        self.uvs_mapper = StyleUVSMapper(self)

    def legacy_preproc_geom(self, geom):
        if self.legacy_geom:
            return geom[0]
        return geom

    def legacy_requires_non_list_geom(self):
        """
        Support pkls where the geom injection is not a list.
        @return:
        """
        self.legacy_geom = True

    def set_render_mode(self, mode):
        if mode not in self.render_modes:
            raise RuntimeError(f"Render mode should be one of {self.render_modes}")
        self.render_mode = mode

    def summary(self):
        return "{} GAN: {} encoder: {}".format(type(self), self.gan_checkpoint, self.encoder_checkpoint)

    def random_style(self, seed):
        #TODO(aliceli): RandomState is deprecated, replace
        style_z = torch.from_numpy(np.random.RandomState(seed=seed).randn(1, self.G.z_dim)).to(self.device)
        return style_z

    def prepare_geom_input(self, stroke_patch):
        """

        @param stroke_patch: W x W x 4 or W x W x 1 uint8 numpy array with stroke guidance (opaque 255 == FG)
        @return: 1 x 1 x W x W float32 tensor (black 0.0 == FG, 1.0 == BG)
        """
        # TIP(aliceli): if you do 255 - uint8, you may get overflow and wrong result; always convert to float first
        geom_input = 1 - torch.from_numpy(stroke_patch[:, :, -1:]).to(torch.float32).permute(2, 0, 1) / 255.0
        geom_input = geom_input.unsqueeze(0).to(self.device)
        return geom_input

    def render_stroke(self, stroke_patch, canvas_patch, opts, **generator_kwargs):
        """

        @param stroke_patch: W x W x 4 or W x W x 1 uint8 numpy array with stroke guidance (black and opaque == FG)
        @param canvas_patch: W x W x 4 uint8 numpy array
        @param opts: GanBrushOptions instance

        @return: W x W x 4 np uint8 array, debug image
        """
        geom_input = self.prepare_geom_input(stroke_patch)
        assert geom_input.shape[1] == self.patch_width and geom_input.shape[2] == self.patch_width

        res, raw_net_output, debug_img = self._render_stroke_torch(geom_input, canvas_patch, opts, **generator_kwargs)
        log_tensor(res, 'raw rendered result', logger, print_stats=True)
        res = (res.detach().squeeze(0).permute(1, 2, 0) * 255).cpu().clip(0, 255).to(torch.uint8).numpy()
        log_tensor(res, 'final rendered result', logger, print_stats=True)
        return np.ascontiguousarray(res), debug_img

    def _render_stroke_torch(self, geom, canvas, opts, **generator_kwargs):
        """
        Renders stroke, given prepared torch input and options.

        @param geom: B x 1 x W x W torch float32 tensor [0..1] with geometry guidance (1 == background, 0 == fg)
        @param canvas: TBD
        @param opts: GanBrushOptions
        @return:  (torch_render, raw_net_output, debug_img) where
             torch_render: B x 4 x W x W torch float array [0..1]
             raw_net_output: custom dictionary
             debug_img: None (if not opts.debug) or W x verywide x 4 numpy uint8 contiguous array
        """
        raise NotImplementedError()


def _touint8np(torch_img):
    return (torch_img.detach() * 255).cpu().to(torch.uint8).numpy()


class TriadGanPaintEngine(GanPaintEngine):
    """
    GAN-based paint engine that renders strokes using original UVS color traid formulation.
    """
    def __init__(self,
                 encoder,
                 gan_checkpoint,
                 device,
                 encoder_checkpoint=''):
        super().__init__(encoder, gan_checkpoint, device, encoder_checkpoint=encoder_checkpoint)

    def _render_stroke_torch(self, geom, canvas, opts, **generator_kwargs):
        """
        Renders stroke, given prepared torch input and options.

        @param geom: B x 1 x W x W torch float32 tensor [0..1] with geometry guidance (1 == background, 0 == fg)
        @param canvas: TBD
        @param opts: GanBrushOptions
        @return:  (torch_render, raw_net_output, debug_img) where
             torch_render: B x 4 x W x W torch float array [0..1]
             raw_net_output: custom dictionary
             debug_img: None (if not opts.debug) or W x verywide x 4 numpy uint8 contiguous array
        """
        geom_feature = self.encoder.encode(geom)
        opts.to(self.device)

        if opts.style_ws is not None:
            result_img, triad_data = self.G.forward_pre_mapped(
                ws=opts.style_ws,
                positions=opts.get_position(geom.device),
                geom_feature=self.legacy_preproc_geom(geom_feature),
                return_debug_data=True,
                noise_mode='const',
                **opts.custom_args,
                **generator_kwargs)
        else:
            result_img, triad_data = self.G(z=opts.style_z,
                                            c=self.style_c,
                                            positions=opts.get_position(geom.device),
                                            geom_feature=self.legacy_preproc_geom(geom_feature),
                                            return_debug_data=True,
                                            noise_mode='const', **generator_kwargs)

        # B x ncolors x W x W
        uvs = triad_data['uvs']
        log_tensor(uvs, 'uvs', logger, print_stats=True)
        #log_tensor(uvs[:, 2, ...], 'uvs BG', logger, print_stats=True)
        #uvs = self._apply_uv_normalization_hack(uvs)

        # B x C x ncolors
        default_colors = (triad_data['colors'] + 1) / 2.0   # correspond to u, v, s
        log_tensor(default_colors, 'default_colors', logger, print_stats=True)

        if opts.enable_uvs_mapping:
            uvs, default_colors = self.uvs_mapper.map_style(opts, uvs, default_colors)

        # If opts don't have color set, we use other colors
        colors = opts.prepare_colors(default_colors)

        # (B x 1 x ncolors x W x W) * (B x C x ncolors x 1 x 1)  = B x C x ncolors x W x W
        stroke_tmp = uvs.unsqueeze(1) * colors.unsqueeze(-1).unsqueeze(-1)
        stroke = torch.sum(stroke_tmp, dim=2)

        # Assumption: canvas alpha/color (S) can be ignored # TODO: verify
        if self.render_mode == 'clear':
            result = torch.cat([stroke,     # B x C x W x W
                                torch.sum(uvs[:, 0:2, ...], dim=1, keepdim=True)],  # B x 1 x W x W
                               dim=1)
        elif self.render_mode == 'full':
            result = torch.cat([stroke,
                                torch.ones_like(stroke_tmp[:, :1, 0, ...])], dim=1)
        else:
            raise RuntimeError('Unknown render mode for TriadGanPaintEngine: {}'.format(self.render_mode))

        debug_img = None
        if opts.debug:
            # TODO: there could be a normalization issue, as stylegan outputs RGB values directly, no enforced normalization
            decoded_geom = self.encoder(self.encoder.preprocess(geom))
            log_tensor(decoded_geom, 'raw decoded geom', logger, print_stats=True)
            decoded_geom = self.encoder.postprocess(decoded_geom)
            #decoded_geom = torch.sigmoid(decoded_geom)
            intermediate_res = stroke
            debug_img = self._make_debug_image(canvas, geom, decoded_geom, uvs, default_colors, colors, result_img, intermediate_res, result)
            debug_img = np.ascontiguousarray(debug_img)
            #imsave('/tmp/t.png', debug_img)
        return result, triad_data, debug_img

    def _make_debug_image(self, canvas, geom, decoded_geom, uvs, default_colors, colors, gan_result, intermediate_user_result, alpha_user_result):
        """
        Generates a debug image laid out as follows:
        A B C D
        where:
        A) current_canvas : input_stroke : autoencoded_stroke
        B) gan_color0 : u : gan_color1 : v : gan_color2 : s
        C) user_color0 : user_color1 : user_color2
        D) original GAN output : stroke rendered with user colors : final_opacity_stroke_with_any_postprocessing

        @param canvas: W x W x 4 uint8 numpy array [0...255]
        @param geom: conditioning spline (B x 1 x W x W float torch tensor [0..1])
        @param decoded_geom: geometry autoencoder reconstruction (B x 1 x W x W float torch tensor [0..1])
        @param uvs: color triad weights output by GAN (B x ncolors x W x W float torch tensor [0..1])
        @param default_colors: RGB colors output by GAN and normalized (B x 3 x ncolors float torch tensor [0..1])
        @param colors: custom RGB colors used to render final stroke (B x 3 x ncolors float torch tensor [0..1])
        @param gan_result: final output produced by original generator code (B x 3 x W x W float torch tensor [-1..1])
        @param intermediate_user_result: same interpolation as gan, but with custom colors (B x 3 x W x W float torch tensor [0..1])
        @param alpha_user_result: final result with alpha channel (B x 4 x W x W float torch tensor [0..1])
        @return: numpy uint8 array W x extra_wide x 4
        """
        # images + colors + margins
        width = 9 * self.patch_width + self.debug_cfg['color_width'] * 6 + self.debug_cfg['margin'] * (9 + 6)
        result = np.zeros((self.patch_width, width, 4), dtype=np.uint8)

        # canvas
        pw = self.patch_width
        if canvas is not None:
            result[:, 0:pw, :] = canvas
        wstart = pw + self.debug_cfg['margin']

        # input spline
        result[:, wstart:wstart + pw, 0:3] = _touint8np(geom[0, 0, ...].unsqueeze(-1))
        result[:, wstart:wstart + pw, 3] = 255
        wstart = wstart + pw + self.debug_cfg['margin']

        # decoded spline
        result[:, wstart:wstart + pw, 0:3] = _touint8np(decoded_geom[0, 0, ...].unsqueeze(-1))
        result[:, wstart:wstart + pw, 3] = 255
        wstart = wstart + pw + self.debug_cfg['margin']

        # color triad data -------
        for cidx in range(3):
            result[:, wstart:wstart + self.debug_cfg['color_width'], 0:3] = _touint8np(default_colors[0, :, cidx].unsqueeze(0).unsqueeze(0))
            result[:, wstart:wstart + self.debug_cfg['color_width'], 3] = 255
            wstart = wstart + self.debug_cfg['color_width'] + self.debug_cfg['margin']

            result[:, wstart:wstart + pw, 0:3] = _touint8np(uvs[0, cidx, ...].unsqueeze(-1))
            result[:, wstart:wstart + pw, 3] = 255
            wstart = wstart + pw + self.debug_cfg['margin']

        for cidx in range(3):
            result[:, wstart:wstart + self.debug_cfg['color_width'], 0:3] = _touint8np(colors[0, :, cidx].unsqueeze(0).unsqueeze(0))
            result[:, wstart:wstart + self.debug_cfg['color_width'], 3] = 255
            wstart = wstart + self.debug_cfg['color_width'] + self.debug_cfg['margin']

        # Final images -------
        result[:, wstart:wstart + pw, 0:3] = _touint8np((gan_result.squeeze(0).permute(1, 2, 0) + 1) / 2.0)
        result[:, wstart:wstart + pw, 3] = 255
        wstart = wstart + pw + self.debug_cfg['margin']

        result[:, wstart:wstart + pw, 0:3] = _touint8np(intermediate_user_result.squeeze(0).permute(1, 2, 0))
        result[:, wstart:wstart + pw, 3] = 255
        wstart = wstart + pw + self.debug_cfg['margin']

        result[:, wstart:wstart + pw, :] = _touint8np(alpha_user_result.squeeze(0).permute(1, 2, 0))
        wstart = wstart + pw + self.debug_cfg['margin']

        return result


class CanvasPaintEngine(GanPaintEngine):
    """GAN-based paint engine that renders strokes using Canvas + UVS foreground formulation."""

    def __init__(self,
                 encoder,
                 gan_checkpoint,
                 device,
                 encoder_checkpoint=''
                 ):
        super().__init__(encoder, gan_checkpoint, device, encoder_checkpoint=encoder_checkpoint)
        self.render_modes.add('stroke')  # Use stroke (UVS weight map), fully opaque
        self.render_modes.add('canvas')  # Use only canvas

    def _render_stroke_torch(self, geom, canvas, opts):
        """
        Renders stroke, given prepared torch input and options.

        @param geom: B x 1 x W x W torch float32 tensor [0..1] with geometry guidance (1 == background, 0 == fg)
        @param canvas: TBD
        @param opts: GanBrushOptions
        @return:  (torch_render, raw_net_output, debug_img) where
        torch_render: B x 4 x W x W torch float array [0..1]
        raw_net_output: custom dictionary
        debug_img: None (if not opts.debug) or W x verywide x 4 numpy uint8 contiguous array
        """
        geom_feature = self.encoder.encode(geom)
        result_img, canvas_data = self.G(z=opts.style_z,
                                         c=self.style_c,
                                         geom_feature=self.legacy_preproc_geom(geom_feature),
                                         return_debug_data=True,
                                         noise_mode='const')
        # B x ncolors x W x W
        uvs = canvas_data['uvs']
        log_tensor(uvs, 'uvs', logger, print_stats=True)
        # B x C x ncolors
        default_colors = (canvas_data['colors'] + 1) / 2.0  # correspond to u, v, s
        log_tensor(default_colors, 'default_colors', logger, print_stats=True)

        # If opts don't have color set, we use other colors
        colors = opts.prepare_colors(default_colors)

        # (B x 1 x ncolors x W x W) * (B x C x ncolors x 1 x 1)  = B x C x ncolors x W x W
        composed_stroke = uvs.unsqueeze(1) * colors.unsqueeze(-1).unsqueeze(-1)
        stroke_rgb = torch.sum(composed_stroke, dim=2)
        # TODO: Avoid creating this temporary tensor everytime
        default_alpha = torch.ones_like(stroke_rgb[:, :1, ...])  # B x 1 x W x W

        alpha_fg = canvas_data['alpha_fg']
        gen_canvas = canvas_data['canvas']

        if self.render_mode == 'clear':
            result = torch.cat([stroke_rgb,  # B x C x W x W
                                alpha_fg,  # B x 1 x W x W
                                ],
                               dim=1)
        elif self.render_mode == 'stroke':
            result = torch.cat([stroke_rgb,  # B x C x W x W
                                default_alpha
                                ],
                               dim=1)
        elif self.render_mode == 'canvas':
            result = torch.cat([(gen_canvas + 1.0) / 2.0,
                                default_alpha
                                ], dim=1)
        elif self.render_mode == 'full':
            result = torch.cat([
                (1 - alpha_fg) * (gen_canvas + 1.0) / 2.0 + alpha_fg * stroke_rgb,
                default_alpha], dim=1)
        else:
            raise RuntimeError(f'Unknown render mode: {self.render_mode}')

        debug_img = None
        if opts.debug:
            decoded_geom = self.geom_autoencoder(geom)
            log_tensor(decoded_geom, 'raw decoded geom', logger, print_stats=True)
            decoded_geom = torch.sigmoid(decoded_geom)
            intermediate_res = torch.sum(composed_stroke, dim=2)
            debug_img = self._make_debug_image(canvas, geom, decoded_geom, uvs, default_colors, colors,
                                               gen_canvas, canvas_data['alpha_fg'],
                                               result_img, intermediate_res, result)
            debug_img = np.ascontiguousarray(debug_img)

        return result, canvas_data, debug_img

    def _make_debug_image(self,
                          canvas,
                          geom,
                          decoded_geom,
                          uvs,
                          default_colors,
                          colors,
                          rgb_canvas,
                          fg_alpha,
                          gan_result,
                          intermediate_user_result,
                          alpha_user_result):
        """
        Generates a debug image laid out as follows:
        A B C D
        where:
        A) current_canvas : input_stroke : autoencoded_stroke
        B) gan_color0 : u : gan_color1 : v : gan_color2 : s
        C) rgb_canvas : canvas_alpha
        C) user_color0 : user_color1 : user_color2
        D) original GAN output : stroke rendered with user colors over white canvas : final_opacity_stroke_with_any_postprocessing

        @param canvas: W x W x 4 uint8 numpy array [0...255]
        @param geom: conditioning spline (B x 1 x W x W float torch tensor [0..1])
        @param decoded_geom: geometry autoencoder reconstruction (B x 1 x W x W float torch tensor [0..1])
        @param uvs: color triad weights output by GAN (B x ncolors x W x W float torch tensor [0..1])
        @param default_colors: RGB colors output by GAN and normalized (B x 3 x ncolors float torch tensor [0..1])
        @param colors: custom RGB colors used to render final stroke (B x 3 x ncolors float torch tensor [0..1])
        @param rgb_canvas: (B x 3 x W x W) float torch tensor [-1..1]
        @param fg_alpha: (B x 1 x W x W) float torch tensor [0..1]
        @param gan_result: final output produced by original generator code (B x 3 x W x W float torch tensor [-1..1])
        @param intermediate_user_result: same interpolation as gan, but with custom colors (B x 3 x W x W float torch tensor [0..1])
        @param alpha_user_result: final result with alpha channel (B x 4 x W x W float torch tensor [0..1])
        @return: numpy uint8 array W x extra_wide x 4
        """

        n_patch = 11
        n_color = 6
        cwidth =  self.debug_cfg['color_width']
        margin = self.debug_cfg['margin']

        # images + colors + margins
        width = n_patch * self.patch_width + n_color * cwidth + (n_patch + n_color) * margin
        result = np.zeros((self.patch_width, width, 4), dtype=np.uint8)

        # canvas
        pw = self.patch_width
        result[:, 0:pw, :] = canvas
        wstart = pw + self.debug_cfg['margin']

        # input spline
        result[:, wstart:wstart + pw, 0:3] = _touint8np(geom[0, 0, ...].unsqueeze(-1))
        result[:, wstart:wstart + pw, 3] = 255
        wstart = wstart + pw + self.debug_cfg['margin']

        # decoded spline
        result[:, wstart:wstart + pw, 0:3] = _touint8np(decoded_geom[0, 0, ...].unsqueeze(-1))
        result[:, wstart:wstart + pw, 3] = 255
        wstart = wstart + pw + self.debug_cfg['margin']

        # canvas data -------
        for cidx in range(3):
            # GAN color
            result[:, wstart:wstart + self.debug_cfg['margin'], 0:3] = _touint8np(
                default_colors[0, :, cidx].unsqueeze(0).unsqueeze(0))
            result[:, wstart:wstart + self.debug_cfg['margin'], 3] = 255
            wstart = wstart + self.debug_cfg['margin'] + self.debug_cfg['margin']

            result[:, wstart:wstart + pw, 0:3] = _touint8np(uvs[0, cidx, ...].unsqueeze(-1))
            result[:, wstart:wstart + pw, 3] = 255
            wstart = wstart + pw + self.debug_cfg['margin']

        # RGB canvas and alpha
        result[:, wstart:wstart + pw, 0:3] = _touint8np((rgb_canvas[0, ...].permute(1, 2, 0) + 1) / 2.0)
        result[:, wstart:wstart + pw, 3] = 255
        wstart = wstart + pw + margin

        result[:, wstart:wstart + pw, 0:3] = _touint8np(fg_alpha[0, ...].permute(1, 2, 0))
        result[:, wstart:wstart + pw, 3] = 255
        wstart = wstart + pw + margin

        # User color
        for cidx in range(3):
            result[:, wstart:wstart +self.debug_cfg['margin'], 0:3] = _touint8np(
                colors[0, :, cidx].unsqueeze(0).unsqueeze(0))
            result[:, wstart:wstart + self.debug_cfg['margin'], 3] = 255
            wstart = wstart + self.debug_cfg['margin'] + self.debug_cfg['margin']

        # Final images -------
        result[:, wstart:wstart + pw, 0:3] = _touint8np(
            (gan_result.squeeze(0).permute(1, 2, 0) + 1) / 2.0)
        result[:, wstart:wstart + pw, 3] = 255
        wstart = wstart + pw + self.debug_cfg['margin']

        result[:, wstart:wstart + pw, 0:3] = _touint8np(
            intermediate_user_result.squeeze(0).permute(1, 2, 0))
        result[:, wstart:wstart + pw, 3] = 255
        wstart = wstart + pw + self.debug_cfg['margin']

        result[:, wstart:wstart + pw, :] = _touint8np(alpha_user_result.squeeze(0).permute(1, 2, 0))
        wstart = wstart + pw + self.debug_cfg['margin']

        return result


class MockPaintEngine(PaintEngine):
    """ Brush should hold a model to render stroke. """

    def __init__(self, patch_width):
        super().__init__()
        self.patch_width = patch_width

    def render_stroke(self, stroke_patch, canvas_patch, opts):
        """ TBD, right now stroke_patch is N x N x 4 image. """
        # stroke_input = np.abs(255 - stroke_patch[:, :, -1:])
        # plt.imsave("/home/aliceli/Documents/experiment/stylegan/ui/income_patch.jpg", arr=np.tile(stroke_input, (1, 1, 3)))
        result = np.copy(canvas_patch)

        # Mock: just draw a red frame
        # top
        result[:3, :, 0] = 255
        result[:3, :, -1] = 255
        # bottom
        result[-3:, :, 0] = 255
        result[-3:, :, -1] = 255
        # left
        result[:, 0, 0] = 255
        result[:, 0, -1] = 255
        # right
        result[:, -3:, 0] = 255
        result[:, -3:, -1] = 255
        return result, None, None

    def summary(self):
        return 'mock engine'

