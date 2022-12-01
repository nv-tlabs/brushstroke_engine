# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import logging
import torch
from forger.viz.bundled import load_bundled_geometry_image
from forger.util.logging import log_tensor

logger = logging.getLogger(__name__)


class StyleUVSMapper(object):
    def __init__(self, engine=None):
        self.sfactors = {}
        self.engine = engine
        self.geom_feature = None  # geometry feature used for mapping
        self.bmask = None  # background mask
        self.fmask = None  # foreground mask

    def _init_geometry(self):
        assert self.engine is not None

        device = self.engine.device
        width = self.engine.G.img_resolution

        geo_files = ['cross_', 'curve_', 'end_', 'line_', 'curve2_']
        geo_thick_files = ['%srad025.png' % x for x in geo_files]
        geo_med_files = ['%srad016.png' % x for x in geo_files]

        geo_thick = torch.stack([load_bundled_geometry_image(f, width) for f in geo_thick_files]).to(torch.float32).to(
            device) / 255
        geo_med = torch.stack([load_bundled_geometry_image(f, width) for f in geo_med_files]).to(torch.float32).to(
            device) / 255

        geo_input = geo_med[..., 1].unsqueeze(1)
        self.geom_feature = self.engine.encoder.encode(geo_input)
        self.fmask = geo_input < 0.01

        geo_bg = geo_thick[..., 1].unsqueeze(1)
        self.bmask = geo_bg > 0.99

    def map_style(self, brush_opts, uvs, colors):
        sfactor = self.get_sfactor(brush_opts)

        uvs_p = StyleUVSMapper._map_style_s(sfactor, uvs)
        return uvs_p, colors

    @staticmethod
    def _map_style_s(sfactor, uvs):
        U = uvs[:, 0:1, ...]
        V = uvs[:, 1:2, ...]
        S = uvs[:, 2:3, ...]

        Sp = sfactor * S
        Sp[Sp > 1.0] = 1.0

        delta = 1 - Sp
        EPS = 0.000001
        zero_factor = delta <= EPS
        needs_factor = torch.logical_not(zero_factor)

        uvfactor = torch.ones((uvs.shape[0], 1, uvs.shape[-2], uvs.shape[-1]), device=uvs.device, dtype=uvs.dtype)
        uvfactor[zero_factor] = 0
        uvfactor[needs_factor] = delta[needs_factor] / (U + V)[needs_factor]

        Up = uvfactor * U
        Vp = uvfactor * V
        return torch.cat([Up, Vp, Sp], dim=1)

    def _to_color_spec(self, colors):
        colors = ((colors[0, ...].detach().cpu() / 2 + 0.5) * 255).to(torch.uint8).numpy()
        colors = [ list(colors[..., i]) for i in range(3) ]
        colors = ':'.join(['rgb(%s)' % (','.join([str(x) for x in color])) for color in colors])
        return colors

    def _render(self, brush_opts, geo_feature):
        device = self.engine.device

        # TODO: will need to handle positions, once needed
        if brush_opts.style_ws is not None:
            ws = brush_opts.style_ws.expand(geo_feature[0].shape[0], -1, -1).to(device)
            renders, raw = self.engine.G.synthesis(ws, geom_feature=geo_feature, noise_mode='const',
                                                   return_debug_data=True)
        else:
            z = brush_opts.style_z.expand(geo_feature[0].shape[0], -1).to(device)
            renders, raw = self.engine.G(z=z, c=None, geom_feature=geo_feature, noise_mode='const',
                                         return_debug_data=True)
        return renders, raw

    def get_colors_raw(self, brush_opts):
        if self.geom_feature is None:
            self._init_geometry()

        _, raw = self._render(brush_opts, [x[:1, ...] for x in self.geom_feature])
        return raw['colors']

    def get_colors(self, brush_opts):
        return self._to_color_spec(self.get_colors_raw(brush_opts))

    def get_brush_icon(self, brush_opts, on_white=True):
        if self.geom_feature is None:
            self._init_geometry()

        logger.info(f'Rendering icon for style {brush_opts.style_id}')

        renders, raw = self._render(brush_opts, [x[:1, ...] for x in self.geom_feature])

        if on_white:
            renders = renders * (1 - raw['uvs'][:, 2:, ...]) + raw['uvs'][:, 2:, ...]

        return ((renders[0, ...].permute(1, 2, 0).detach() / 2 + 0.5) * 255).to(torch.uint8).cpu().numpy()

    def get_sfactor(self, brush_opts):
        style_id = brush_opts.style_id

        if style_id in self.sfactors:
            return self.sfactors[style_id]

        logger.info(f'Computing clear background mapping of style {style_id}')

        if self.geom_feature is None:
            self._init_geometry()

        renders, raw = self._render(brush_opts, self.geom_feature)

        S = raw['uvs'][:, 2:3, ...]
        # val = S[self.bmask].max()
        val = torch.stack([torch.topk(S[i, ...][self.bmask[i, ...]], k=15)[0].min() for i in range(S.shape[0])]).min()
        sfactor = 1 / val

        self.sfactors[style_id] = sfactor
        return sfactor