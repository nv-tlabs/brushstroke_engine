# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import logging
import numpy as np
import re
import torch
from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import conv2d_resample
from torch_utils.ops import upfirdn2d
from torch_utils.ops import bias_act
from torch_utils.ops import fma

import thirdparty.stylegan2_ada_pytorch.dnnlib as dnnlib
from thirdparty.stylegan2_ada_pytorch.training.networks import SynthesisBlock, MappingNetwork
from forger.train.positional import PositionalEncodingFactory

logger = logging.Logger(__name__)


@persistence.persistent_class
class SynthesisNetwork(torch.nn.Module):
    @staticmethod
    def create_from_unpickled(unpickled_synthesis):
        synthesis = SynthesisNetwork(None, None, None, None, None, None)

        for attr in dir(unpickled_synthesis):
            if attr not in dir(synthesis):
                # Also modify the synthesis blocks
                if re.match(r'b\d+', attr) is not None:
                    setattr(synthesis, attr, SynthesisBlock.create_from_unpickled(getattr(unpickled_synthesis, attr)))
                else:
                    setattr(synthesis, attr, getattr(unpickled_synthesis, attr))
                logger.info(f'Setting attribute {attr}')
        return synthesis

    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        geom_feature_channels,      # THe number of channels in the geometry feature.
        geom_feature_resolutions,    # The resolution of the geometry feature
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        color_w_channels,           # Number of channels in w to use for colors (0 if all)
        enable_geom_linear = False, # First apply a linear layer before injecting geometry
        pos_encoding_channels=0,       # Number of channels in positional encoding
        pos_encoding_feature_resolutions=[],  # Layers to inject
        pos_encoding_injection_mode=None,
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        super().__init__()
        if w_dim is None:
            return  # special case when creating from pkl

        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0

        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.geom_feature_resolutions = geom_feature_resolutions
        self.geom_feature_channels = geom_feature_channels
        self.geom_linear = None
        assert not enable_geom_linear, 'Not implemented'

        self.pos_encoding_channels = pos_encoding_channels
        self.pos_encoding_feature_resolutions = pos_encoding_feature_resolutions
        self.pos_encoding_injection_mode = pos_encoding_injection_mode

        self.num_ws = 0
        for res in self.block_resolutions:
            out_channels = channels_dict[res]
            in_channels = channels_dict[res // 2] if res > 4 else 0
            _geo_channels = 0
            if res // 2 in self.geom_feature_resolutions:
                _idx = self.geom_feature_resolutions.index(res // 2)
                _geo_channels = self.geom_feature_channels[_idx]
                # TODO: fix logging in subprocess func
                print(f'Injecting geometry at resolution {res // 2} : {res} -> {in_channels} channels + {_geo_channels} geo channels : {out_channels}')
                in_channels += _geo_channels

            if res // 2 in self.pos_encoding_feature_resolutions:
                if self.pos_encoding_injection_mode == 'cat':
                    print(f'Injecting (cat) positional encoding at resolution {res // 2} : {res} -> {in_channels} channels + '
                          f'{self.pos_encoding_channels} encoding channels : {out_channels}')
                    in_channels += self.pos_encoding_channels
                elif self.pos_encoding_injection_mode == 'add':
                    print(f'Adding (+) {self.pos_encoding_channels}-channel positional encoding at resolution '
                          f'{res // 2} : {res} -> {in_channels - _geo_channels} channels + {_geo_channels} geo channels : '
                          f'{out_channels}')
                    assert self.pos_encoding_channels in [_geo_channels, in_channels, in_channels - _geo_channels], \
                        'In add (+) mode for positional encoding, encoding channels must match layer channels'
                else:
                    raise RuntimeError(f'Unknown --pos_encoding_injection_mode {self.pos_encoding_injection_mode}')

            use_fp16 = (res >= fp16_resolution)
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels,
                                   out_channels,
                                   w_dim=w_dim, resolution=res,
                                   img_channels=img_channels, is_last=is_last, use_fp16=use_fp16,
                                   color_w_channels=color_w_channels, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def get_last_block(self):
        return getattr(self, f'b{self.block_resolutions[-1]}')

    def forward(self, ws,  geom_feature, pos_encoding=None, return_debug_data=False, return_features=None,
                blended_features=None, noise_buffers=None, **block_kwargs):
        """

        @param ws:
        @param geom_feature:
        @param pos_encoding:
        @param return_debug_data:
        @param return_features: resolutions at which to return features 4, 8, 16...
        @param blended_features: dictionary resoluton : stitching.BlendedFeatures
        @param block_kwargs:
        @return:
        """
        block_ws = []

        if return_features is None:
            return_features = []

        if blended_features is None:
            blended_features = {}

        with torch.autograd.profiler.record_function('split_ws'):
            misc.assert_shape(ws, [None, self.num_ws, self.w_dim])
            ws = ws.to(torch.float32)
            w_idx = 0
            for res in self.block_resolutions:
                block = getattr(self, f'b{res}')
                block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
                w_idx += block.num_conv

        debug_data = {}
        x = img = None
        geo_idx = 0
        pos_idx = 0
        logger.info(f'Blended features: {blended_features}')
        for res, cur_ws in zip(self.block_resolutions, block_ws):
            block_name = f'b{res}'
            block = getattr(self, block_name)
            conv0_noise = None
            conv1_noise = None
            if noise_buffers is not None:
                conv0_noise = noise_buffers.get(f'{block_name}.conv0.noise_const')
                conv1_noise = noise_buffers.get(f'{block_name}.conv1.noise_const')

            if return_debug_data and res == self.block_resolutions[-1]:
                x, img, _debug_data = block(x, img, cur_ws, return_debug_data=True,
                                            conv0_noise=conv0_noise, conv1_noise=conv1_noise, **block_kwargs)
                debug_data.update(_debug_data)
            else:
                x, img = block(x, img, cur_ws, conv0_noise=conv0_noise, conv1_noise=conv1_noise,
                               **block_kwargs)

            # HACK
            if res in return_features:
                debug_data['features%d_preblend' % res] = x

            if res in blended_features:
                logger.info('Blending!!')
                x = blended_features[res].blend(x).to(x.dtype)
                if res == self.block_resolutions[-1]:
                    rgb_res = block.torgb(x, cur_ws[:, -1, :], return_debug_data=True, fused_modconv=True)
                    img = rgb_res[0]
                    debug_data.update(rgb_res[1])

            if res in return_features:
                debug_data['features%d' % res] = x

            block_geom_feature = None
            if res in self.geom_feature_resolutions:
                #if self.geom_linear is not None:
                    #geom_feature = self.geom_linear(geom_feature[geo_idx].permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                block_geom_feature = geom_feature[geo_idx]
                geo_idx += 1

            if hasattr(self, 'pos_encoding_feature_resolutions') and res in self.pos_encoding_feature_resolutions:
                block_pos_encoding = pos_encoding[pos_idx]
                assert block_pos_encoding.shape[1] == self.pos_encoding_channels, \
                    f'Shape {block_pos_encoding.shape}; expected channels {self.pos_encoding_channels}'
                pos_idx += 1

                if self.pos_encoding_injection_mode == 'cat':
                    x = torch.cat([x, block_pos_encoding], dim=1)
                elif self.pos_encoding_injection_mode == 'add':
                    if self.pos_encoding_channels == x.shape[1]:
                        # Add to x
                        x = x + block_pos_encoding
                    else:
                        assert block_geom_feature is not None, \
                            f'Wrong channel counts at layer res {res} for adding positional encoding'
                        if self.pos_encoding_channels == block_geom_feature.shape[1]:
                            block_geom_feature = block_geom_feature + block_pos_encoding
                        elif self.pos_encoding_channels == block_geom_feature.shape[1] + x.shape[1]:
                            x = torch.cat([x, block_geom_feature], dim=1)
                            x = x + block_pos_encoding
                            block_geom_feature = None
            if block_geom_feature is not None:
                x = torch.cat([x, block_geom_feature], dim=1)

        if len(debug_data) > 0:
            return img, debug_data
        return img

#----------------------------------------------------------------------------

@persistence.persistent_class
class Generator(torch.nn.Module):
    @staticmethod
    def create_from_unpickled(unpickled_G):
        G = Generator(None, None, None, None, None)

        for attr in dir(unpickled_G):
            if attr not in dir(G):
                setattr(G, attr, getattr(unpickled_G, attr))

        G.synthesis = SynthesisNetwork.create_from_unpickled(unpickled_G.synthesis)
        return G

    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        positional_kwargs = None, # Type of positional encoding or None
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        synthesis_kwargs    = {},   # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        if z_dim is None:
            return
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.positional_kwargs = positional_kwargs
        if self.positional_kwargs is None:
            self.positional_kwargs = dnnlib.EasyDict()
            self.positional_kwargs.positional_encoding = None
            self.positional_kwargs.posenc_inject_resolutions = []
            self.positional_kwargs.posenc_featuremap_mode = 'fixed'
            self.positional_kwargs.posenc_injection_mode = 'cat'

        if self.positional_kwargs.positional_encoding is None:
            self.positional_encoder = None
            self.positional_kwargs.posenc_inject_resolutions = []
        else:
            self.positional_encoder = PositionalEncodingFactory.create_from_string(
                self.positional_kwargs.positional_encoding, img_resolution)
            assert len(self.positional_kwargs.posenc_inject_resolutions) > 0, 'Positional encoding will be ignored'

        self.synthesis = SynthesisNetwork(
            w_dim=w_dim, img_resolution=img_resolution, img_channels=img_channels,
            pos_encoding_channels=(0 if self.positional_encoder is None else self.positional_encoder.out_channels()),
            pos_encoding_feature_resolutions=[2 ** (2 + res) for res in self.positional_kwargs.posenc_inject_resolutions],
            pos_encoding_injection_mode=self.positional_kwargs.posenc_injection_mode,
            **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)
        self.geom_inject = True

    def set_trainable_layers(self, mode='all'):
        """
        Option

        @param mode: string of 'all', 'rgb', 'last_and_rgb', 'linear', 'all_but_linear' OR list thereof
        @return:
        """
        def _set_parameters_trainable(model, enabled):
            for p in model.parameters():
                p.requires_grad = enabled

        def _set_trainable_for_mode(G, mode):
            if mode == 'all':
                G.requires_grad_(True)
                _set_parameters_trainable(G, True)
            elif mode == 'all_but_linear':
                G.requires_grad_(True)
                _set_parameters_trainable(G, True)
                _set_parameters_trainable(G.synthesis.geom_linear, False)
            elif mode == 'rgb':
                _set_parameters_trainable(G.synthesis.get_last_block().torgb, True)
            elif mode == 'last_and_rgb':
                _set_parameters_trainable(G.synthesis.get_last_block(), True)
            elif mode == 'linear':
                _set_parameters_trainable(G.synthesis.geom_linear, True)
            else:
                raise RuntimeError(f'Not implemented mode {mode}')

        _set_parameters_trainable(self, False)
        if type(mode) == list:
            for m in mode:
                _set_trainable_for_mode(self, m)
        else:
            _set_trainable_for_mode(self, mode)

    def generate_positional_encoding(self, z, positions):
        if self.positional_encoder is None:
            return None

        if positions is None:
            positions = torch.randint(0, self.img_resolution, (z.shape[0], 2), dtype=torch.int64).to(z.device)

        pos_encoding = []
        if self.positional_kwargs.posenc_featuremap_mode == 'fixed':
            # Same positional encoding for every pixel in the feature map
            encoding = self.positional_encoder(positions[:, 1], positions[:, 0])  # B x nchannels

            for res in self.positional_kwargs.posenc_inject_resolutions:
                # 0 - 4, 1 - 8, 2 - 16, 3 - 32
                fmap_size = 2 ** (2 + res)
                pos_encoding.append(encoding.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, fmap_size, fmap_size))
        elif self.positional_kwargs.posenc_featuremap_mode == 'varying':
            for res in self.positional_kwargs.posenc_inject_resolutions:
                # 0 - 4, 1 - 8, 2 - 16, 3 - 32
                fmap_size = 2 ** (2 + res)
                pos_encoding.append(self.positional_encoder.encode_grid(positions[:, 1], positions[:, 0], fmap_size))
        else:
            raise RuntimeError(f'Unknown posenc_featuremap_mode {self.positional_kwargs.posenc_featuremap_mode}')

        return pos_encoding

    def forward_pre_mapped(self, ws, geom_feature, positions=None,
                           return_debug_data=False, return_features=None, blended_features=None,
                           noise_buffers=None, **synthesis_kwargs):
        pos_encoding = self.generate_positional_encoding(ws, positions)

        norm_positions = None
        if positions is not None:
            norm_positions = (positions % self.img_resolution) / (self.img_resolution - 1)
        syn_res = self.synthesis(ws, geom_feature, pos_encoding=pos_encoding,
                                 return_debug_data=return_debug_data,
                                 return_features=return_features, blended_features=blended_features,
                                 **synthesis_kwargs, norm_noise_positions=norm_positions,
                                 noise_buffers=noise_buffers)  # B x ncolors x W x W
        if return_debug_data or return_features:
            img = syn_res[0]
            debug_data = syn_res[1]
            if return_debug_data:
                debug_data['ws'] = ws
            return img, debug_data
        return syn_res

    def forward(self, z, c, geom_feature, positions=None, noise_buffers=None, truncation_psi=1, truncation_cutoff=None,
                return_debug_data=False, return_features=None, blended_features=None, style_mixing_prob=0, **synthesis_kwargs):
        """

        @param z:
        @param c:
        @param geom_feature:
        @param positions: B x 2 int tensor of y and x positions of a patch
        @param truncation_psi:
        @param truncation_cutoff:
        @param return_debug_data:
        @param return_features: resolutions at which to return features 4, 8, 16...
        @param blended_features: dictionary resoluton : stitching.BlendedFeatures
        @param style_mixing_prob:
        @param synthesis_kwargs:
        @return:
        """
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        if style_mixing_prob > 0:
            # Style mixing
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < style_mixing_prob, cutoff,
                                     torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.mapping(torch.randn_like(z), c,
                                              truncation_psi=truncation_psi,
                                              truncation_cutoff=truncation_cutoff,
                                              skip_w_avg_update=True)[:, cutoff:]

        return self.forward_pre_mapped(
            ws, geom_feature, positions=positions,
            return_debug_data=return_debug_data, return_features=return_features, blended_features=blended_features,
            noise_buffers=noise_buffers,
            **synthesis_kwargs)

