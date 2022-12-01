# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import torch.nn as nn

from forger.experimental.autoenc.base import BaseGeoEncoder


def add_model_flags(parser):
    parser.add_argument(
        '--encoder_pre_filters', action='store', type=int, default=64,
        help='Encoder filters computed at full input resolution.'
    )
    parser.add_argument(
        '--encoder_down_filters', action='store', type=str, default='128,256,256',
        help='Number of filters to use for encoder downsampling layers.'
    )
    parser.add_argument(
        '--encoder_post_filters', action='store', type=str, default='32,16',
        help='Number of filters to use at the encoding resolution.'
    )
    parser.add_argument(
        '--decoder_up_filters', action='store', type=str, default='256,128,64',
        help='Number of filters to use for decoder upsampling layers.'
    )
    parser.add_argument(
        '--neg_slope', type=float, default=None,
        help='The negative slope for LeakyReLU in convolutional layers'
    )
    parser.add_argument(
        '--decoder_pre_filters', action='store', type=int, default=-1,
        help='Encoder filters computed at full input resolution.'
    )


def _to_int_list(val):
    return [int(x) for x in val.split(',') if len(x) > 0]


def model_from_flags(args):
    enc_kwargs = {}
    dec_kwargs = {}
    # Support newer version of the model without breaking backaward compatibility
    if hasattr(args, 'neg_slope') and args.neg_slope is not None:
        # TODO: this is a bit of a hack, linking newer versions to neg_slope
        # In practice we see worse performace for our specific domain, so sticking with orig
        # code for experiments (i.e. don't set neg_slope to replicate)
        enc_kwargs = {'neg_slope': args.neg_slope,
                      'batchnorm_after_activation': True}
        dec_kwargs = {k: v for k,v in enc_kwargs.items()}
        dec_kwargs['scale_up_v2'] = True
    if hasattr(args, 'decoder_pre_filters'):
        dec_kwargs['pre_layer_filters'] = args.decoder_pre_filters

    model = AutoEncoder(
        Encoder(
            args.encoder_in_channels,
            pre_layer_filters=args.encoder_pre_filters,
            down_layer_filters=_to_int_list(args.encoder_down_filters),
            post_layer_filters=_to_int_list(args.encoder_post_filters),
            **enc_kwargs),
        Decoder(
            in_channels=_to_int_list(args.encoder_post_filters)[-1],
            out_channels=args.decoder_out_channels,
            up_layer_filters=_to_int_list(args.decoder_up_filters),
            **dec_kwargs))
    return model


def summary_from_flags(args):
    # TODO: can we do this summary more elegantly?
    latent_width = max(*_to_int_list(args.widths)) / pow(2, len(args.encoder_down_filters))
    return {"encoder": f'fixed(C-BN-LRlu:{args.encoder_pre_filters}) - down(C-BN-LRlu:{args.encoder_down_filters}) - fixed(C-BN-LRlu:{args.encoder_post_filters})',
            "decoder": f'up(Up-C-BN-LRlu{args.decoder_up_filters}) - fixed(C{args.decoder_out_channels})',
            "latent_shape": '%d x %d x %d' % (int(latent_width), int(latent_width), int(args.encoder_post_filters[-1]))
            }


# class DoubleConvolution(nn.Module):
#     def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, neg_slope=0.2):
#         super().__init__()
#         self.conv = nn.Sequential(
#             SingleConvolution(in_ch, out_ch, kernel_size, padding, stride),
#             SingleConvolution(out_ch, out_ch, kernel_size, padding, stride))
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x


class SingleConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, neg_slope=None, batchnorm_after_activation=False):
        super().__init__()
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, stride=stride, padding_mode='reflect')
        activation = nn.LeakyReLU(inplace=True) if neg_slope is None else nn.LeakyReLU(neg_slope, inplace=True)

        # Backward compatibility
        if batchnorm_after_activation is True:
            self.conv = nn.Sequential(conv, activation, nn.BatchNorm2d(out_ch))
        else:
            self.conv = nn.Sequential(conv, nn.BatchNorm2d(out_ch), activation)

    def forward(self, x):
        x = self.conv(x)
        return x


class ScaleUp(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = SingleConvolution(in_ch, out_ch)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class ScaleUpV2(nn.Module):
    def __init__(self, in_ch, out_ch, neg_slope):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
            nn.LeakyReLU(neg_slope),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        return self.conv(x)


# 256x256 --> 64 filters, same resolution
# 128x128 --> 128
# 64x64 --> 256
# 32x32 --> 256, 32
class Encoder(nn.Module):
    def __init__(self, in_channels,
                 neg_slope=None,
                 batchnorm_after_activation=False,
                 pre_layer_filters=64,
                 down_layer_filters=[128, 256, 256],
                 post_layer_filters=[32]):
        super().__init__()
        self.in_channels = in_channels
        self.emb_channels = post_layer_filters[-1]
        self.num_down_layers = len(down_layer_filters)

        # Architecture variants:
        # - Instead of stride can use Max pool
        #    MUNIT uses stride to downsample
        # - May want to consider instance normalization instead
        # - May want to add res blocks at the end to post-process feature map

        self.layers = []
        if pre_layer_filters > 0:
            layer_filters = [pre_layer_filters] + down_layer_filters
            self.layers.append(SingleConvolution(
                in_channels, layer_filters[0], kernel_size=7, stride=1, padding=3, neg_slope=neg_slope,
                batchnorm_after_activation=batchnorm_after_activation))
        else:
            layer_filters = [in_channels] + down_layer_filters

        for i in range(1, len(layer_filters)):
            self.layers.append(
                SingleConvolution(
                    layer_filters[i - 1], layer_filters[i],
                    kernel_size=3, stride=2, padding=1, neg_slope=neg_slope,
                    batchnorm_after_activation=batchnorm_after_activation))
        layer_filters = [layer_filters[-1]] + post_layer_filters
        for i in range(1, len(layer_filters)):
            self.layers.append(
                SingleConvolution(
                    layer_filters[i - 1], layer_filters[i],
                    kernel_size=3, stride=1, padding=1,
                    batchnorm_after_activation=batchnorm_after_activation))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.model(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels,
                 out_channels,
                 neg_slope=None,
                 pre_layer_filters=-1,
                 batchnorm_after_activation=False,
                 scale_up_v2=False,
                 up_layer_filters=[256, 128, 64]):
        super().__init__()
        self.out_channels = out_channels
        self.up_layer_filters = up_layer_filters

        self.layers = []

        if pre_layer_filters > 0:
            self.first = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=pre_layer_filters,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode='reflect'
                ),
                nn.LeakyReLU(neg_slope),
                nn.BatchNorm2d(pre_layer_filters)
            )
            layer_filters = [pre_layer_filters] + up_layer_filters
        else:
            self.first = None
            layer_filters = [in_channels] + up_layer_filters

        for i in range(1, len(layer_filters)):
            if scale_up_v2:
                layer = ScaleUpV2(layer_filters[i - 1], layer_filters[i], neg_slope=neg_slope)
            else:
                layer = ScaleUp(layer_filters[i - 1], layer_filters[i])
            self.layers.append(layer)

        if out_channels != layer_filters[-1]:
            self.layers.append(nn.Conv2d(layer_filters[-1], out_channels, 1))
        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        if self.first is not None:
            x = self.first(x)
        x = self.model(x)
        return x

    def decode_partial(self, x, nlayers):
        assert nlayers < len(self.layers)
        results = []

        if self.first is not None:
            x = self.first(x)

        for i in range(nlayers):
            x = self.layers[i](x)
            results.append(x.detach())
        return results


class AutoEncoder(BaseGeoEncoder):
    def __init__(self, ec, dec):
        super().__init__()

        self.encoder = ec
        self.decoder = dec

    def forward(self, x, return_encoding=False):
        encoding = self.encoder(x)
        recon = self.decoder(encoding)
        if return_encoding:
            return recon, encoding
        return recon

    def feature_channels(self, res=0):
        channels = [self.encoder.emb_channels] + self.decoder.up_layer_filters
        assert 0 <= res < len(channels), 'Other resolutions not implemented'
        return channels[res]

    def decoder_out_channels(self):
        return self.decoder.out_channels

    def num_downsampling_layers(self):
        return self.encoder.num_down_layers

    def _encode(self, preprocessed_geom, res=0):
        encoding = self.encoder(preprocessed_geom)

        max_res = res if type(res) != list else max(res)
        results = [encoding] + self.decoder.decode_partial(encoding, max_res)

        if type(res) != list:
            res = [res]
        return [results[x] for x in res]
