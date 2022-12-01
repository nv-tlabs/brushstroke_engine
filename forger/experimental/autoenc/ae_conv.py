# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import numpy as np
import torch.nn as nn

from forger.experimental.autoenc.base import BaseGeoEncoder


def add_model_flags(parser):
    parser.add_argument(
        '--emb_channel', action='store', type=int, default=4,
        help='The number of channel of the feature map'
    )
    parser.add_argument(
        '--enc_layer', action='store', type=int, default=4,
        help='The number of layers in encoder'
    )
    parser.add_argument(
        '--dec_layer', action='store', type=int, default=4,
        help='The number of convolutional layers in the decoder'
    )
    parser.add_argument(
        '--channel_factor', action='store', type=int, default=4,
        help='The multiplier that determines the number of channels in the output tensor'
    )
    parser.add_argument(
        '--neg_slope', type=float, default=0.2,
        help='The negative slope for LeakyReLU in convolutional layers'
    )
    return parser


def model_from_flags(args):
    encoder_kwargs = {
        'in_channel': args.encoder_in_channels,
        'num_layer': args.enc_layer,
    }
    decoder_kwargs = {
        'out_channel': args.decoder_out_channels,
        'num_layer': args.dec_layer,
    }
    model = Autoencoder(
        img_width=args.width,
        emb_channel=args.emb_channel,
        channel_factor=args.channel_factor,
        neg_slope=args.neg_slope,
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
    )
    return model


def summary_from_flags(args):
    latent_width = args.width / pow(2, args.enc_layer)
    filter_list = [args.channel_factor * (2 ** i) for i in range(args.enc_layer)]
    summary = {
        "latent shape": "%d x %d x %d" % (latent_width, latent_width, args.emb_channel),
        "channel factor": args.channel_factor,
        "LReLU neg slope": args.neg_slope,
        "encoder filter": f'down(Conv-BN-LReLU:{filter_list}) - fixed(Conv-BN-LReLU:{args.emb_channel})',
        "decoder filter": f'up(ConvTran-BN-LReLU:{[i for i in reversed(filter_list)]})'
    }
    return summary


class Autoencoder(BaseGeoEncoder):
    """
    Autoencoder
    The latent code is a 3D tensor of shape (emb_channel, L, L), where
        emb_channel: a user-configurable parameter
        L: (Input image resolution) / 2^(enc_layer). enc_layer is a user-configurable parameter.
    """
    def __init__(self,
                 img_width=None,     # Input and output resolution
                 emb_channel=4,      # The number of channel of the feature map
                 channel_factor=4,   # The factor for controlling number of channel in the down-sampling and up-sampling layers
                 neg_slope=0.2,      # Negative slope
                 encoder_kwargs={},  # Arguments for Encoder
                 decoder_kwargs={},  # Arguments for Decoder
                 ):
        super(Autoencoder, self).__init__()
        self.encoder = CnnEncoder(
            img_width,
            emb_channel,
            channel_factor=channel_factor,
            neg_slope=neg_slope,
            **encoder_kwargs)
        self.decoder = CnnDecoder(
            img_width,
            emb_channel,
            channel_factor=channel_factor,
            neg_slope=neg_slope,
            **decoder_kwargs
        )

    def forward(self, x, return_encoding=False):
        encoding = self.encoder(x)
        recon = self.decoder(encoding)
        if return_encoding:
            return recon, encoding
        return recon

    def feature_channels(self, res=0):
        assert res == 0, 'Other resolutions not implemented'
        return self.encoder.emb_channel

    def decoder_out_channels(self):
        return self.decoder.out_channel

    def num_downsampling_layers(self):
        return self.encoder.num_layer

    def _encode(self, preprocessed_geom, res=0):
        assert res == 0 or res == [0], 'Other resolutions not implemented'
        return [self.encoder(preprocessed_geom)]


class CnnEncoder(nn.Module):
    def __init__(self,
                 in_resolution,         # The resolution of the input image
                 emb_channel,           # Number of channels of the geometry feature
                 channel_factor=None,   # The multiplier that controls the number of out_channels in each down-sampling layer.
                 neg_slope=0.2,         # The 'neg_slope' parameter of Leaky ReLU in down-sampling convolutional layers
                 num_layer=4,           # The number of down-sampling convolutional layers
                 in_channel=3,          # The number of channels of the input image
                 ):
        """
        CnnEncoder consists of `num_layer` down-sampling layers, and each of these layers reduce the resolution of
        the tensor by a factor of 2.
        Consider layer i.
        Its output tensor will have the shape [B, `channel_factor` * 2^i, in_resolution / 2 ^ (i+1), in_resolution / 2 ^ (i+1) ]
        """
        super(CnnEncoder, self).__init__()
        assert num_layer > 1
        self.num_layer = num_layer
        self.in_resolution = in_resolution
        self.in_channels = in_channel
        self.emb_channel = emb_channel
        self.in_resolution_log2 = int(np.log2(in_resolution))
        self.conv_layer_resolutions = [2 ** i for i in range(self.in_resolution_log2, max(self.in_resolution_log2 - num_layer, 2), -1)]
        self.channels_dict = {self.conv_layer_resolutions[i]: channel_factor * (2 ** i) for i in range(len(self.conv_layer_resolutions))}

        self.final = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.channels_dict[self.conv_layer_resolutions[-1]],
                    out_channels=emb_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    padding_mode='reflect',
                ),
                nn.LeakyReLU(neg_slope),
                nn.BatchNorm2d(emb_channel)
            )

        for i, res in enumerate(self.conv_layer_resolutions):
            in_channels = self.channels_dict[res * 2] if i != 0 else self.in_channels
            out_channels = self.channels_dict[res]
            layer = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    padding_mode='reflect',
                ),
                nn.LeakyReLU(neg_slope),
                nn.BatchNorm2d(out_channels)
            )
            setattr(self, f'layer{res}', layer)

    def forward(self, x):
        for res in self.conv_layer_resolutions:
            conv_layer = getattr(self, f'layer{res}')
            x = conv_layer(x)
        # print("encoder, x shape before flatten", x.shape)
        x = self.final(x)
        # print("after encoder fc", x.shape)
        return x


class CnnDecoder(nn.Module):
    def __init__(self,
                 out_resolution,
                 emb_channel,
                 out_channel=3,
                 channel_factor=4,
                 num_layer=4,
                 neg_slope=0.2):
        """
        CnnDecoder is a mirror of CnnEncoder. See the doc of CnnEncoder.
        """
        super(CnnDecoder, self).__init__()
        self.out_resolution = out_resolution
        self.emb_channel = emb_channel
        self.out_channel = out_channel
        self.out_resolution_log2 = int(np.log2(out_resolution))
        self.conv_layer_resolutions = [2 ** i for i in range(max(self.out_resolution_log2 - num_layer, 2), self.out_resolution_log2)]
        self.channels_dict = {self.conv_layer_resolutions[i]: channel_factor * 2 ** (num_layer-i-1) for i in range(num_layer)}

        self.first = nn.Sequential(
            nn.Conv2d(
                in_channels=emb_channel,
                out_channels=self.channels_dict[self.conv_layer_resolutions[0]],
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode='reflect'
            ),
            nn.LeakyReLU(neg_slope),
            nn.BatchNorm2d(self.channels_dict[self.conv_layer_resolutions[0]])
        )

        for i, res in enumerate(self.conv_layer_resolutions):
            in_channels = self.channels_dict[res]
            out_channels = self.channels_dict[res * 2] if res < self.conv_layer_resolutions[-1] else self.out_channel
            layer = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    dilation=1,
                    output_padding=1,
                ),
                nn.LeakyReLU(neg_slope),
                nn.BatchNorm2d(out_channels)
            )
            setattr(self, f'layer{res}', layer)

    def forward(self, x):
        x = self.first(x)
        for res in self.conv_layer_resolutions:
            conv_layer = getattr(self, f'layer{res}')
            x = conv_layer(x)
        return x