# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
from abc import ABC, abstractmethod
import torch
import torch.nn as nn


def _softmax_uvs_normalization(raw_out):
    """

    @param raw_out: B x 3 x W x W
    @return:
    """
    res = torch.softmax(raw_out, dim=1)
    # Return black on white by default


class BaseGeoEncoder(nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.preproc_name = None
        self.preproc_func = lambda x: x
        self.postproc_final_func = lambda x: x
        self.preproc_truth = lambda x: x  # Only needed if loss takes raw output, eg BCEWithLogitsLoss
        self.res = 0

    def set_preprocessing(self, name):
        """

        @param name: "none", "inverse", "-11inverse"
        @return:
        """
        self.preproc_name = name
        self.preproc_func = lambda x: x
        self.postproc_final_func = lambda x: x

        if name is None or name == 'none':
            pass
        elif name == '-11inverse':
            self.preproc_func = lambda x: (1 - x) * 2 - 1  # pre-norm and turn the stroke to white on black
            if self.decoder_out_channels() == 1:
                self.postproc_final_func = lambda x: 1 - x  # because there is still sigmoid on output
        elif name == 'inverse':
            self.preproc_func = lambda x: 1 - x
            if self.decoder_out_channels() == 1:
                self.postproc_final_func = lambda x: 1 - x
        else:
            raise RuntimeError(f'Unknown preprocessing type "{name}"')

        if name is not None and 'inverse' in name or self.decoder_out_channels() == 3:
            self.preproc_truth = lambda x: 1 - x
        else:
            self.preproc_truth = lambda x: x

    def preprocess_truth_for_logits(self, x):
        return self.preproc_truth(x)

    def preprocess(self, x):
        """

        @param x: B x 1 X H x W binary black stroke on white 0..1 torch float tensor
        @return: B x 1 x H x W torch float tensor
        """
        return self.preproc_func(x)

    # Note: these are not pretty, but these are only used for encoder training
    def postprocess(self, y):
        res = self.postprocess_partial(y)
        if self.decoder_out_channels() == 1:
            res = torch.sigmoid(res + 0.5)  # bias zero-ceontered output
        elif self.decoder_out_channels() == 3:
            res = res[:, 1:, ...]  # Take background, as black on white is default
        else:
            raise RuntimeError(f'Unsupported decoder channels {self.decoder_out_channels()}')
        return self.postproc_final_func(res)

    def postprocess_partial(self, y):
        if self.decoder_out_channels() == 1:
            return y
        elif self.decoder_out_channels() == 3:
            res = torch.softmax(y, dim=1)
            res = torch.cat([torch.sum(res[:, :2, ...], dim=1, keepdim=True),
                             res[:, 2:, ...]], dim=1)
            return res  # Always FG, BG channel order for dim=1
        else:
            raise RuntimeError(f'Unsupported decoder channels {self.decoder_out_channels()}')

    @abstractmethod
    def feature_channels(self, res=0):
        """
        @param res: 0 - bottleneck, 1 - one layer up in the decoder, 2 - one layer up
        @return:
        """
        pass

    def featuremap_resolution(self, input_res, res=0):
        """

        @param input_res:
        @param res: 0 - bottleneck, 1 - one layer up in the decoder, 2 - one layer up
        @return:
        """
        enc_res = input_res // (2 ** self.num_downsampling_layers())
        return enc_res * (2 ** res)

    @abstractmethod
    def decoder_out_channels(self):
        pass

    @abstractmethod
    def num_downsampling_layers(self):
        pass

    @abstractmethod
    def _encode(self, preprocessed_geom, res=0):
        pass

    def encode(self, geom, res=None):
        """
        Does appropriate pre-processing and encodes the geometry into the representation
        that stylegan can ingest.

        @param geom: B x 1 x H x W 0..1 black on white float tensors of geometry images
        @param res: resolutions to provide encodings for
        @return: list of feature maps
        """
        if res is None:
            res = self.res
        return self._encode(self.preprocess(geom), res=res)

    def set_default_encode_resolutions(self, res):
        """
        Sets default resolution(s) to pass to encode.

        @param res: resolutions to provide encodings for, 0 - bottleneck, 1 - one layer up in the decoder, etc.
        @return:
        """
        self.res = res