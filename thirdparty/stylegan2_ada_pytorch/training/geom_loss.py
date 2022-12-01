# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import logging
import torch
import torch.nn as nn

import warnings
warnings.warn("this module is deprecated", DeprecationWarning,
              stacklevel=2)

logger = logging.Logger(__name__)

# ----------------------------------------------------------------------------
"""
Format of internal_loss_dict:
{
    geom_loss_weights: [],
    geom_uvs_types: [],
    geom_alpha_types: []
}
"""

class GeometryLoss:
    """
    The loss that wraps one or more segmentation losses between the "foreground" component and the geometry input.
    """
    def __init__(self,
                 device,  # We are not using any nn.Module currently
                 color_format='triad',  # Possible values: "triad", "canvas"
                 internal_loss_dict={},
                 ):
        super().__init__()
        self._validate_internal_dict(color_format=color_format, internal_loss_dict=internal_loss_dict)
        logger.debug("Color format: ", color_format)
        self.loss_func_dict = dict()
        self.weights = dict()
        self.result = {
            'total': 0.0,
            'unweighted': {},
        }

        # Update parameters related to mask loss.

        for loss_type in internal_loss_dict['geom_loss_types']:
            parsed = loss_type.split('_')
            if parsed[0] == 'mask':
                self.loss_func_dict.update({loss_type: _MaskLoss(loss_types=[parsed[-1]])})
            else:
                raise RuntimeError("Unsupported geometry loss type")
        logger.debug("Loss func dict           : ", self.loss_func_dict)

        # Populate the result dict and
        self._init_loss_dicts(component='uvs', internal_loss_dict=internal_loss_dict)
        self._init_loss_dicts(component='alpha', internal_loss_dict=internal_loss_dict)
        logger.debug("Geometry loss result dict: ", self.result)
        logger.debug("Geometry loss weight dict:", self.weights)

    def _validate_internal_dict(self, color_format, internal_loss_dict):
        """
        Given two possible values for color_format - 'triad' and 'canvas' check if the corresponding weight of losses is provided.
        For 'triad', the user must provide `geom_uvs_weights`, and `geom_alpha_weights` is ignored.
        For `canvas`, the user must provide either `geom_alpha_weights` or `geom_alpha_weights`, or both.
        @param color_format
        @param internal_loss_dict: A dict object with keys `geom_loss_types`, `geom_uvs_weights` and `geom_alpha_weights`
                                    `geom_loss_types` maps to a list of name string of losses;
                                    `geom_uvs_weights` and `geom_alpha_weights` maps to a list of floats.
        """
        assert color_format in ('triad', 'canvas')
        self.color_format = color_format
        if self.color_format == 'triad':
            assert internal_loss_dict.get('geom_uvs_weights') is not None
        else:
            assert internal_loss_dict.get('geom_alpha_weights') is not None or internal_loss_dict.get(
                'geom_uvs_weights') is not None
        assert internal_loss_dict.get('geom_loss_types') is not None
        assert len(internal_loss_dict['geom_uvs_weights']) == len(internal_loss_dict['geom_loss_types']), \
            f"For geometry loss, loss_types has len {internal_loss_dict['geom_loss_types']} " \
            f"but loss_weights has len {len(internal_loss_dict['geom_loss_weights'])}"

    # TODO: Remove the loss dict
    def _init_loss_dicts(self,
                        component,  # takes value in ['uvs', 'alpha']
                        internal_loss_dict):
        assert component in ('uvs', 'alpha')
        loss_weights = internal_loss_dict.get('geom_' + component + '_weights')
        if loss_weights is None:
            return
        self.weights.update({component: {}})
        self.result['unweighted'].update({component: {}})

        for loss_type, loss_weight in zip(internal_loss_dict['geom_loss_types'], loss_weights):
            # Skip those weight = 0
            if loss_weight <= 0.0:
                continue
            self.weights[component].update({loss_type: loss_weight})
            self.result['unweighted'][component].update({loss_type: 0.0})

    def _foreground_data(self, component, gen_data):
        """
        Returns the foreground component of the Generator output.
        If color_format is `triad`, component can only take the value of `uvs`.
        If color_format is `canvas`, component can be either `uvs` or `alpha`, although we never use `uvs` here.
        @param component: a str, indicating the component to use for computing geom_loss. Valid values: "uvs" and "alpha"
        @param gen_data: a dict object that holds the output of the Generator.
        @return: a torch.Tensor of size [B, C, W, H]
        """
        if component == 'uvs':
            if self.color_format == 'triad':
                return gen_data['uvs'][:, 1:, ...]
            else:
                return gen_data['uvs']
        elif component == 'alpha':
            assert self.color_format == 'canvas'
            return gen_data['alpha_fg']
        else:
            raise RuntimeError("Unrecognized foreground component")

    def compute(self, gen_data, geom_truth):
        total_loss = 0.0
        for component, weights in self.weights.items():
            foreground_data = self._foreground_data(component=component, gen_data=gen_data)
            for loss_type, loss_weight in weights.items():
                loss_func = self.loss_func_dict[loss_type]
                result = loss_func.compute(foreground_data=foreground_data, geom_truth=geom_truth)
                prefix = loss_func.prefix
                for key, unweighted in result.items():
                    self.result['unweighted'][component][prefix + '_' + key] = unweighted
                    total_loss += loss_weight * unweighted

        self.result['total'] = total_loss
        return total_loss, self.result


class _MaskLoss:
    """
    maskLoss makes use of the mask produced by color triad.
    """
    def __init__(self,
                 loss_types=["bce"],
                 ):
        logger.debug(f"Internal loss types: {loss_types}")
        # assert color_format is not None  # maskLoss shouldn't have to know about color_format
        self.prefix = "mask"
        # self.color_format = color_format
        self.compute_func = dict()
        # self._register_compute_func()
        self.loss_funcs = {}
        self.unweighted_results = {}
        for loss_type in loss_types:
            self.loss_funcs.update({loss_type: init_segmentation_loss(loss_type)})
            self.unweighted_results.update({loss_type: 0.0})

    def compute(self, foreground_data, geom_truth):
        """
        @param foreground_data: torch.Tensor of shape [B, C, W, H]
        @param geom_truth: torch.Tensor of shape [B, 1, W, H]
        @return the loss value, of type torch scalar
        """
        loss_func_input = torch.sum(foreground_data, dim=1).unsqueeze(1)
        for loss_type, loss_func in self.loss_funcs.items():
            loss = loss_func(loss_func_input, 1.0 - geom_truth)
            self.unweighted_results[loss_type] = loss
        return self.unweighted_results

def init_segmentation_loss(loss_type):
    assert loss_type in ["bce", "dice", "iou", "l1", "l2"], "Unrecognized loss type for segmentation"
    if loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "iou":
        return IoULoss()
    elif loss_type == "l1":
        return nn.L1Loss()
    else:
        return nn.MSELoss()





# -----------------------------------------------------------------
# Various Loss class used in segmentation literature


class IoULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source: torch.Tensor, target: torch.Tensor):
        intersection = torch.sum(torch.mul(source, target))
        total = torch.sum(source + target)
        # print("total", total)
        union = total - intersection
        # print("union ", union)
        return 1.0 - intersection / union


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, source, target):
        intersection = torch.sum(source * target)
        total = torch.sum(torch.pow(source, 2) + torch.pow(target, 2))
        return 1.0 - 2.0 * intersection / total
