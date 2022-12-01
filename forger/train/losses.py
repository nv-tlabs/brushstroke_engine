# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
from abc import ABC, abstractmethod
import logging
import re
import torch
import torch.nn as nn
import torchvision.transforms

import forger.metrics.geom_metric

logger = logging.getLogger(__name__)


def _robust_split_string(in_str, delimiter):
    return [x for x in in_str.strip().strip("'").replace(' ', '').split(delimiter) if len(x) > 0]


def extract_geometry_triloss_components(value, truth):
    """
    Expects truth to contain tri-images, with 1 - background, 0 - foreground and gray - neither.
    Extracts only FG and BG (not neither) components of both value and truth for evaluating loss.

    @param value:
    @param truth:
    @return:
    """
    mask = torch.logical_or(truth < 0.1, truth > 0.9)
    return value[mask], truth[mask]


class ForgerLosses(object):
    """
    High-level class containing a list of losses and their weights.
    Can be configured from a simple human-readable string.
    """

    @staticmethod
    def create_from_string(config_string):
        """
        Accepts string of the following form, summing N losses with weights:

        <float>*<loss_name>(<component>)+<float>*<loss_name>(<component>)

        For example:
        0.11*plpips(canvas)+0.5*mask_iou(alpha)

        @param config_string:
        @return:
        """
        parts = _robust_split_string(config_string, '+')
        logger.debug(f'Detected {len(parts)} loss parts')

        weights_losses = [ForgerLossItemFactory.create_from_string(x) for x in parts]
        weights = [x[0] for x in weights_losses]
        losses = [x[1] for x in weights_losses]
        return ForgerLosses(losses, weights)

    def __init__(self, losses, weights):
        self.losses = losses
        self.weights = weights
        self.validate()

    def set_partial_loss_with_triband_input(self, val):
        """
        Triband input means that geometry truth has white = BG, black = FG, grey = neither.
        Partial loss ignores these gray regions.
        @param val:
        @return:
        """
        for loss in self.losses:
            loss.partial_loss_with_triband_input = val

    def require_original_fake_image(self):
        for loss in self.losses:
            if loss.component == 'fake_orig':
                return True
        return False

    def is_empty(self):
        return len(self.losses) == 0

    def validate(self):
        assert len(self.losses) == len(self.weights)
        names = set()

        for loss in self.losses:
            name = loss.full_name()

            if name in names:
                raise RuntimeError(f'Loss with identifier {name} defined more than once')
            names.add(name)

    def compute(self, raw, geom_truth):
        """
        Computes all configured losses on the raw output of the generator.

        @param raw: debug_data returned from the generator
        @param geom_truth:
        @return: tuple (loss, dict)  where dict contains plain loss items
        """
        total = 0
        results = {}

        for loss, weight in zip(self.losses, self.weights):
            name = loss.full_name()
            results[name] = loss.compute(raw, geom_truth)
            total = total + weight * results[name]
        return total, results

    def print_summary(self, prefix='  '):
        for loss, weight in zip(self.losses, self.weights):
            print('%s%0.5f * %s' % (prefix, weight, loss.config_string()))


# TODO: Why is this called many times for every loss?? Something is not right.
def register_loss(LossClass):
    logger.info(f'Registering loss {LossClass}')
    ForgerLossItemFactory.register_loss(LossClass)
    return LossClass


class ForgerLossItemFactory:
    __function_pattern = re.compile(r'(\w*)\((\w*)(,[a-zA-Z0-9_,=\.]*)?\)')
    _registered_losses = {}
    _valid_components = {'canvas', 'uvs', 'u', 'alpha', 'fake_img', 'color_0', 'color_1', 'color_2',
                         'fake_orig', 'fake_composite', 'patch', 'fake'}

    @staticmethod
    def get_format_info_string():
        return '\n'.join(
            ['each loss formatted as: <float>*<loss_name>(<component>)',
             'valid losses: ' + ', '.join(ForgerLossItemFactory._registered_losses.keys()),
             'valid components: ' + ', '.join(list(ForgerLossItemFactory._valid_components))])

    @staticmethod
    def register_loss(LossClass):
        item = LossClass(component=None)
        ForgerLossItemFactory._register_loss(item.name, LossClass)

    @staticmethod
    def create_from_string(config_string):
        """
        Accepts config string for a single loss component, of the form:

        <float>*<loss_name>(<component>)
        or
        <float>*<loss_name>(<component>,arg0=val0,arg1=val1)

        @param config_string:
        @return: tuple (weight, instance of ForgerLossItem or subclass)
        """
        weight, loss_name, component, arg_dict = ForgerLossItemFactory.split_loss_string(config_string)

        if loss_name not in ForgerLossItemFactory._registered_losses:
            raise RuntimeError(f'Loss {loss_name} not found in registered losses: ' +
                               ', '.join(ForgerLossItemFactory._registered_losses.keys()))

        if component not in ForgerLossItemFactory._valid_components:
            raise RuntimeError(f'Component "{component}" not in valid values: ' +
                               ', '.join(ForgerLossItemFactory._valid_components))

        loss_item = ForgerLossItemFactory._registered_losses[loss_name](component=component, **arg_dict)
        loss_item.save_string_config(config_string)

        return weight, loss_item

    @staticmethod
    def _register_loss(name, LossClass):
        if name in ForgerLossItemFactory._registered_losses:
            raise RuntimeError(
                f'Attempting to register loss ({LossClass}) with {name}, but '
                f'loss {ForgerLossItemFactory._registered_losses[name]} already registered '
                'with the same name')
        ForgerLossItemFactory._registered_losses[name] = LossClass

    @staticmethod
    def args_string_to_dict(in_str):
        res = {}
        if in_str is None or len(in_str) == 0:
            return res

        parts = _robust_split_string(in_str, ',')
        for part in parts:
            arg_val = _robust_split_string(part, '=')
            assert len(arg_val) == 2, f'Invalid argument string {in_str}'
            assert arg_val[0] not in res, f'Argument {arg_val[0]}repeated in args {in_str}'
            res[arg_val[0]] = arg_val[1]
        return res

    @staticmethod
    def split_loss_string(in_string):
        """
        Splits string of the format:
            <float>*<loss_name>(<component>)
        or with optional arguments
            <float>*<loss_name>(<component>,arg0=val0,arg1=val1)
        or
            <loss_name>(<component>)   (1 assumed for weight)

        into parts.

        @param in_str:
        @return: tuple (weight :float, loss_name, component, arg_dict)
        """
        parts = _robust_split_string(in_string, '*')

        weight = 1.0
        if len(parts) == 1:
            logger.debug(f'Assuming weight 1 for loss string {in_string}')
        elif len(parts) == 2:
            weight = float(parts[0])
        else:
            raise RuntimeError(f'Mis-configured loss string {in_string}')
        loss_string = parts[-1]

        m = re.match(ForgerLossItemFactory.__function_pattern, loss_string)
        if m is None:
            raise RuntimeError(f'Mis-configured loss string {in_string}; '
                               'expected pattern <float>*<loss_name>(<component>)')

        loss_name = m.group(1)
        component = m.group(2)
        arg_dict = ForgerLossItemFactory.args_string_to_dict(m.group(3))

        return weight, loss_name, component, arg_dict

# Individual Losses ----------------------------------------------------------------------------------------------------


class ForgerLossItem(ABC):
    """
    Subclasses know how to compute a given loss type on
    each of the supported triad components -- slightly different
    logic on the inputs may be necessary, depending on the loss.
    """

    def __init__(self, name, component):
        self.name = name
        self.component = component
        self.string_config = None
        self.partial_loss_with_triband_input = False  # Hacky

    def prepare_geometry_values(self, value, truth):
        if self.partial_loss_with_triband_input:
            return extract_geometry_triloss_components(value, truth)
        else:
            return value, truth

    def save_string_config(self, string_config):
        self.string_config = string_config

    def full_name(self):
        return f'{self.name}_{self.component}'

    def config_string(self):
        if self.string_config is not None:
            return self.string_config
        return f'{self.name}({self.component})'

    @abstractmethod
    def compute(self, debug_data, geom_truth):
        pass

    def loss_function_by_name(self, loss_name):
        if loss_name == 'L1':
            return nn.L1Loss()
        elif loss_name == 'L2':
            return nn.MSELoss()
        else:
            raise RuntimeError(f'Unknown loss name {loss_name}')

    def throw_unsupported_component(self):
        raise RuntimeError(f'Unsupported component for {self.component} for loss {self.name}')

    def make_get_rgb_component_function(self):
        if self.component == 'canvas':
            return lambda debug_data: debug_data['canvas']

        # B x C x ncolors
        if self.component == 'color_0':
            return lambda debug_data: debug_data['colors'][..., 0]
        elif self.component == 'color_1':
            return lambda debug_data: debug_data['colors'][..., 1]
        elif self.component == 'color_2':
            return lambda debug_data: debug_data['colors'][..., 2]
        else:
            self.throw_unsupported_component()

    def get_foreground(self, debug_data, throw=True):
        """

        @param debug_data:
        @return: B x H x W [0..1] foreground mask, depending on self.component
        """
        if self.component == 'uvs':
            # Note: back to U - primary, V - secondary, S - canvas
            return torch.sum(debug_data['uvs'][:, :2, ...], dim=1)
        elif self.component == 'u':
            return debug_data['uvs'][:, 0, ...]
        elif self.component == 'alpha':
            return debug_data['alpha'][:, 0, ...]
        else:
            if throw:
                self.throw_unsupported_component()
            else:
                return None

    def get_background(self, debug_data, throw=True):
        if self.component == 'uvs':
            # Note: back to U - primary, V - secondary, S - canvas
            return debug_data['uvs'][:, 2, ...]
        elif self.component == 'alpha':
            return debug_data['alpha'][:, 1, ...]
        else:
            if throw:
                self.throw_unsupported_component()
            else:
                return None

    def random_patches(self, images, patch_width=None):
        """

        @param images: B x ch x W x W
        @param patch_width: int
        @return: B x ch x pW x pW
        """
        if patch_width is None:
            patch_width = images.shape[-1] // 4

        crop = torchvision.transforms.RandomCrop(patch_width)
        return crop(images)


@register_loss
class RgbTargetLossItem(ForgerLossItem):
    def __init__(self, component, r=0.5, g=0.5, b=0.5, loss='L1', mean_rgb=False):
        """

        @param component:
        @param r:
        @param g:
        @param b:
        @param loss:
        @param mean_rgb: if true, takes mean to get just one R,G,B per batch before applying loss
        """
        super().__init__('rgb', component=component)
        self.get_rgb_component = None
        self.loss = self.loss_function_by_name(loss)
        self.rgb = torch.tensor([float(r), float(g), float(b)], dtype=torch.float32)
        self.mean_rgb = bool(mean_rgb)

        if component is not None:  # true when registering
            if self.component == 'uvs':
                self.get_rgb_component = lambda debug_data: debug_data['uvs'] * 2 - 1
            else:
                self.get_rgb_component = self.make_get_rgb_component_function()

    def compute(self, debug_data, geom_truth):
        inpt = self.get_rgb_component(debug_data) * 0.5 + 0.5
        if self.mean_rgb:
            inpt = torch.stack([inpt[:, 0].mean(), inpt[:, 1].mean(), inpt[:, 2].mean()])

        shp = [1 for _ in inpt.shape]
        if len(shp) > 1:
            shp[1] = 3
        else:
            shp[0] = 3
        self.rgb = self.rgb.to(inpt.device).to(inpt.dtype).reshape(*shp)
        return self.loss(inpt, self.rgb.expand_as(inpt))


@register_loss
class HsvTargetLossItem(ForgerLossItem):
    def __init__(self, component, v=None, s=None, loss='L2'):
        super().__init__('hsv', component=component)
        self.get_rgb_component = None
        self.loss = self.loss_function_by_name(loss)
        self.v = v if v is None else float(v)
        self.s = s if s is None else float(s)

        if component is not None:  # true when registering
            assert v is not None or s is not None, 'Must enter at least one target'
            self.get_rgb_component = self.make_get_rgb_component_function()

    def to_sv(self, input):
        """
        Convert RGB to S (saturation) and V (value).

        @param input: B x 3 x ...  [-1...1]
        @return: B x 2 x ...
        """
        maxes = torch.max(input, dim=1)[0] * 0.5 + 0.5
        mins = torch.min(input, dim=1)[0] * 0.5 + 0.5
        v = maxes

        maxes = torch.clamp(maxes, min=0, max=1)
        mins = torch.clamp(mins, min=0, max=1)
        delta = maxes - mins
        maxes = torch.clamp(maxes, min=1.0/255)
        s = delta / maxes

        return torch.stack([s, v], dim=1)

    def ensure_tensor_targets(self, rgb):
        def _tensorize(x):
            shp = [1 for _ in rgb.shape]
            return torch.tensor([x], dtype=rgb.dtype, device=rgb.device).reshape(*shp)

        if self.v is not None and not torch.is_tensor(self.v):
            self.v = _tensorize(self.v)

        if self.s is not None and not torch.is_tensor(self.s):
            self.s = _tensorize(self.s)

    def compute(self, debug_data, geom_truth):
        rgb = self.get_rgb_component(debug_data)
        sv = self.to_sv(rgb)

        res = 0
        self.ensure_tensor_targets(rgb)
        if self.v is not None:
            res = res + self.loss(sv[:, 1:, ...], self.v.expand_as(sv[:, 1:, ...]))
        if self.s is not None:
            res = res + self.loss(sv[:, 0:, ...], self.s.expand_as(sv[:, 0:, ...]))
        return res


@register_loss
class PatchLPIPSLossItem(ForgerLossItem):
    def __init__(self, component):
        super().__init__('plpips', component=component)

    def compute(self, debug_data, geom_truth):
        if self.component == 'canvas':
            images = debug_data['canvas']  # TODO: Already -1...1?
        else:
            images = self.get_background(debug_data).unsqueeze(1).expand(-1, 3, -1, -1) * 2 - 1

        patches0 = self.random_patches(images)
        patches1 = self.random_patches(images)
        # TODO: ok to use alex net for loss?
        res = forger.metrics.geom_metric.lpips_batched(patches0, patches1)
        return res.mean()


@register_loss
class IoULossItem(ForgerLossItem):
    def __init__(self, component):
        super().__init__('iou', component=component)

    def compute(self, debug_data, geom_truth):
        target = 1 - geom_truth.squeeze(1)
        source = self.get_foreground(debug_data)
        source, target = self.prepare_geometry_values(source, target)
        return compute_iou(source, target)


@register_loss
class IoUInverseLossItem(ForgerLossItem):
    def __init__(self, component):
        super().__init__('iou_inv', component=component)

    def compute(self, debug_data, geom_truth):
        target = geom_truth.squeeze(1)
        source = self.get_background(debug_data)
        source, target = self.prepare_geometry_values(source, target)
        return compute_iou(source, target)


@register_loss
class DiceLossItem(ForgerLossItem):
    def __init__(self, component):
        super().__init__('dice', component=component)

    def compute(self, debug_data, geom_truth):
        target = 1 - geom_truth.squeeze(1)
        source = self.get_foreground(debug_data)
        source, target = self.prepare_geometry_values(source, target)
        return compute_dice(source, target)


@register_loss
class DiceInverseLossItem(ForgerLossItem):
    def __init__(self, component):
        super().__init__('dice_inv', component=component)

    def compute(self, debug_data, geom_truth):
        target = geom_truth.squeeze(1)
        source = self.get_background(debug_data)
        source, target = self.prepare_geometry_values(source, target)
        return compute_dice(source, target)


@register_loss
class L1LossItem(ForgerLossItem):
    def __init__(self, component):
        super().__init__('l1', component=component)
        self.l1 = nn.L1Loss()

    def compute(self, debug_data, geom_truth):
        if self.component == 'canvas':
            # TODO: convert to 0..1?
            # Compare canvas patches to each other
            target = self.random_patches(debug_data['canvas'])
            source = self.random_patches(debug_data['canvas'])
        elif self.component == 'fake_img':
            target = debug_data['fake_img'].detach()
            source = debug_data['fake_img']
        elif self.component == 'fake_orig':
            target = debug_data['fake_orig'].detach()
            source = debug_data['fake_img']
        elif self.component == 'fake_composite':
            target = debug_data['fake']
            source = debug_data['fake_composite']
        elif self.component == 'patch':
            target = debug_data['patch1']
            source = debug_data['patch2']
        else:
            target = 1 - geom_truth.squeeze(1)
            source = self.get_foreground(debug_data)
            source, target = self.prepare_geometry_values(source, target)

        return self.l1(source, target)


@register_loss
class GANLossItem(ForgerLossItem):
    def __init__(self, component):
        super().__init__('gan', component=component)

    def compute(self, debug_data, geom_truth):
        logits_key = '%s_logits' % self.component
        if logits_key not in debug_data:
            raise RuntimeError(f'Key {logits_key} expected in: {debug_data.keys()}')

        res = torch.nn.functional.softplus(-debug_data[logits_key])
        return res.mean()


@register_loss
class LPIPSLossItem(ForgerLossItem):
    def __init__(self, component):
        super().__init__('lpips', component=component)

    def compute(self, debug_data, geom_truth):
        if self.component == 'fake_composite':
            target = debug_data['fake']
            source = debug_data['fake_composite']
        elif self.component == 'fake_orig':
            target = debug_data['fake_orig'].detach()
            source = debug_data['fake_img']
        elif self.component == 'patch':
            target = debug_data['patch1']
            source = debug_data['patch2']
        else:
            self.throw_unsupported_component()

        res = forger.metrics.geom_metric.lpips_batched(target, source)
        return res.mean()


@register_loss
class BceLossItem(ForgerLossItem):
    def __init__(self, component):
        super().__init__('bce', component=component)
        self.bce = nn.BCELoss()  # Note: we can't use logits

    def compute(self, debug_data, geom_truth):
        target = (1 - geom_truth.squeeze(1)).unsqueeze(1)
        source = (self.get_foreground(debug_data).to(target.dtype)).unsqueeze(1)
        source, target = self.prepare_geometry_values(source, target)
        return self.bce(source, target)


@register_loss
class BgStdLossItem(ForgerLossItem):
    def __init__(self, component):
        super().__init__('bgstd', component=component)

    def compute(self, debug_data, geom_truth):
        target_bin = preproc_geometry_for_background_loss(geom_truth).squeeze(1)
        source = self.get_background(debug_data).to(target_bin.dtype)

        # TODO: this is slow, but this is essential to run std per image in batch
        factor = 1.0 / source.shape[0]
        res = 0
        for i in range(source.shape[0]):
            res = res + factor * torch.std(source[i, ...][target_bin[i, ...] > 0.9], unbiased=True)
        return res


@register_loss
class BgL2LossItem(ForgerLossItem):
    def __init__(self, component):
        super().__init__('bgl2', component=component)

    def compute(self, debug_data, geom_truth):
        target_bin = preproc_geometry_for_background_loss(geom_truth).squeeze(1)
        source = self.get_background(debug_data).to(target_bin.dtype)

        eps = 1e-8
        total = torch.sum(target_bin, dim=(1, 2)) + eps
        num = torch.sum(torch.pow(source, 2) * target_bin, dim=(1, 2))

        return 1 - (num / total).mean()


@register_loss
class FgGatedL4LossItem(ForgerLossItem):
    def __init__(self, component):
        super().__init__('fgl4gt', component=component)
        self.relu = torch.nn.ReLU()

    def compute(self, debug_data, geom_truth):
        target_bin = preproc_geometry_for_fg_loss(geom_truth).squeeze(1)
        if self.component == 'uvs':
            source = debug_data['uvs'][:, 0, ...].to(target_bin.dtype)
        else:
            self.throw_unsupported_component()

        eps = 1e-8
        total = torch.sum(target_bin, dim=(1, 2)) + eps
        num = torch.sum(torch.pow(source, 4) * target_bin, dim=(1, 2))
        return self.relu(0.6 - num / total).mean()


# Learning to Predict Crisp Boundaries, Deng et al., ECCV 2018
def compute_dice(source, target):
    """
    @param source: B x H x W float
    @param target: B x H x W float
    @return:
    """
    assert len(source.shape) == 3
    assert source.shape == target.shape

    eps = 1e-8
    intersection = torch.sum(source * target, dim=(1, 2))
    total = torch.sum(torch.pow(source, 2) + torch.pow(target, 2), dim=(1, 2)) + eps
    return 1.0 - 2.0 * (intersection / total).mean()


def compute_iou(source: torch.Tensor, target: torch.Tensor):
    """

    @param source: B x H x W float
    @param target: B x H x W float
    @return:
    """
    assert source.shape == target.shape

    eps = 1e-8
    if len(source.shape) == 3:
        intersection = torch.sum(source * target, dim=(1, 2))
        union = torch.sum(source + target, dim=(1, 2)) - intersection + eps
        return 1.0 - (intersection / union).mean()
    else:
        intersection = torch.sum(source * target)
        union = torch.sum(source + target) - intersection + eps
        return 1.0 - (intersection / union)


def preproc_geometry_for_background_loss(target):
    """
    Preprocesses geometry for background loss, setting all to 0 or 1,
    with conservative estimation of which pixels are background (1).

    @param geometry: [0..1] float tensor with 1=bg, 0=fg, with some blur
    @return: [0..1] float tensor with only 0 or 1
    """
    target_bin = torch.zeros_like(target)
    target_bin[target > 0.99] = 1.0  # be conservative in what is background
    return target_bin


def preproc_geometry_for_fg_loss(target):
    target_bin = torch.zeros_like(target)
    target_bin[target <= 0.9] = 1.0  # be conservative in what is foreground
    return target_bin

# What are all the loss types:
# BCE
# IoU
# Dice
# l1
# l2
# sum_gradient
# std
# patch_lpips
# rank
