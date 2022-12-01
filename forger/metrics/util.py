# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import logging
import numpy as np
import os
import re
import torch
import torch.utils.data
import warnings

from forger.util.torch_data import get_image_data_iterator

import forger.ui.brush

logger = logging.getLogger(__name__)


def make_white_canvas(width):
    return torch.ones([1, 3, width, width], dtype=torch.float32)


def style_seeds_from_flag(style_seeds, gan_checkpoint, random_state):
    use_range = False
    if style_seeds == 'default':
        style_seeds = gan_checkpoint.replace('.pkl', '.saved_zs.txt')
        logger.info(f'Using curated seeds in {style_seeds}')
        if not os.path.isfile(style_seeds):
            raise RuntimeError(f'Failed to get default seeds in {style_seeds}')
    elif re.match('default([0-9]+)', style_seeds) is not None:
        use_range = True
        style_seeds = style_seeds.replace('default', '')

    style_seeds = parse_style_seeds(style_seeds)
    if type(style_seeds) is int:  # if only number of seeds provided, create random
        if style_seeds > random_state.max_style_seed or use_range:
            logger.info(f'Using deterministic range style seeds for {style_seeds}')
            style_seeds = range(style_seeds)
        else:
            style_seeds = list(set([random_state.generate_style_seed() for _ in range(style_seeds)]))
    return style_seeds


# TODO: use BrushLibrary
def parse_style_seeds(style_seeds):
    warnings.warn("parse_style_seeds deprecated; use BrushLibrary", category=DeprecationWarning)

    if os.path.isfile(style_seeds):
        logger.info('Parsing file {}'.format(style_seeds))
        result = []
        with open(style_seeds) as f:
            for line in f:
                line = line.strip()
                if len(line) > 0 and line[0] != '#':
                    try:
                        val = int(line.split()[0])
                        result.append(val)
                    except ValueError:
                        logger.warning('Line does not start with seed: {}'.format(line))
        return result
    else:
        try:
            values = [int(x) for x in style_seeds.split(',')]
        except ValueError as e:
            logger.error('style seeds must be CVS ints, got: {}'.format(style_seeds))
            raise e

        if len(values) == 1:
            return values[0]
        return values


class RandomState(object):
    def __init__(self, seed, max_style_seed=10000):
        self.seed_rng = np.random.default_rng(seed=seed)
        self.tgenerator = torch.Generator()
        self.max_style_seed = max_style_seed
        if seed is not None:
            self.tgenerator.manual_seed(seed + 1)

    def generate_style_seed(self):
        return self.seed_rng.integers(low=0, high=self.max_style_seed, size=1)[0]

    def random_tensor(self, shape, dtype=torch.float32):
        return torch.rand(shape, dtype=dtype, generator=self.tgenerator)

    def generate_style_seeds(self, num):
        """
        Note: may return fewer than num if duplicate samples are encountered.

        @param num:
        @return:
        """
        if num > self.max_style_seed:
            res = list(range(num))
        else:
            res = list(set([self.generate_style_seed() for _ in range(num)]))
        return res


class RandomStyleGenerator:
    @staticmethod
    def create_from_seeds(seeds, generator):
        return RandomStyleGenerator(generator, seeds=seeds)

    @staticmethod
    def create_without_seeds(num, generator):
        return RandomStyleGenerator(generator, num=num)

    def __init__(self, generator, seeds=None, num=None):
        """

        @param seeds: list of seeds
        """
        warnings.warn("RandomStyleGenerator is deprecated; use BrushLibrary", category=DeprecationWarning)
        assert (seeds is None) != (num is None), "must set seeds or num"
        self.generator = generator
        self.seeds = seeds
        self.num = len(seeds) if seeds is not None else num
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.idx >= self.num:
            raise StopIteration()

        seed = None
        if self.seeds is not None:
            seed = self.seeds[self.idx]
        self.idx += 1

        return seed, self.generator.get_random_style(seed)


class PaintStrokeGenerator(object):
    @staticmethod
    def create(encoder_checkpoint: str, gan_checkpoint: str, batch_size: int,
               device, random_state=None, seed=None):
        paint_engine = forger.ui.brush.PaintEngineFactory.create(
            encoder_checkpoint=encoder_checkpoint,
            gan_checkpoint=gan_checkpoint,
            device=device)
        if random_state is None:
            random_state = RandomState(seed)
        return PaintStrokeGenerator(batch_size, paint_engine, random_state)

    def __init__(self, batch_size: int, paint_engine,  # forger.ui.brush.GanPaintEngine
                 random_state: RandomState, primary_color_idx=0):
        super().__init__()
        self.random_state = random_state
        self.batch_size = batch_size
        self.geom_iter = None
        self.engine = paint_engine
        self.brush_options = forger.ui.brush.GanBrushOptions()
        self.brush_options.debug = False
        self.geom = None
        self.geom_truth = None
        self.primary_color_idx = primary_color_idx
        self.gic = 1  # geom input channel
        self.gtc = 2

    def set_render_mode(self, mode):
        self.engine.set_render_mode(mode)

    def set_geometry_source(self, geom_data_path: str, batch_size, shuffle=True, geom_input_channel=1):
        self.geom_iter, self.batch_size = get_image_data_iterator(
            geom_data_path, batch_size, shuffle=shuffle, return_batch_size=True)
        self.gic = geom_input_channel

    def set_geometry_source_from_iterator(self, iterator, batch_size, geom_input_channel=1, geom_truth_channel=2):
        self.geom_iter = iterator
        self.batch_size = batch_size
        self.gic = geom_input_channel
        self.gtc = geom_truth_channel

    def random_colors(self):
        return self.random_state.random_tensor((self.batch_size, 3)).to(self.engine.device)

    def random_color(self):
        return self.random_state.random_tensor((1, 3)).to(self.engine.device).expand(self.batch_size, -1)

    def set_random_colors(self):
        for i in range(3):
            self.set_new_color(i, self.random_colors())

    def unset_colors(self):
        for i in range(3):
            self.brush_options.set_color(i, None)

    def set_new_geom(self, geom=None):
        """

        @param geom: B x 1 x W x W float32 torch tensor [0...1] with 0 == foreground
        @return:
        """
        if geom is not None:
            assert geom.shape[0] == self.batch_size
            self.geom = geom
        else:
            assert self.geom_iter is not None
            geom, _ = next(self.geom_iter)
            self.geom = geom[:, self.gic:self.gic+1, ...].to(torch.float32).to(self.engine.device) / 255.0
            self.geom_truth = geom[:, self.gtc:self.gtc+1, ...].to(torch.float32).to(self.engine.device) / 255.0

    def set_new_primary_color(self, colors=None):
        self.set_new_color(self.primary_color_idx, colors)

    def set_new_color(self, color_idx, colors=None):
        assert 0 <= color_idx < 3
        if colors is not None:
            assert colors.shape[0] == self.batch_size
            assert colors.shape[1] == 3
        self.brush_options.set_color(color_idx, colors)

    def get_random_styles(self, seeds=None, return_seeds=False):
        """
        Gets different random styles for elements of batch.

        @param seeds:
        @param return_seeds:
        @return:
        """
        if seeds is not None:
            assert type(seeds) == list and len(seeds) == self.batch_size, "Wrong seeds {}".format(seeds)
        elif return_seeds:
            seeds = [self.random_state.generate_style_seed() for _ in range(self.batch_size)]

        if seeds is not None:
            styles = torch.cat([self.engine.random_style(seed) for seed in seeds], dim=0)
        else:
            styles = self.random_state.random_tensor((self.batch_size, self.engine.G.z_dim)).to(self.engine.device)

        if return_seeds:
            return styles, seeds
        return styles

    def get_random_style(self, seed=None, return_seed=False):
        """
        Gets single random style for the entire batch.
        @param seed:
        @param return_seed:
        @return:
        """
        if seed is None and not return_seed:
            return self.random_state.random_tensor((1, self.engine.G.z_dim)).to(self.engine.device).expand(self.batch_size, -1)
        if seed is None:
            seed = self.random_state.generate_style_seed()

        seeds = [seed for _ in range(self.batch_size)]
        style = self.get_random_styles(seeds=seeds, return_seeds=False)

        if return_seed:
            return style, seed
        return style

    def set_new_styles(self, style_z):
        """
        @param style_z: [B x zdim] float32 tensor
        """
        assert style_z.shape == (self.batch_size, self.engine.G.z_dim), "wrong z shape"
        self.brush_options.set_style(style_z, -1)

    def current_styles(self):
        return self.brush_options.style_z

    def generate_raw(self):
        assert self.geom is not None, 'Must call set_new_geom first'
        render, raw, _ = self.engine._render_stroke_torch(self.geom, None, self.brush_options)
        return render, raw

    def generate(self, rgb_on_white_canvas=False):
        """
        Generate synthetic strokes, given options set using set_new_*
        @return: B x 4 x W x W rendering or B x 3 x W x W if rgb_on_white_canvas
        """
        assert self.geom is not None, 'Must call set_new_geom first'
        self.brush_options.prepare_style(self.batch_size, self.engine.device)
        render, _, _ = self.engine._render_stroke_torch(self.geom, None, self.brush_options)

        if rgb_on_white_canvas:
            alpha = render[:, 3, ...].unsqueeze(1)
            return alpha * render[:, :3, ...] + (1 - alpha) * 1.0
        return render
