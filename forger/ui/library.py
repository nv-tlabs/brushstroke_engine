# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import logging
import os
import pickle
import re
import torch
import random
import numpy as np
import zipfile
import PIL
import io

import forger.metrics.util

logger = logging.getLogger(__name__)


class ZipIcons(object):
    def __init__(self, path, extension='.jpg'):
        self.path = path
        self.extension = extension
        self.zip = zipfile.ZipFile(path, mode='a')

    def get_icon(self, key):
        fname = key + self.extension
        if fname in self.zip.namelist():
            with self.zip.open(fname, 'r') as f:
                image = np.array(PIL.Image.open(f))
                return image
        return None

    def set_icon(self, key, npimg):
        logger.info(f'Saving icon for style {key} in {self.path}')
        fname = key + self.extension

        image = PIL.Image.fromarray(npimg)
        byte_io = io.BytesIO()
        image.save(byte_io, format="JPEG")
        image.close()
        self.zip.writestr(fname, byte_io.getvalue())


def read_zs(saved_file):
    zs = []
    if not os.path.isfile(saved_file):
        return zs

    zdim = 0
    with open(saved_file) as f:
        for line in f:
            line = line.strip()
            if len(line) > 0 and line[0] != '#':
                try:
                    val = int(line.split()[0])
                    zdim = len(line.split()) - 1
                    zs.append(val)
                except ValueError:
                    logger.error(f'Failed to parse saved seed line {line} from {saved_file}')
    return zs, zdim


def _interp_style_id(style_id1, style_id2, alpha):
    return '%s_%0.2f__%s' % (str(style_id1), alpha, str(style_id2))


class BrushLibrary(object):
    @staticmethod
    def from_arg(arg_val, z_dim=64):
        if os.path.isfile(arg_val):
            return BrushLibrary.from_file(arg_val, z_dim=z_dim)

        m = re.match(r'^rand(\d+)$', arg_val)
        if m is not None:
            try:
                return RandomBrushLibrary(int(m.group(1)), zdim=z_dim)
            except Exception as e:
                logger.error(f'invalid value, expected rand{int}, e.g. rand10, got: {arg_val}')
                raise e

        try:
            values = [int(x) for x in arg_val.split(',')]
        except ValueError as e:
            logger.error('style seeds must be CVS ints, got: {}'.format(arg_val))
            raise e

        if len(values) == 1:
            num_seeds = values[0]
            seeds = list(range(0, max(10000, num_seeds)))
            random.shuffle(seeds)
            return SeedBrushLibrary(seeds[:num_seeds], z_dim)
        else:
            return SeedBrushLibrary(values, z_dim)

    @staticmethod
    def from_file(fname, z_dim=64):
        logger.info('Parsing file {}'.format(fname))
        try:
            res = WBrushLibrary.from_file(fname)
        except Exception as e:
            logger.info(f'Could not load W library, loading seed library from {fname}')
            res = SeedBrushLibrary.from_file(fname, z_dim=z_dim)

        res.set_icon_file(fname + '.icons.zip')
        return res

    def __init__(self):
        self.iconzip = None
        self.mapper = None

    def set_icon_file(self, icon_zipfile):
        self.iconzip = ZipIcons(icon_zipfile)

    def enable_dynamic_icons(self, style_mapper):
        self.mapper = style_mapper

    def get_style_icon(self, style_id):
        if self.iconzip is not None:
            icon = self.iconzip.get_icon(style_id)
            if icon is not None:
                return icon
        if self.mapper is not None:
            opts = forger.ui.brush.GanBrushOptions()
            self.set_style(style_id, opts)
            icon = self.mapper.get_brush_icon(opts)
            if self.iconzip is not None:
                self.iconzip.set_icon(style_id, icon)
            return icon
        return None

    def get_style_ids(self):
        pass

    def set_style(self, style_id, brush_options):
        pass

    def set_interpolated_style(self, style_id1, style_id2, alpha, brush_options):
        pass


class WBrushLibrary(BrushLibrary):
    @staticmethod
    def from_file(fname):
        styles_dict = {}
        if os.path.isfile(fname):
            with open(fname, 'rb') as f:
                # TODO: later support various formats, e.g. more organized style catalog
                styles_dict = pickle.load(f)
                logger.info(f'Loaded w library with {len(styles_dict.keys())} styles: {styles_dict.keys()}')
        return WBrushLibrary(styles_dict)

    def __init__(self, styles_dict):
        super().__init__()
        self.styles = styles_dict

    def get_style_ids(self):
        return sorted(self.styles.keys())

    def set_style(self, style_id, brush_options):
        style_info = self.styles[style_id]
        w = None
        noise = None
        if type(style_info) is dict:
            w = style_info['w']
            if 'noise' in style_info:
                noise = style_info['noise']
            else:
                noise = dict([(k, v) for k, v in style_info.items() if k != 'w'])
                if len(noise) == 0:
                    logger.warning(f'Suspicious; no noise found for dictionary style with id {style_id}')
                    noise = None
        else:
            w = self.styles[style_id]

        # TODO: remove once no longer needed and no legacy projections
        if noise is not None:
            for k, v in noise.items():
                if not torch.is_tensor(v):
                    noise[k] = torch.from_numpy(v)

        brush_options.set_style_w(w, style_id=style_id, custom_args={'noise_buffers': noise})

    def set_interpolated_style(self, style_id1, style_id2, alpha, brush_options):
        opts1 = forger.ui.brush.GanBrushOptions()
        opts2 = forger.ui.brush.GanBrushOptions()
        self.set_style(style_id1, opts1)
        self.set_style(style_id2, opts2)

        w = opts1.style_ws * alpha + opts2.style_ws * (1 - alpha)
        custom_args = None
        if 'noise_buffers' in opts1.custom_args and 'noise_buffers' in opts2.custom_args:
            noise = {}
            for k, v in opts1.custom_args['noise_buffers'].items():
                noise[k] = v * alpha + opts2.custom_args['noise_buffers'][k] * (1 - alpha)
            custom_args = {'noise_buffers': noise}

        brush_options.set_style_w(w, style_id=_interp_style_id(style_id1, style_id2, alpha), custom_args=custom_args)


class SeedBrushLibrary(BrushLibrary):
    @staticmethod
    def from_file(fname, z_dim=None):
        zs, zdim = read_zs(fname)
        if z_dim is not None:
            zdim = z_dim
        logger.info(f'Loaded seed library with {len(zs)} styles: {zs}')
        return SeedBrushLibrary(zs, zdim)

    def __init__(self, seeds_list, zdim):
        super().__init__()
        self.zs = seeds_list
        self.zdim = zdim

    def get_style_ids(self):
        return sorted([str(x) for x in self.zs])

    def set_style(self, style_id, brush_options):
        seed = int(style_id)
        brush_options.set_style(torch.from_numpy(np.random.RandomState(seed=seed).randn(1, self.zdim)),
                                style_id=style_id)

    def set_interpolated_style(self, style_id1, style_id2, alpha, brush_options):
        opts1 = forger.ui.brush.GanBrushOptions()
        opts2 = forger.ui.brush.GanBrushOptions()
        self.set_style(style_id1, opts1)
        self.set_style(style_id2, opts2)

        z = opts1.style_z * alpha + opts2.style_z * (1 - alpha)
        brush_options.set_style(z, style_id=_interp_style_id(style_id1, style_id2, alpha))


class RandomBrushLibrary(BrushLibrary):
    def __init__(self, num, zdim, random_state=None):
        super().__init__()
        self.num = num
        self.zdim = zdim
        self.random_state = random_state if random_state is not None else forger.metrics.util.RandomState(0)

    def get_style_ids(self):
        return ['rand' + str(x) for x in range(self.num)]

    def set_style(self, style_id, brush_options):
        brush_options.set_style(self.random_state.random_tensor((1, self.zdim)))

    def set_interpolated_style(self, style_id1, style_id2, alpha, brush_options):
        self.set_style(style_id1, brush_options)



