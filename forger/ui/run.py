# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
from __future__ import print_function

import argparse
import logging
import os
import sys
import warnings
import io

import torch
import flask
from flask import Flask, render_template, Response
from tornado.wsgi import WSGIContainer
from tornado.web import Application, FallbackHandler
from tornado.ioloop import IOLoop
import PIL
import numpy as np
import re
import random

from forger.ui.util import DrawingWebSocketHandler
import forger.ui.library
import forger.ui.util
import forger.ui.brush

import forger.util

logger = logging.getLogger(__name__)


def generate_z_file(gan_checkpoint):
    return os.path.splitext(gan_checkpoint)[0] + '.saved_zs.txt'


# # TODO: swap to using library.py
# def read_zs(saved_file):
#     warnings.warn("deprecated; use BrushLibrary", category=DeprecationWarning)
#     zs = []
#     if not os.path.isfile(saved_file):
#         return zs
#
#     with open(saved_file) as f:
#         for line in f.readlines():
#             try:
#                 zs.append(int(line.split()[0]))
#             except Exception as e:
#                 logger.error(f'Failed to parse saved seed line {line} from {saved_file}')
#     return zs


def create_server(encoder_checkpoint,
                  gan_checkpoint,
                  debug_dir=None,
                  style_seed=None,
                  enable_z_saving=False,
                  library_specs=None):
    """ Creates HTTP & websocket helper. """
    engine = forger.ui.brush.PaintEngineFactory.create(
        encoder_checkpoint=encoder_checkpoint,
        gan_checkpoint=gan_checkpoint,
        device=torch.device(0)
    )

    z_file = generate_z_file(gan_checkpoint)
    logger.info(f'Using saved zs file {z_file}')

    libraries = {}
    for spec in library_specs:
        spec_name = spec[0]
        spec_mode = spec[1]
        spec_path = spec[2]
        if spec_path == 'default':
            spec_path = z_file
        lib = forger.ui.library.BrushLibrary.from_file(spec_path, z_dim=engine.G.z_dim)
        lib.enable_dynamic_icons(engine.uvs_mapper)  # Mapper should be renamed
        libraries[spec_name] = lib

    # Flask for HTTP
    _base_dir = os.path.dirname(__file__)
    _template_dir = os.path.join(_base_dir, 'templates')
    _static_dir = os.path.join(_base_dir, 'static')
    app = Flask('art_forger',
                template_folder=_template_dir,
                static_url_path='/static',
                static_folder=_static_dir)

    @app.route('/')
    def index():
        library_infos = {}
        for spec in library_specs:
            spec_name = spec[0]
            spec_mode = spec[1]
            lib = libraries[spec_name]
            brushes = list(lib.get_style_ids())
            m = re.match(r'rand(\d+)', spec_mode)
            if m is not None:
                try:
                    num_items = int(m.group(1))
                    random.shuffle(brushes)
                    brushes = brushes[:num_items]
                except Exception as e:
                    logger.warning(f'Malformed lib spec? {spec_mode}')
            library_infos[spec_name] = {'brushes': brushes}

        return render_template('home.html', subtitle=engine.summary(),
                               canvas_width=flask.request.args.get('canvas', 2000),
                               demo=('demo' in flask.request.args),
                               library_infos=library_infos)

    @app.route('/brush/<library_name>/<brush_name>.jpg')
    def brush_image(library_name, brush_name):
        # TODO: this gotta be async
        if library_name in libraries:
            image = libraries[library_name].get_style_icon(brush_name)
        else:
            image = np.zeros((128, 128, 3), dtype=np.uint8)
        image = PIL.Image.fromarray(image)
        byte_io = io.BytesIO()
        image.save(byte_io, format="JPEG")
        jpg_buffer = byte_io.getvalue()
        byte_io.close()
        response = Response(jpg_buffer, mimetype='image/jpeg')
        return response

    # Tornado server to handle websockets
    container = WSGIContainer(app)
    server = Application([
        (r'/websocket/',
         DrawingWebSocketHandler, dict(paint_engine=engine, style_seed=style_seed, debug_dir=debug_dir,
                                       saved_zs_filename=(z_file if enable_z_saving else None),
                                       libraries=libraries)),
        (r'.*',
         FallbackHandler, dict(fallback=container))
    ])
    return server


def parse_libraries(libraries_arg):
    libraries = []
    if libraries_arg is not None and len(libraries_arg) > 0:
        libraries = [x.split(':') for x in libraries_arg.split(',')]
    for i in range(len(libraries)):
        if len(libraries[i]) == 1:
            libraries[i] = [ os.path.basename(libraries[i][0]), 'disp', libraries[i][0] ]
        elif len(libraries[i]) == 2:
            libraries[i] = [libraries[i][0], 'disp', libraries[i][1]]
        assert len(libraries[i]) == 3, f'Malformed library spec {libraries[i]}'
        assert libraries[i][1] in ['disp', 'random'] or re.match(r'rand\d+', libraries[i][1])
    return libraries


def run_main():
    aparser = argparse.ArgumentParser(description='ArtForger user interface.')
    aparser.add_argument('--gan_checkpoint', action='store', type=str, required=True)
    aparser.add_argument('--encoder_checkpoint', action='store', type=str, default=None)
    aparser.add_argument('--port', action='store', default=8000)
    aparser.add_argument('--debug_dir', type=str, default=None,
                         help='The directory to save the debug outputs.')
    aparser.add_argument('--style_seed', type=int, default=None, help='Set for random style determinism')
    aparser.add_argument('--disable_z_saving', action='store_true')
    aparser.add_argument('--libraries', action='store', type=str, default='Default:random:default',
                         help='Paths to library files with known styles, specified as: '
                         'name:mode:path, where name will be displayed and mode is "random" or "disp".')
    forger.util.logging.add_log_level_flag(aparser)
    args = aparser.parse_args()

    forger.util.logging.default_log_setup(args.log_level)

    server = create_server(encoder_checkpoint=args.encoder_checkpoint,
                           gan_checkpoint=args.gan_checkpoint,
                           debug_dir=args.debug_dir,
                           style_seed=args.style_seed,
                           enable_z_saving=(not args.disable_z_saving),
                           library_specs=parse_libraries(args.libraries))
    server.listen(args.port)
    IOLoop.instance().start()


if __name__ == "__main__":
    run_main()

# python -m forger.ui.run --port=8000 --log_level=20 --encoder_checkpoint=/mnt/tobby2/Experiments/GANtool/cy/checkpoints/test0/encoder.pt --gan_checkpoint=/mnt/tobby2/Experiments/GANtool/forger2/warmtrain0/00003-run04_gint100_last_and_rgb_st100-styles1-triad-auto1-glr0.0002-dlr0.00015-zw64/network-snapshot-010000.pkl --debug_dir=/tmp/gan_brush