# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import json
import logging
import numpy as np
from skimage.io import imsave

from tornado.websocket import WebSocketHandler
import tornado.gen

from forger.ui.brush import PaintingHelper
from forger.util.logging import log_tensor

logger = logging.getLogger(__name__)


def int32_to_binary(single_int):
    return np.array([single_int], dtype=np.int32).tobytes()


def image_patch_to_binary(img, x, y):
    """Converts an image patch to binary string.
    Must be compatible with javascript: nvidia.Controller.prototype.decodeDrawingResponse.

    Args:
        img: H x W x 4 np array
        x: metadata to encode
        y: metadata to encode

    Returns: bytes encoding
    """
    if img.dtype != np.uint8:
        raise RuntimeError('Image must be uint8 in range 0...255')

    height = img.shape[0]
    width = img.shape[1]
    nchannels = img.shape[2]
    assert nchannels < height, f'Wrong shape {img.shape}'

    binstr = np.array([width, height, x, y], dtype=np.int32).tobytes()
    binstr += img.tobytes()
    return binstr


def binary_to_image_patches(bytes_msg, offset=0):
    """Converts bytes message to image patches.
    Must be compatible with: nvidia.Controller.prototype.encodeDrawingRequest.

    Args:
        @param bytes_msg: raw bytes to decode
        @param offset: start read offset in bytes

    Return:
        metadata dict, stroke_image, current_canvas_image
    """
    metadata_length = 5
    metadata = np.frombuffer(bytes_msg, dtype=np.int32, count=metadata_length, offset=offset)
    meta = {'width': metadata[0],
            'height': metadata[1],
            'x': metadata[2],
            'y': metadata[3],
            'crop_margin': metadata[4]}   # Margin to crop off from the output
    logger.debug(f'Decoded meta {meta}')
    img_data = np.frombuffer(bytes_msg, dtype=np.uint8, offset=offset + metadata_length * 4)
    log_tensor(img_data, 'decoded array', logger)
    imgsize = meta['height'] * meta['width'] * 4
    img_stroke = img_data[0:imgsize].reshape((meta['height'], meta['width'], 4))

    # Note: in current implementation we do not need the canvas
    img_canvas = None  # img_data[imgsize:].reshape((meta['height'], meta['width'], 4))
    return meta, img_stroke, img_canvas


def decode_render_request_metadata(bytes_msg, offset=0):
    """
    Decodes metadata associated with the render request from the client.

    Expected layout (uint8):
    uint8:
    [0] - 0/1 is debug
    [1] - 0/1/2/3 number of custom colors
    [2] - int encoding any extra data
    4 * number colors uints for primary/secondary colors if included with coloridx, R, G, B

    @param offset: read start offset in bytes
    @param bytes_msg:
    @return: dict, next_read_offset
    """
    metadata_length = 3
    metadata = np.frombuffer(bytes_msg, dtype=np.uint8, count=metadata_length, offset=offset)
    read_start = offset + metadata_length
    meta = {'debug': metadata[0] != 0,
            'colors': [],
            'extra_data': metadata[2]}
    num_colors = metadata[1]
    for c in range(num_colors):
        meta['colors'].append(np.frombuffer(bytes_msg, dtype=np.uint8, count=4, offset=read_start))
        read_start = read_start + 4
    return meta, read_start


class DrawingWebSocketHandler(WebSocketHandler):
    """ Handles websocket communication with the JS client.
    """

    def initialize(self, paint_engine, style_seed, debug_dir, saved_zs_filename=None, libraries=None):
        """ Takes TBD helper type that can actually run the model."""
        # Note: this is correct, __init__ method should not be written for this
        self.helper = PaintingHelper(paint_engine, style_seed=style_seed, debug_dir=debug_dir)
        self.zs_file = saved_zs_filename
        self.libraries = libraries
        self.use_positions = False
        self.uvs_mapping = False

    def open(self):
        """ Open socket connection and send information about available geometry."""
        logger.debug("Socket opened.")
        message = {"type": "modelinfo",
                   "data": {"patch_width": self.helper.engine.patch_width}}
        self.write_message(message, binary=False)
        self.send_current_brush_info()

    @tornado.gen.coroutine
    def send_current_brush_info(self):
        message = {"type": "brushinfo",
                   "data": {"style_id": "%s" % str(self.helper.brush_options.style_id),
                            "library_id": "%s" % self.helper.brush_options.library_id,
                            "colors": "%s" % self.helper.engine.uvs_mapper.get_colors(self.helper.brush_options)}}
        self.write_message(message, binary=False)

    @tornado.gen.coroutine
    def save_current_brush(self):
        if self.zs_file is None or self.helper.brush_options.style_id is None:
            return
        try:
            with open(self.zs_file, 'a') as f:
                f.write(('%d ' % self.helper.brush_options.style_id) +
                        ' '.join(['%f' % x for x in self.helper.brush_options.style_z[0, ...].tolist()]) +
                        '\n')
        except RuntimeError as e:
            logger.warning('Failed to save z')

    @tornado.gen.coroutine
    def on_message(self, message):
        """ Handles new messages on the socket."""
        logger.debug('Received message of type {}'.format(type(message)))  #, message))

        try:
            if type(message) == bytes:
                self._handle_binary_request(message)
            else:
                self._handle_json_request(message)
        except Exception as e:
            logger.error('Failed to decode incoming message: {}'.format(e))

    def _encode_type_render(self, extra_data):
        if extra_data == 0:
            return int32_to_binary(0)
        else:
            return int32_to_binary(extra_data)

    def _encode_type_debug_img(self):
        return int32_to_binary(1)

    def _encode_type_brush_sample(self):
        return int32_to_binary(2)


    @tornado.gen.coroutine
    def _handle_image_request(self, meta, bg_img, fg_img):
        brush_options = self.helper.default_brush_options()
        for colorinfo in meta['colors']:
            cidx = colorinfo[0]
            brush_options.set_color(cidx, colorinfo[1:])
        brush_options.debug = meta['debug']
        if self.use_positions:
            brush_options.set_position(int(meta['x']), int(meta['y']))
        else:
            brush_options.position = None

        brush_options.enable_uvs_mapping = self.uvs_mapping

        res_img, debug_img, meta_out = self.helper.render_stroke(bg_img, fg_img, brush_options, meta)
        bin_str = self._encode_type_render(meta['extra_data']) + image_patch_to_binary(res_img, meta_out['x'], meta_out['y'])
        self.write_message(bin_str, binary=True)

        # Also send debug image
        if debug_img is not None:
            bin_str = self._encode_type_debug_img() + image_patch_to_binary(debug_img, 0, 0)
            self.write_message(bin_str, binary=True)

    @tornado.gen.coroutine
    def _handle_binary_request(self, raw_message):
        logger.debug('Decoding binary message')
        meta, read_offset = decode_render_request_metadata(raw_message)
        # logger.debug(f'Decoded meta {meta}')
        patch_meta, img_stroke, img_canvas = binary_to_image_patches(raw_message, read_offset)
        # logger.debug(f'Decoded patch meta {patch_meta}')
        meta.update(patch_meta)
        # log_tensor(img_stroke, 'decoded image (stroke)', logger)
        self._handle_image_request(meta, img_stroke, img_canvas)

    def _handle_set_option(self, msg):
        if msg.get('option') == 'positions':
            self.use_positions = msg.get('value')
            logger.info(f'Set use_positions to {self.use_positions}')
        elif msg.get('option') == 'uvs_mapping':
            self.uvs_mapping = msg.get('value')

    @tornado.gen.coroutine
    def _handle_json_request(self, raw_message):
        logger.debug('Decoding string message')
        msg = json.loads(raw_message)

        if msg.get('type') == 'set_brush':
            if msg.get('style_id') and msg.get('library_id'):
                library_id = msg.get('library_id')
                style_id = msg.get('style_id')
                if library_id in self.libraries and style_id in self.libraries[library_id].get_style_ids():
                    self.libraries[library_id].set_style(style_id, self.helper.brush_options)
                    self.helper.brush_options.library_id = library_id
            else:
                self.helper.set_new_brush(msg.get('seed'))
            self.send_current_brush_info()
        elif msg.get('type') == 'save_brush':
            self.save_current_brush()
        elif msg.get('type') == 'set_option':
            self._handle_set_option(msg)
        elif msg.get('type') == 'set_render_mode':
            self.helper.set_render_mode(msg.get('mode'))
        elif msg.get('type') == 'new_canvas':
            print(msg)
            self.helper.make_new_canvas(int(msg.get('rows')), int(msg.get('cols')),
                                        feature_blending=int(msg.get('feature_blending')))
        else:
            logger.warning('Received unknown json message of type {}: {}'.format(
                msg.get('type'), msg))

    def on_close(self):
        logger.info("Socket closed.")