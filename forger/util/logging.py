# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import glob
import logging
import numpy as np
import os
import re
import sys
import torch

logger = logging.getLogger(__name__)


def default_log_setup(level=logging.INFO, filename=None, append_count=False):
    """
    Sets up default logging, always logging to stdout as well as
    file, if file is specified. If append_count, will search for
    filename with count and create a unique filename with the next
    int count before the extension.
    E.g., repeated calls to:
      defaultLoggingSetup(filename='log.txt', append_count=True)
    will result in files:
      log00.txt
      log01.txt
      log02.txt ...

    :param level: logging level, e.g. logging.INFO
    :param filename: if to log to file in addition to stdout, set filename
    :param append_count: see above
    :return: count, if append_count, else None
    """
    handlers = [logging.StreamHandler(sys.stdout)]
    count = None
    if filename is not None:
        if append_count:
            count = 0
            directory = os.path.dirname(filename)
            basename, extension = os.path.splitext(os.path.basename(filename))
            pattern = re.compile(r'%s(\d+)%s' % (basename, extension))
            fnames = [os.path.basename(x) for x in
                      glob.glob(os.path.join(directory, '%s[0-9]*%s' % (basename, extension)))]
            counts = [int(m.groups()[0]) for m in [pattern.match(f) for f in fnames] if m is not None]
            if len(counts) > 0:
                count = max(counts) + 1
            filename = os.path.join(directory, '%s%02d%s' % (basename, count, extension))
        handlers.append(logging.FileHandler(filename))
    logging.basicConfig(level=level,
                        format='%(asctime)s|%(levelname)8s|%(name)15s| %(message)s',
                        handlers=handlers)
    logger.info('Logging to stdout and %s' % filename)
    logging.getLogger('PIL.PngImagePlugin').setLevel(20)
    return count


def add_log_level_flag(parser):
    parser.add_argument(
        '--log_level', action='store', type=int, default=logging.INFO,
        help='Logging level to use globally, DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40.')


def log_tensor(t, name, use_logger, level=logging.DEBUG, print_stats=False, detailed=False):
    if level < use_logger.level:
        pass

    def _get_stats_str():
        if type(t) == np.ndarray:
            return ' - [min %0.4f, max %0.4f, mean %0.4f]' % (np.min(t), np.max(t), np.mean(t))
        elif torch.is_tensor(t):
            return ' - [min %0.4f, max %0.4f, mean %0.4f]' % (torch.min(t).item(),
                                                              torch.max(t).item(),
                                                              torch.mean(t.to(torch.float32)).item())
        else:
            raise RuntimeError('Not implemented for {}'.format(type(t)))

    def _get_details_str():
        if torch.is_tensor(t):
            return ' - req_grad={}, is_leaf={}, device={}, layout={}'.format(
                t.requires_grad, t.is_leaf, t.device, t.layout)

    if t is None:
        use_logger.log(level, '%s: None' % name)
        return

    shape_str = ''
    if hasattr(t, 'shape'):
        shape_str = '%s ' % str(t.shape)

    if hasattr(t, 'dtype'):
        type_str = '%s' % str(t.dtype)
    else:
        type_str = '{}'.format(type(t))

    use_logger.log(level, '%s: %s(%s) %s %s' %
                   (name, shape_str, type_str,
                    (_get_stats_str() if print_stats else ''),
                    (_get_details_str() if detailed else '')))


def log_tensor_dict(d, name, use_logger, level=logging.DEBUG):
    use_logger.log(level, 'Tensor dict %s' % name)
    for k, v in d.items():
        log_tensor(v, str(k), use_logger, level=level)