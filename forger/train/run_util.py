# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import contextlib
import gc
import glob
import logging
import os
import re
import shutil
import stat
import sys
import torch

import forger.util.logging

logger = logging.getLogger(__name__)

# TODO(mshugrina): move this to an installable tools repo instead of copying!


def default_device(device_id=0):
    if torch.cuda.is_available():
        torch.cuda.init()  # TODO: needed?
        dev = 'cuda:%d' % device_id
    else:
        dev = 'cpu'
    return torch.device(dev)


def print_memory_diagnostics(device, level=logging.DEBUG):
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                logger.log(level, 'Tensor {} memory {}'.format(type(obj), obj.size()))
        except Exception as e:
            pass

    if torch.cuda.is_available():
        logger.log(level, 'Device {} memory: {}GB'.format(device, torch.cuda.get_device_properties(device).total_memory/1000000000.0))


def reclaim_cuda_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def youngest_file_matching(pattern):
    files = glob.glob(pattern)
    files.sort(key=os.path.getmtime, reverse=True)
    if len(files) == 0:
        return None
    return files[0]


def write_gradients(model, global_its, viz_writer, in_logger):
    for name, param in model.named_parameters():
        if param.requires_grad:
            try:
                viz_writer.add_histogram('grad/%s' % name, param.grad, global_its)  # dL / dweight
            except Exception as e:
                in_logger.error(e)
                in_logger.error('Failed to get grad of {}'.format(name))
                raise e


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _item(val):
    try:
        return val.item()
    except Exception:
        return val


def write_losses(in_losses, tag, global_its, viz_writer):
    for k, v in in_losses.items():
        viz_writer.add_scalar('%s/%s' % (k, tag), _item(v), global_its)


def log_losses(in_losses, tag, ep, its, in_logger):
    info = []
    for k, v in in_losses.items():
        if k != 'loss':
            info.append('%s %0.6f ' % (k, _item(v)))
    in_logger.info('It %s (ep %03d): %s loss %0.6f (%s)' %
                   (('%6d' % its).rjust(8), ep, tag.upper(), _item(in_losses['loss']), ', '.join(info)))

# TODO: double check this works in all cases
@contextlib.contextmanager
def dummy_ctx_manager():
    yield None


class RunHelper(object):
    """
    Sets up a run directory for the experiment, including checkpoints, logging, etc.
    Help with checkpoints, logging, etc, based on default args (see below).
    """
    def __init__(self, args):
        run_dir = args.run_dir
        self.args = args
        self.run_dir = run_dir
        self.log_dir = os.path.join(run_dir, 'logs')
        self.checkpt_dir = os.path.join(run_dir, 'checkpts')
        self.config_dir = os.path.join(run_dir, 'config')
        self.viz_dir = os.path.join(run_dir, 'viz')
        self.eval_dir = os.path.join(run_dir, 'eval')
        self.current_eval_subdir = None
        self.logfile = os.path.join(self.log_dir, 'train_log.txt')
        self.run_count = 0

    def purge(self):
        if os.path.exists(self.run_dir):
            # Note: this is print because this is typically called before logger is set up
            print('Purging past run directory: %s' % self.run_dir)
            shutil.rmtree(self.run_dir)

    def setup(self, purge=False, log_level=logging.INFO):
        if purge:
            self.purge()

        os.makedirs(os.path.abspath(self.run_dir), exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpt_dir, exist_ok=True)
        os.makedirs(self.config_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)

        self.run_count = forger.util.logging.default_log_setup(
            level=log_level, filename=self.logfile, append_count=True)

    def get_checkpt_filename(self, basename, epoch=None, its=None):
        """ Consistent checkpoint naming. """
        if epoch is None and its is None:
            return os.path.join(
                self.checkpt_dir, '%s_run%d_latest.pt' % (basename, self.run_count))
        return os.path.join(
            self.checkpt_dir, '%s_run%d_it%d_e%d.pt' % (basename, self.run_count, its, epoch))

    @staticmethod
    def get_checkpt_components(fname):
        """Extracts iteration number from a checkpoint filename."""
        m = re.match(r'(.*)_run\d+_it(\d+)_e(\d+).pt', os.path.basename(fname))
        if m:
            return m.groups()[0], int(m.groups()[1]), int(m.groups()[2])
        else:
            raise RuntimeError('Cannot get iteration number from checkpoint file %s' % fname)

    @staticmethod
    def get_checkpt_its(fname):
        basename, its, epoch = RunHelper.get_checkpt_components(fname)
        return its

    def init_test_run(self, test_settings_proto_file, iteration, data_basedir=''):
        os.makedirs(self.eval_dir, exist_ok=True)

        subdir = 'test_%d' % iteration
        subdir = os.path.join(self.eval_dir, subdir)
        if os.path.isdir(subdir):
            raise RuntimeError('Evaluation run already exists: %s' % subdir)

        self.current_eval_subdir = subdir
        os.makedirs(self.current_eval_subdir)
        return self.current_eval_subdir

    def save_cmdline(self, args):
        run_filename = os.path.join(self.config_dir, 'run_cmdline%d.sh' % self.run_count)
        with open(run_filename, 'w') as f:
            f.write('#!/bin/sh -e\n')
            f.write('set -o nounset\n')
            f.write('cd %s\n\n' % os.getcwd())
            f.write('# Note: must run as module instead: python -m ...\n')
            f.write(sys.argv[0] + ' \\ \n')
            f.write(' '.join(sys.argv[1:]) + '\n\n')
            f.write('# ALL FLAG VALUES: \n')
            f.write('# %s\n' % str(args))
        st = os.stat(run_filename)
        os.chmod(run_filename, st.st_mode | stat.S_IEXEC)

    def get_latest_checkpoint_filename(self, model_name):
        return youngest_file_matching(os.path.join(self.checkpt_dir, '%s*.pt' % model_name))

    def maybe_load_checkpoint(self, model_name):
        checkpt_name = youngest_file_matching(os.path.join(self.checkpt_dir, '%s*.pt' % model_name))

        if checkpt_name is not None:
            pretrain_its = RunHelper.get_checkpt_its(checkpt_name)
            logger.info('Restoring from latest checkpoint (it=%d): %s' % (pretrain_its, checkpt_name))
            checkpt = torch.load(checkpt_name)
            return checkpt, pretrain_its
        return None, 0

    def prune_checkpoints(self, dry_run=False):
        """
        Prunes checkpoints to only keep the latest version of each model per checkpoint.
        Works if multiple models are saved with different basenames.
        """
        files = glob.glob(os.path.join(self.checkpt_dir, '*.pt'))
        models = {}
        for f in files:
            try:
                basename, its, epoch = RunHelper.get_checkpt_components(f)
            except Exception as e:
                logger.warning('Skipping {} from auto-pruning'.format(f))
                continue
            if basename not in models:
                models[basename] = []
            models[basename].append((epoch, its, f))
        logger.debug('Found %d checkpt files' % len(files))
        logger.debug('Found %d models' % len(models.keys()))

        to_remove = []
        to_keep = []
        for k in models.keys():
            models[k].sort(reverse=True)

            prev_epoch = -1
            for x in models[k]:
                epoch = x[0]
                if epoch != prev_epoch:
                    prev_epoch = epoch
                    to_keep.append(x[2])
                else:
                    to_remove.append(x[2])

        if not dry_run:
            for f in to_remove:
                logger.debug('Deleting: %s' % f)
                os.remove(f)
            logger.info('Pruned %d files from %s' % (len(to_remove), self.checkpt_dir))
        else:
            for f in to_remove:
                logger.debug('Would prune %s' % f)
            logger.info('Would prune %d files from %s' % (len(to_remove), self.checkpt_dir))
            for f in to_keep:
                logger.debug('Would keep %s' % f)
        return to_remove

    def needs_train_log(self, total_its):
        if self.args.train_log_interval == 0:
            return False

        return total_its < 20 or \
               ((total_its % self.args.train_log_interval) == 0)

    def needs_train_viz(self, total_its):
        if self.args.train_viz_interval == 0:
            return False

        return total_its < 5 or \
               (total_its < 100 and (total_its % 10) == 0) or \
               ((total_its % self.args.train_viz_interval) == 0)

    def needs_eval(self, total_its):
        if self.args.eval_interval == 0:
            return False

        return (total_its % self.args.eval_interval) == 0

    def needs_checkpoint(self, total_its):
        return (total_its % self.args.checkpt_interval) == 0

    def process_checkpoint(self,
                           model_name,               # The name of the model to be used in the checkpoint filename
                           args,                     # argparse.Namespace object that contains the parameters of the run
                           epoch,
                           total_its,
                           model_state_dict,         # The state dict of the model we are going to save
                           opt_state_dict=None,      # The state dict of the optimizer we used to train the model
                           ):
        checkpt_filename = self.get_checkpt_filename(model_name, epoch=epoch, its=total_its)
        if os.path.exists(checkpt_filename):
            logger.warning(f"The file {checkpt_filename} already exists and will be overwritten.")

        checkpt_dict = {
            'args': args,
            'epoch': epoch,
            'model_state': model_state_dict,
            'opt_state': opt_state_dict
        }
        torch.save(checkpt_dict, checkpt_filename)
        logger.info('Saved checkpoint: %s' % checkpt_filename)
        if not self.args.no_self_prune_checkpoints:
            self.prune_checkpoints()

    def detect_grad_anomaly_if_debug(self):
        return torch.autograd.detect_anomaly() if self.args.debug else dummy_ctx_manager()


def add_standard_flags(parser, train=False):
    forger.util.logging.add_log_level_flag(parser)
    parser.add_argument(
        '--run_dir', action='store', type=str, required=True,
        help='Directory for logs, checkpoints, etc.')
    parser.add_argument(
        '--data_basedir', action='store', type=str, default='',
        help='If set, will prefix all the loaded dataset paths with this value.')
    parser.add_argument(
        '--device_id', action='store', type=int, default=0,
        help='If running on a CUDA enabled machine, device id to use.')
    parser.add_argument(
        '--batch_size', action='store', type=int, default=20,
        help='Batch size to use for train or eval.')
    parser.add_argument('--num_worker', action='store', type=int,
                        help='Number of worker for one dataloader to use')
    parser.add_argument(
        '--debug', action='store_true',
        help='If set, enable diagnostics and debug features.')

    if train:
        parser.add_argument(
            '--purge', action='store_true', default=False,
            help='If set, will purge run_dir and start from scratch; else will pick '
                 'up where it left off.')
        parser.add_argument(
            '--no_self_prune_checkpoints', action='store_true',
            help='Unless this is passed in, will prune checkpoints after every save to prevent '
            'run directory bloat.')
        parser.add_argument(
            '--checkpt_interval', action='store', type=int, default=2000,
            help='How often (in iterations) to save model checkpoints. '
                 'Set to 0 or below to disable all but the last checkpoint.')
        parser.add_argument(
            '--train_viz_interval', action='store', type=int, default=2000,
            help='How often (in iterations) to visualize train batch output. '
                 'Set to 0 or below to disable.')
        parser.add_argument(
            '--train_log_interval', action='store', type=int, default=200,
            help='How often (in iterations) to visualize train batch output. '
                 'Set to 0 or below to disable.')
        parser.add_argument(
            '--eval_interval', action='store', type=int, default=2000,
            help='How often (in iterations) to evaluate and visualize on the '
                 'evaluation set, if provided in --train_settings_proto. '
                 'Set to 0 or below to disable.')
        parser.add_argument(
            '--epochs', action='store', type=int, default=4,
            help='How many epochs to run for.')
        parser.add_argument(
            '--max_its', action='store', type=int, default=-1,
            help='If set, cap maximum number of iterations, even if --epochs is not reached '
                 '(useful to debug setup with a large dataset).')


def set_up_run(args, train=False):
    helper = RunHelper(args)
    helper.setup(purge=(train and args.purge), log_level=args.log_level)

    helper.save_cmdline(args)
    return helper
