# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import argparse
import logging
import os
import tensorboardX
import torch
import torchvision
from torch.utils.data import DataLoader
import wandb
import random

import forger.train.run_util as run_util
import forger.experimental.autoenc.factory as factory
from forger.experimental.autoenc.ae_conv import add_model_flags as ae_conv_add_model_flags
from forger.experimental.autoenc.simple_autoencoder import add_model_flags as sa_add_model_flags
from forger.util.logging import log_tensor

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the autoencoder")
    parser.add_argument(
        '--model_name', action='store', type=str, default='sauto', choices=('sauto', 'conv'),
        help='Type of models: sauto - simple_autoencoder | conv - ae_conv .')
    run_util.add_standard_flags(parser, train=True)
    args, _ = parser.parse_known_args()
    run_helper = run_util.set_up_run(args, train=True)
    checkpt, total_its = run_helper.maybe_load_checkpoint(args.model_name)
    if checkpt is None:
        logger.info(f"No previous checkpoint is found in {args.run_dir}")

    args_training = None
    if checkpt is None:
        logger.info(f'Train from scratch')
        if args.model_name == 'sauto':
            sa_add_model_flags(parser)
        elif args.model_name == 'conv':
            ae_conv_add_model_flags(parser)
        else:
            raise RuntimeError(f'Unknown model: {args.model_name}')
        parser.add_argument(
            '--encoder_in_channels', action='store', type=int, default=1,
            help='Encoder input channels.'
        )
        parser.add_argument(
            '--decoder_out_channels', action='store', type=int, default=1,
            help='Final decoder output channels.'
        )
        parser.add_argument(
            '--widths', action='store', type=str, default='256,128,64',
            help='Training widths to use.'
        )
        parser.add_argument(
            '--lr', action='store', type=float, default=0.0001
        )
        parser.add_argument(
            '--preproc_type', action='store', type=str, default=None,
            help='Type of normalization or preprocessing to use, supports what BaseGeoEncoder.set_preprocessing supports.')
        args_training, _ = parser.parse_known_args()
    else:
        args_training = checkpt['args']

    parser.add_argument('--wandb_project', help='The name of the wandb project', type=str, default=None)
    parser.add_argument(
        '--wandb_experiment_group', action='store', type=str, default=None,
        help='If set, will also log to wandb; experiment name is the run_dir basename.')
    parser.add_argument(
        '--train_images', action='store', type=str, required=True)
    parser.add_argument(
        '--eval_images', action='store', type=str, default=None)
    parser.add_argument(
        '--triband_input', action='store_true',
        help='If set expects R-grayscale image, G-binary image, B-fg black, bg white, neither is gray.')
    parser.add_argument(
        '--balanced_loss', action='store_true',
        help='If true, balances loss weights.')
    parser.add_argument(
        '--exact_loss_with_triband_input', action='store_true',
        help='If set, will not use B for loss, but G as in regular training.')

    args, _ = parser.parse_known_args()
    device = run_util.default_device(args.device_id)

    # Initialize model
    model, model_summary = factory.create_autoencoder(args_training)
    print(model)
    num_parameters = run_util.count_parameters(model)
    print('Trainable parameters: {}'.format(num_parameters))

    if total_its == 0:
        model.apply(factory.weight_init)
    else:
        logger.info('Resume training')
        model.load_state_dict(checkpt['model_state'])

    model.to(device)
    train_widths = [int(x) for x in args_training.widths.split(',') if len(x) > 0]
    # Load training and validation data
    #torchvision.transforms.Resize(args_training.width)
    assert len(train_widths) > 0, 'Require at least one training width'
    random_crops = [torchvision.transforms.RandomCrop(w) for w in train_widths]

    def _get_random_crop():
        return random_crops[random.randint(0, len(random_crops) - 1)]

    transforms = []
    if not args.triband_input:
        transforms.append(torchvision.transforms.Grayscale())
    transforms.append(torchvision.transforms.RandomCrop(max(*train_widths)))
    transforms.append(torchvision.transforms.ToTensor())
    transforms = torchvision.transforms.Compose(transforms)

    def _make_loader(path, shuffle=False):
        dataset = torchvision.datasets.ImageFolder(
            os.path.join(args.data_basedir, path), transform=transforms)
        return DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=3,
            shuffle=shuffle,
            prefetch_factor=2, persistent_workers=True)
    train_loader = _make_loader(args.train_images, shuffle=True)
    eval_loader = _make_loader(args.eval_images) if args.eval_images is not None else None
    # Set up criteria
    criteria = {}

    # torch.nn.L1Loss()

    def _prep_2channel_truth(x):
        return torch.cat([x, 1 - x], dim=1)  # BG, FG

    prep_truth = lambda x: x
    reduction = 'mean'
    if args.balanced_loss:
        reduction = 'none'
    if args_training.decoder_out_channels == 1:
        criteria['BCE'] = (torch.nn.BCEWithLogitsLoss(reduction=reduction), 1.0, False)  # Name: (criterion, weight, preprocess_output)
    elif args_training.decoder_out_channels == 3:
        criteria['BCE'] = (torch.nn.BCELoss(reduction=reduction), 1.0, False)
        prep_truth = _prep_2channel_truth

    eval_criteria = {}
    eval_criteria['BCE_eval'] = (torch.nn.BCELoss(), 1.0, True)
    eval_criteria.update(criteria)

    # Set up optimizer
    opt = torch.optim.Adam(model.parameters(), lr=args_training.lr, eps=1e-07)
    if checkpt is not None:
        opt.load_state_dict(checkpt['opt_state'])

    # TODO: can add patch-GAN loss with these
    discriminator = None
    disc_criterion = None
    disc_opt = None

    # Initialize wandb
    if args.wandb_experiment_group is not None:
        model_summary.update({
            "architecture": args_training.model_name,
            "batch_size": args.batch_size,
            "width": args_training.width,
            "losses": ",".join(criteria.keys()),
            "num_parameters": num_parameters,
            "init_lr": args_training.lr
        })
        wandb_dir = os.path.join(args.run_dir, 'wandb')
        os.makedirs(wandb_dir, exist_ok=True)
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_experiment_group,
            name=os.path.basename(run_helper.run_dir),
            config=model_summary,
            reinit=args.purge,
            resume=total_its > 0,
            dir=wandb_dir,
            sync_tensorboard=True)
        wandb.watch(model)

    # Initialize visualization
    viz_writer = tensorboardX.SummaryWriter(run_helper.viz_dir)

    def _make_viz(gt_input, gt_truth, final_output, max_items=10):
        max_items = min(max_items, gt_input.shape[0])
        in_grid = torchvision.utils.make_grid(gt_input[0:max_items, ...].to('cpu'),
                                              nrow=max_items, padding=0)
        gt_grid = torchvision.utils.make_grid(gt_truth[0:max_items, ...].to('cpu'),
                                              nrow=max_items, padding=0)
        out_grid = torchvision.utils.make_grid(final_output[0:max_items, ...].to('cpu'),
                                               nrow=max_items, padding=0)
        grid = torch.cat([in_grid, gt_grid, out_grid], dim=-2)
        return grid

    def _get_input_truth(in_data):
        in_data = _get_random_crop()(in_data.to(device))
        if not args.triband_input:
            out_x = in_data
            out_truth = out_x
        else:
            out_x = in_data[:, 1:2, ...]  # Binary image is G channel
            if args.exact_loss_with_triband_input:
                out_truth = out_x
            else:
                out_truth = in_data[:, 2:, ...]  # Target image is B channel
        return out_x, out_truth

    # Training loop
    break_now = False
    for ep in range(args.epochs):
        for i, (train_data, _) in enumerate(train_loader, 0):
            train_x, truth = _get_input_truth(train_data)

            # Train model
            with run_helper.detect_grad_anomaly_if_debug():
                opt.zero_grad()  # zero the gradient buffers
                # log_tensor(train_x, 'model_input', logger, print_stats=True, detailed=True)

                # Let's convert to 0..1
                raw_out = model(model.preprocess(train_x))
                partial_proc_out = model.postprocess_partial(raw_out)
                proc_out = model.postprocess(raw_out)

                # Compute losses
                losses = {'loss': 0}
                for crit_name, (criterion, weight, preproc) in criteria.items():
                    _tmp_truth = model.preprocess_truth_for_logits(prep_truth(truth)) if not preproc else truth
                    _tmp_result = proc_out.clip(0, 1) if preproc else partial_proc_out
                    # Note: occasionally some weird exception is thrown due to non 0...1 output
                    # log_tensor(_tmp_truth, 'tmp truth', logger, level=logging.INFO, print_stats=True)
                    # log_tensor(_tmp_result, 'tmp result', logger, level=logging.INFO, print_stats=True)
                    # log_tensor(raw_out, 'raw result', logger, level=logging.INFO, print_stats=True)
                    if not args.balanced_loss:
                        losses[crit_name] = criterion(_tmp_result, _tmp_truth.clip(0, 1))
                    else:
                        thresh = 0.1
                        nzeros = torch.sum(train_x < thresh, dim=(1, 2, 3)) + train_x.shape[-2]
                        nones = torch.sum(train_x >= thresh, dim=(1, 2, 3)) + train_x.shape[-2]
                        total = nzeros + nones
                        bg_weight = (nzeros / total).reshape((-1, 1, 1, 1))
                        fg_weight = (nones / total).reshape((-1, 1, 1, 1))
                        loss_weight = ((train_x >= thresh) * bg_weight) + ((train_x < thresh) * fg_weight)
                        raw_loss = criterion(_tmp_result, _tmp_truth.clip(0, 1))
                        losses[crit_name] = (raw_loss * loss_weight).mean()
                        del loss_weight
                        del raw_loss
                    del _tmp_truth
                    del _tmp_result
                    losses['loss'] += weight * losses[crit_name]

                # Learn
                losses['loss'].backward()
                opt.step()  # Does the update

                # Log things
                if run_helper.needs_train_log(total_its):
                    log_tensor(raw_out, 'raw output', logger, print_stats=True, detailed=True)
                    run_util.write_losses(losses, 'train', total_its, viz_writer)
                    run_util.log_losses(losses, 'train', ep, total_its, logger)
                    if args.wandb_experiment_group is None:
                        # TODO: figure out a more compact way to log gradients
                        run_util.write_gradients(model, total_its, viz_writer, logger)

                # Visualize
                if run_helper.needs_train_viz(total_its):
                    viz_img = _make_viz(train_x, truth, proc_out)
                    viz_writer.add_image('train_viz', viz_img, total_its)
                    # TODO: maybe also save to image?
                viz_writer.flush()

                # Checkpoint
                total_its += 1
                if run_helper.needs_checkpoint(total_its):
                    run_helper.process_checkpoint(args.model_name,
                                                  args_training,
                                                  ep,
                                                  total_its,
                                                  model_state_dict=model.state_dict(),
                                                  opt_state_dict=opt.state_dict())

            # Evaluate model
            if eval_loader is not None and run_helper.needs_eval(total_its):
                eval_losses = {'loss': 0}
                neval_batches = 0
                model.eval()
                for j, (eval_data, _) in enumerate(eval_loader, 0):
                    eval_x, eval_truth = _get_input_truth(eval_data)

                    neval_batches += 1
                    with torch.no_grad():
                        eval_raw_out = model(model.preprocess(eval_x))
                        eval_partial_proc_out = model.postprocess_partial(eval_raw_out)
                        eval_proc_out = model.postprocess(eval_raw_out)
                        if j < 6:
                            eval_viz_img = _make_viz(eval_x, eval_truth, eval_proc_out)
                            viz_writer.add_image('eval_viz%d' % j, eval_viz_img, total_its)

                        # Compute losses
                        for crit_name, (criterion, weight, preproc) in eval_criteria.items():
                            _tmp_truth = model.preprocess_truth_for_logits(prep_truth(eval_truth)) if not preproc else eval_truth
                            _tmp_result = eval_proc_out.clip(0, 1) if preproc else eval_partial_proc_out
                            batch_ecrit = criterion(_tmp_result, _tmp_truth.clip(0, 1)).mean()
                            del _tmp_truth
                            del _tmp_result
                            if crit_name not in eval_losses:
                                eval_losses[crit_name] = 0
                            eval_losses[crit_name] += batch_ecrit
                            eval_losses['loss'] += weight * batch_ecrit

                    del eval_raw_out
                    del eval_proc_out
                    run_util.reclaim_cuda_memory()

                # Normalize loss by number of eval batches
                for lk, lv in eval_losses.items():
                    eval_losses[lk] = lv / neval_batches
                run_util.write_losses(eval_losses, 'eval', total_its, viz_writer)
                run_util.log_losses(eval_losses, 'eval', ep, total_its, logger)
                model.train()

            if args.max_its > 0 and total_its >= args.max_its:
                logger.info('Terminating due to --max_its %d reached' % total_its)
                break_now = True
                break

        if break_now or ep == args.epochs - 1:
            run_helper.process_checkpoint(args.model_name,
                                          args_training,
                                          ep,
                                          total_its,
                                          model_state_dict=model.state_dict(),
                                          opt_state_dict=opt.state_dict())
            break
    logger.info('Training run completed.')
