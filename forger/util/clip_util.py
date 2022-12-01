# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import logging
import torch
import torchvision
import numpy as np
import clip
import pickle
import os
import torch.nn.functional as F
import torch.utils.data
import re

import thirdparty.stylegan2_ada_pytorch.experiment.util.latent as latent
from forger.util.torch_data import get_image_data_iterator_from_dataset, get_image_data_iterator

from thirdparty.stylegan2_ada_pytorch.training.dataset import ImageFolderDataset
from thirdparty.stylegan2_ada_pytorch.torch_utils.misc import InfiniteSampler

import forger.train.losses
import forger.metrics.geom_metric
from forger.util.logging import log_tensor
import PIL

logger = logging.getLogger(__name__)


class FeatureDictionary:
    def __init__(self, clip_model, img_preprocess, keys, path_from_key_fn):
        self.features = None
        self.keys = keys
        self.clip_model = clip_model
        self.device = 'cuda'
        self._init_features(clip_model, img_preprocess, path_from_key_fn)

    def _init_features(self, clip_model, img_preprocess, path_from_key_fn, batch_size=100):
        self.features = None

        nbatches = int(len(self.keys) / batch_size) + 1
        # has_set = torch.zeros((len(self.keys),))
        with torch.no_grad():
            for i in range(nbatches):
                bkeys = self.keys[i * batch_size:(i + 1) * batch_size]
                sources = torch.stack([img_preprocess(PIL.Image.open(path_from_key_fn(s))) for s in bkeys]).to(self.device)
                log_tensor(sources, 'sources', logger)
                features = clip_model.encode_image(sources)
                log_tensor(features, 'features', logger)

                if self.features is None:
                    self.features = torch.zeros((len(self.keys), features.shape[-1]), dtype=features.dtype,
                                                device=self.device)
                self.features[i * batch_size:(i + 1) * batch_size] = features
                # has_set[i * batch_size:(i + 1) * batch_size] = 1
        # print(has_set)

        self.features = self.features / self.features.norm(dim=1, keepdim=True)

    def get_top_results(self, query, features=None, topk=10):
        text = clip.tokenize([query]).to(self.device)
        text_features = self.clip_model.encode_text(text)

        # normalized features
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        if features is not None:
            text_features = features / features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.clip_model.logit_scale.exp()
        logits_per_image = logit_scale * self.features @ text_features.t()

        log_tensor(logits_per_image, 'logits_per_image', logger)
        logits_per_image = logits_per_image.reshape(-1)
        res = torch.topk(logits_per_image, k=topk)
        log_tensor(res[0], 'res[0]', logger)
        log_tensor(res[1], 'res[1]', logger)

        key_indices = list(res[1].detach().cpu().numpy())
        print(key_indices)
        result_keys = self.keys[key_indices]
        print(result_keys)

        # shape = [global_batch_size, global_batch_size]
        return result_keys


class ClipStyleOptimizer:
    def __init__(self, clip_model, clip_preprocess, engine, geom_dataset, w_avg_samples=10000, batch_size=30,
                 geom_input_channel=1, geom_truth_channel=2):
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.engine = engine
        self.batch_size = batch_size
        device = self.engine.device

        output_resolution = self.engine.G.img_resolution
        geom_set_cropped = ImageFolderDataset(path=geom_dataset, use_labels=False,
                                              resolution=output_resolution, resize_mode='crop')
        geom_set_cropped.print_info()
        sampler = InfiniteSampler(dataset=geom_set_cropped, shuffle=True)
        self.geom_it = iter(torch.utils.data.DataLoader(
            dataset=geom_set_cropped, sampler=sampler, batch_size=batch_size))
            #pin_memory=False, num_workers=0, prefetch_factor=0, persistent_workers=True))

        self.geom_input_channel = geom_input_channel
        self.geom_truth_channel = geom_truth_channel

        logger.info(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
        G = self.engine.G
        self.w_avg, self.w_std = latent.get_w_stats(num_samples=w_avg_samples,
                                                    z_dim=G.z_dim,
                                                    mapping_network=G.mapping,
                                                    device=device)
        self.w_avg = torch.from_numpy(self.w_avg)
        log_tensor(self.w_avg, 'w_avg', logger, print_stats=True)

        # Note: hand-deduced for a specific clip model to work on tensors
        # This is not 100% the same as for PIL image due to convert("RGB") step, but it needs
        # to be figured out what space the generator operates in and if this is actually needed
        n_px = 224
        self.custom_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(n_px, interpolation=PIL.Image.BICUBIC),
            torchvision.transforms.CenterCrop(n_px),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def get_clip_loss(self, text_features, images):
        images = images / 2 + 0.5  # convert to 0...1
        images = self.custom_preprocess(images)

        image_features = self.clip_model.encode_image(images)
        #image_features = image_features / image_features.norm(dim=1, keepdim=True)

        return -torch.cosine_similarity(torch.mean(image_features, dim=0),
                                        torch.mean(text_features, dim=0), dim=0)

        # logit_scale = self.clip_model.logit_scale.exp()
        # logits_per_image = logit_scale * image_features @ text_features.t()
        # return logits_per_image.mean()

    def optimize(self, query, text_features=None, resume_from_z=None, resume_from_w=None,
                 w_plus=True,
                 num_steps=1000,
                 initial_learning_rate=0.1,
                 initial_noise_factor=0.05,
                 lr_rampdown_length=0.25,
                 lr_rampup_length=0.05,
                 noise_ramp_length=0.75,
                 regularize_noise_weight=10,  # 1e5,
                 optimize_noise=True,
                 norm_positions=None,
                 bg_weight=0.01,
                 min_loss_improvement=0.0001,
                 output_video=False,
                 partial_loss_with_triband_input=True,
                 iou_weight=1.0,
                 return_result_every_n=-1,
                 video_output_interval=100,
                 on_white=False):
        G = self.engine.G
        device = self.engine.device

        if text_features is None:
            text = clip.tokenize([query]).to(device)
            text_features = self.clip_model.encode_text(text)
        #text_features = text_features / text_features.norm(dim=1, keepdim=True)

        assert resume_from_z is None or resume_from_w is None, 'can only provide z or w, not both'

        loss_weights = {'clip': 1.0,
                        'reg': regularize_noise_weight,
                        'bg': bg_weight,
                        'iou': iou_weight }

        l1_crit = torch.nn.L1Loss()
        iou_crit = forger.train.losses.IoUInverseLossItem('uvs')
        iou_crit.partial_loss_with_triband_input = partial_loss_with_triband_input
        noise_mode = 'const'

        w_start = self.w_avg

        # Allow optimization into W+
        if w_plus:
            w_start = torch.cat([w_start for _ in range(G.mapping.num_ws)], dim=1)

        if resume_from_z is not None:
            resume_from_w = self.engine.G.mapping(resume_from_z, c=None).detach()

        if resume_from_w is not None:
            if w_start.shape != resume_from_w.shape:
                w_start = torch.cat([resume_from_w for _ in range(G.mapping.num_ws)], dim=1).to(device)
            else:
                w_start = resume_from_w.to(device)

        w_start = w_start.to(torch.float32).to(device)
        w_opt = w_start.clone().detach().requires_grad_(True)

        # Setup noise inputs.
        noise_bufs = {}
        if optimize_noise:
            noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

        #             if resume_from is not None and 'noise' in resume_from:
        #                 logger.info('Resuming from noise')
        #                 start_noise = resume_from['noise']
        #                 for k, v in noise_bufs.items():
        #                     v[:] = start_noise[k][:]

        # Init noise.
        for bkey, buf in noise_bufs.items():
            noise_bufs[bkey] = torch.randn_like(buf, requires_grad=True)

        best_img = None
        prev_loss_best = None
        loss_best = None
        w_best = w_opt.detach().cpu()
        noise_best = dict([(k, v.detach().cpu()) for k, v in noise_bufs.items()])
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)
        optimizer.zero_grad(set_to_none=True)

        frames = []
        max_viz_items = 5

        results = []
        for step in range(num_steps):
            geom, _ = next(self.geom_it)
            geom_truth = geom[:, self.geom_truth_channel:self.geom_truth_channel + 1, :, :]
            geom_truth = geom_truth.to(device).to(torch.float32) / 255.0
            geom = geom[:, self.geom_input_channel:self.geom_input_channel + 1, :, :]
            geom = geom.to(device).to(torch.float32) / 255.0
            geom_feature = self.engine.encoder.encode(geom)
            fg, bg = forger.metrics.geom_metric.get_conservative_fg_bg(geom)
            fg = fg.expand(-1, 3, -1, -1)

            # Learning rate schedule.
            t = step / num_steps
            w_noise_scale = self.w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = (w_opt + w_noise).repeat([self.batch_size, 1 if w_plus else G.mapping.num_ws, 1])
            synth_images, raw = G.synthesis(ws, geom_feature=geom_feature, noise_mode=noise_mode,
                                            norm_noise_positions=norm_positions, noise_buffers=noise_bufs,
                                            return_debug_data=True)

            if on_white:
                #log_tensor(synth_images, 'synth', logger)
                #log_tensor(raw['uvs'], 'uvs', logger)
                synth_images = synth_images * (1 - raw['uvs'][:, 2:, ...]) + raw['uvs'][:, 2:, ...]

            losses = {'clip': self.get_clip_loss(text_features, synth_images)}

            if bg_weight > 0:
                losses['bg'] = (1 - raw['uvs'][:, 2:, ...][bg]).mean()

            if loss_weights['iou'] > 0:
                losses['iou'] = iou_crit.compute(raw, geom_truth)
            #del raw

            # Noise regularization.
            losses['reg'] = 0
            for v in noise_bufs.values():
                noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    losses['reg'] += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                    losses['reg'] += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                    if noise.shape[2] <= 8:
                        break
                    noise = F.avg_pool2d(noise, kernel_size=2)

            loss = 0
            for k, v in losses.items():
                loss = loss + loss_weights[k] * v

            if loss_best is None or loss < loss_best:
                loss_best = loss
                w_best = w_opt.detach().cpu()
                noise_best = dict([(k, v.detach().cpu()) for k, v in noise_bufs.items()])
                best_img = synth_images[:max_viz_items, ...].detach().cpu()

            if step % 10 == 0:
                logger.info('Step %d: %s (best loss %0.4f)' %
                            (step,
                             ' + '.join(
                                 ['%s %0.4f * weight %0.4f' % (k, v, loss_weights[k]) for k, v in losses.items()]),
                                loss_best ))

            if return_result_every_n > 0 and step % return_result_every_n == 0:
                results.append({'w': w_best,
                                'noise': noise_best,
                                'step': step})

            if step % video_output_interval == 0:
                if output_video:
                    frames.append(best_img)  #synth_images[:max_viz_items, ...].detach().cpu())

                if prev_loss_best is None:
                    prev_loss_best = loss_best
                else:
                    if prev_loss_best - loss_best < min_loss_improvement:
                        logger.info(
                            'Not enough loss improvement since prior log %0.5f --> %0.5f, stopping after %d steps' %
                            (prev_loss_best, loss_best, step))
                        break
                    prev_loss_best = loss_best

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)  # why does this need the graph??
            optimizer.step()

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()

        frames.append(best_img)
        result = {'w': w_best,
                  'noise': noise_best,
                  'step': step}
        if len(results) > 0:
            results.append(result)
        else:
            results = result
        return results, frames

    def write_to_pickle(self, fname_all_pkl, query, result, overwrite=False):
        key = re.sub("[^0-9a-zA-Z]+", "_", query)

        all_data = {}
        if os.path.isfile(fname_all_pkl):
            with open(fname_all_pkl, 'rb') as f:
                all_data = pickle.load(f)
        if key in all_data:
            if not overwrite:
                logger.info(f'All pickle already has projection for {key}, skipping overwriting: {fname_all_pkl}')
                return
            else:
                logger.warning(f'All pickle already has projection for {key}, overwriting: {fname_all_pkl}')
        all_data[key] = result

        with open(fname_all_pkl, 'wb') as f:
            pickle.dump(all_data, f)
            logger.info(f'Added w entry for {key} to {fname_all_pkl}')