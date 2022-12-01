# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.


import numpy as np
import torch
from torch_utils import training_stats
from torch_utils import misc
from torch_utils.ops import conv2d_gradfix

import forger.train.losses
import forger.train.stitching

"""
This is the StyleGANLoss with geometry loss integration
"""
class ForgerLoss(object):
    def __init__(self, device,
                 G, D,
                 geom_encoder,
                 stitcher,
                 augment_pipe=None, style_mixing_prob=0.9,
                 r1_gamma=10, pl_batch_shrink=2, pl_decay=0.01, pl_weight=2,
                 geom_mode_D='orig',  # The mode of geometry feature to use when in Dmain phase
                 geom_mode_G='orig',  # The mode of geometry feature to use when in Gmain phase
                 color_format='triad',  # Possible values: "triad", "canvas"
                 geom_phase_losses=None,
                 main_phase_losses=None,
                 geom_warmstart_losses=None,
                 stitch_phase_losses=None,
                 partial_loss_with_triband_input=False
                 ):
        super().__init__()
        self.device = device
        self.G = G
        self.D = D
        self.stitcher = stitcher  # forger.train.stitching.RandomStitcher
        self.color_format = color_format
        self.geom_phase_losses = forger.train.losses.ForgerLosses.create_from_string(geom_phase_losses)
        self.geom_phase_losses.set_partial_loss_with_triband_input(partial_loss_with_triband_input)
        self.stitch_phase_losses = forger.train.losses.ForgerLosses.create_from_string(stitch_phase_losses)
        self.main_phase_losses = forger.train.losses.ForgerLosses.create_from_string(main_phase_losses)
        self.main_phase_losses.set_partial_loss_with_triband_input(partial_loss_with_triband_input)
        if geom_warmstart_losses is not None:
            self.geom_warmstart_losses = forger.train.losses.ForgerLosses.create_from_string(geom_warmstart_losses)
        else:
            self.geom_warmstart_losses = self.geom_phase_losses

        # TODO: Perhaps try "downsampling"?
        geom_modes = ('orig', 'enc', 'rand', 'zero')
        assert geom_mode_D in geom_modes, "geom_mode_D not valid."
        assert geom_mode_G in geom_modes, "geom_mode_G not valid."
        self.geom_mode_D = geom_mode_D
        self.geom_mode_G = geom_mode_G

        self.augment_pipe = augment_pipe
        self.style_mixing_prob = style_mixing_prob
        self.r1_gamma = r1_gamma
        self.pl_batch_shrink = pl_batch_shrink
        self.pl_decay = pl_decay
        self.pl_weight = pl_weight
        self.pl_mean = torch.zeros([], device=device)
        self.print_summary()

    def requires_frozen_generator(self):
        return self.geom_phase_losses.require_original_fake_image() or \
               self.geom_phase_losses.require_original_fake_image() or \
                self.geom_warmstart_losses.require_original_fake_image()

    def print_summary(self):
        if self.geom_warmstart_losses != self.geom_phase_losses:
            print('ForgerLoss - Geometry warmstart losses: ')
            self.geom_warmstart_losses.print_summary()
        if not self.geom_phase_losses.is_empty():
            print('ForgerLoss - Geometry phase losses: ')
            self.geom_phase_losses.print_summary()
        if not self.main_phase_losses.is_empty():
            print('ForgerLoss - Main phase losses: ')
            self.main_phase_losses.print_summary()
        if not self.stitch_phase_losses.is_empty():
            print('ForgerLoss - Stitch phase losses: ')
            self.stitch_phase_losses.print_summary()

    def run_G(self, z, c, geom_feature, sync, style_mixing_prob=None):
        if style_mixing_prob is None:
            style_mixing_prob = self.style_mixing_prob
        # return_debug_data must be True
        with misc.ddp_sync(self.G, sync):
            img, debug_data = self.G(z=z,
                                     c=c,
                                     geom_feature=geom_feature,
                                     return_debug_data=True,
                                     style_mixing_prob=style_mixing_prob)
        return img, debug_data

    def run_D(self, img, c, sync):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        with misc.ddp_sync(self.D, sync):
            logits = self.D(img, c)
        return logits

    def accumulate_gradients_stitch(self, geom_feature1, geom_feature2, crop1, crop2, gen_z, gen_c, gain):
        assert not self.stitch_phase_losses.is_empty()

        with torch.autograd.profiler.record_function('Gmain_forward_stitch'):
            res = self.stitcher.generate_with_stitching(self.G, gen_z, gen_c, geom_feature1, geom_feature2, crop1, crop2)

            fake = torch.cat([res['fake1'], res['fake2']], dim=0)
            fake_logits = self.run_D(fake, gen_c, sync=False)

            composite = torch.cat([res['fake1_composite'], res['fake2_composite']], dim=0)
            composite_logits = self.run_D(composite, gen_c, sync=False)

            training_stats.report('Loss/forger_stitch/scores/fake', fake_logits)
            training_stats.report('Loss/forger_stitch/scores/composite', composite_logits)

            training_stats.report('Loss/forger_stitch/signs/fake', fake_logits.sign())
            training_stats.report('Loss/forger_stitch/signs/composite', composite_logits.sign())

            debug_data = {'fake': fake,
                          'fake_logits': fake_logits,
                          'fake_composite': composite,
                          'fake_composite_logits': composite_logits,
                          'patch1': res['patch1'],
                          'patch2': res['patch2']}
            forger_loss, loss_vals = self.stitch_phase_losses.compute(debug_data, None)
            training_stats.report(f'Loss/forger/Gstitch/total', forger_loss)
            for k, v in loss_vals.items():
                training_stats.report(f'Loss/forger/Gstitch/{k}', v)

        with torch.autograd.profiler.record_function('Gmain_backward'):
            forger_loss.mul(gain).backward()

    def accumulate_gradients(self, phase, real_style, real_c, real_geom, geom_feature, gen_z, gen_c, sync, gain, G_orig=None):
        assert phase in ['Gmain', 'Greg', 'Gall', 'Ggeom', 'Ggeom-warm', 'Dmain', 'Dreg', 'Dall']
        do_Gmain = (phase in ['Gmain', 'Gall'])
        do_Dmain = (phase in ['Dmain', 'Dall'])
        do_Gpl   = (phase in ['Greg', 'Gall']) and (self.pl_weight != 0)
        do_Dr1   = (phase in ['Dreg', 'Dall']) and (self.r1_gamma != 0)
        do_Ggeom = (phase in ['Ggeom', 'Ggeom-warm'])

        # Gmain: Maximize logits for generated images. We also do the geometry loss back-prop here.
        if do_Gmain:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # When updating the Generator, different ways to generate the geometry feature
                # TODO: Avoid conditioning by creating a function object in __init__
                phase_geom_feature = geom_feature  # mode 'orig'
                assert self.geom_mode_G == 'orig', 'Not implemented'
                if self.geom_mode_G == 'rand':
                    phase_geom_feature = torch.rand_like(geom_feature)
                elif self.geom_mode_G == 'zero':
                    phase_geom_feature = torch.zeros_like(geom_feature)
                elif self.geom_mode_G == "enc":
                    phase_geom_feature = self.geom_feature_from_style_data(real_style=real_style)

                gen_img, gen_data = self.run_G(gen_z, gen_c, geom_feature=phase_geom_feature, sync=(sync and not do_Gpl)) # May get synced by Gpl.
                gen_logits = self.run_D(gen_img, gen_c, sync=False)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))+
                training_stats.report('Loss/G/loss', loss_Gmain)

                # Note: if not set, this is just zero
                forger_loss_Gmain, loss_vals = self.main_phase_losses.compute(gen_data, real_geom)
                for k, v in loss_vals.items():
                    training_stats.report(f'Loss/forger/Gmain/{k}', v)
                loss_Gmain = loss_Gmain + forger_loss_Gmain
                training_stats.report(f'Loss/forger/Gmain/total', forger_loss_Gmain)
                training_stats.report('Loss/G/total_loss', loss_Gmain)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Update G based on geometry loss
        if do_Ggeom:
            geom_losses = self.geom_phase_losses
            if phase == 'Ggeom-warm':
                geom_losses = self.geom_warmstart_losses

            if not geom_losses.is_empty():
                has_frozen_g_loss = geom_losses.require_original_fake_image()
                with torch.autograd.profiler.record_function('%s_forward' % phase):
                    gen_img, gen_data = self.run_G(gen_z,
                                                   gen_c,
                                                   geom_feature=geom_feature,
                                                   sync=True,
                                                   style_mixing_prob=0 if has_frozen_g_loss else None)  # May get synced by Gpl.
                    gen_data['fake_img'] = gen_img
                    if has_frozen_g_loss:
                        gen_data['fake_orig'] = G_orig(z=gen_z, c=gen_c,
                                                       geom_feature=geom_feature,
                                                       return_debug_data=False, style_mixing_prob=0)

                    loss, loss_vals = geom_losses.compute(gen_data, real_geom)
                    for k, v in loss_vals.items():
                        training_stats.report(f'Loss/forger/{phase}/{k}', v)
                with torch.autograd.profiler.record_function('%s_backward' % phase):
                    loss.mean().backward()

        # Gpl: Apply path length regularization.
        if do_Gpl:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_data = self.run_G(gen_z[:batch_size], gen_c[:batch_size],
                                               geom_feature=[geof[:batch_size] for geof in geom_feature], sync=sync)
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients():
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_data['ws']], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                (gen_img[:, 0, 0, 0] * 0 + loss_Gpl).mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if do_Dmain:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                # When updating the Discriminator, different ways to generate the geometry feature
                # TODO: Have a function for getting phase_geom_feature, since it's also used for Generator
                phase_geom_feature = geom_feature  # mode 'orig'
                if self.geom_mode_D == 'rand':
                    phase_geom_feature = [torch.rand_like(geof) for geof in geom_feature]
                elif self.geom_mode_D == 'zero':
                    phase_geom_feature = [torch.zeros_like(geof) for geof in geom_feature]
                elif self.geom_mode_D == "enc":
                    raise RuntimeError('Not supported')
                    phase_geom_feature = self.geom_feature_from_style_data(real_style=real_style)
                gen_img, _gen_data = self.run_G(gen_z, gen_c, geom_feature=phase_geom_feature, sync=False)
                gen_logits = self.run_D(gen_img, gen_c, sync=False) # Gets synced by loss_Dreal.
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if do_Dmain or do_Dr1:
            name = 'Dreal_Dr1' if do_Dmain and do_Dr1 else 'Dreal' if do_Dmain else 'Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_style.detach().requires_grad_(do_Dr1)
                real_logits = self.run_D(real_img_tmp, real_c, sync=sync)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if do_Dmain:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if do_Dr1:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (real_logits * 0 + loss_Dreal + loss_Dr1).mean().mul(gain).backward()

    def _compute_geom_loss_canvas(self, gen_data, real_geom):
        """
        The function that performs Ggeom phase operations in 'canvas' model.
        """
        loss_canvas = self.canvas_loss.compute(gen_data['canvas'])
        training_stats.report(f'Loss/G/canvas_loss', loss_canvas)
        return loss_canvas

    def _compute_geom_loss_triad(self, gen_data, real_geom):
        """
        The function that performs Ggeom phase operations in 'triad' model.
        """
        loss_geom, loss_geom_dict = self.geom_loss.compute(gen_data, real_geom)
        training_stats.report('Loss/G/geom_loss/total', loss_geom)
        for key, val in loss_geom_dict['unweighted']['uvs'].items():
            training_stats.report(f'Loss/G/geom_loss/unweighted/{key}', val)
        return loss_geom

    def geom_feature_from_style_data(self, real_style):
        """
        This function is used in case one wants to use "style data" as "geometry data"
        Args:
            real_style:torch.Tensor of shape [B, 3, W, H], within the range [-1, 1-
        """
        with torch.no_grad():
            phase_gray_style = torch.mean((real_style + 1.0) * 0.5, dim=1).unsqueeze(1)
            temp_min = torch.amin(phase_gray_style, dim=(2, 3), keepdim=True)
            temp_max = torch.amax(phase_gray_style, dim=(2, 3), keepdim=True)
            temp_min = temp_min.expand(real_style.shape[0], 1, real_style.shape[2], real_style.shape[3])
            temp_max = temp_max.expand(real_style.shape[0], 1, real_style.shape[2], real_style.shape[3])
            phase_gray_style = (phase_gray_style - temp_min) / (temp_max - temp_min + torch.ones_like(temp_max) * 1e-7)  # Normalize to range [0.0, 1.0]
            phase_geom_feature = self.geom_encoder(phase_gray_style)
            return phase_geom_feature
#----------------------------------------------------------------------------
