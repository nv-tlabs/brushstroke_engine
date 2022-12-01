# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import copy
import numpy as np
import random
import torch
import torchvision
import torch.nn as nn

from thirdparty.stylegan2_ada_pytorch.dnnlib.util import EasyDict


class BlendedFeatures:
    """ Features that can be blended into other features using a soft mask. """
    def __init__(self, features, alpha):
        self.features = features
        self.alpha = alpha

    def blend(self, other_features):
        return self.alpha * self.features + (1 - self.alpha) * other_features


class CropHelper(object):
    @staticmethod
    def position_delta(crop1, crop2):
        return torch.tensor([crop2[0] - crop1[0], crop2[1] - crop1[1]], dtype=torch.int64)

    @staticmethod
    def make_area(rstart, cstart, rows, cols):
        rend = rstart + rows
        cend = cstart + cols
        return CropHelper.make_area_direct(rstart, cstart, rend, cend)

    @staticmethod
    def make_area_direct(rstart, cstart, rend, cend):
        res = EasyDict()
        # Row and column start (inclusive) and end (exclusive)
        res.rstart = rstart
        res.cstart = cstart
        res.rend = rend
        res.cend = cend
        res.min_width = min(res.rend - res.rstart, res.cend - res.cstart)  # Negative if no overlap
        return res

    @staticmethod
    def compute_absolute_overlap(cropA, cropB):
        res = EasyDict()
        # Row and column start (inclusive) and end (exclusive)
        res.rstart = max(cropA[0], cropB[0])
        res.cstart = max(cropA[1], cropB[1])
        res.rend = min(cropA[0] + cropA[2], cropB[0] + cropB[2])
        res.cend = min(cropA[1] + cropA[3], cropB[1] + cropB[3])
        res.min_width = min(res.rend - res.rstart, res.cend - res.cstart)  # Negative if no overlap
        return res

    @staticmethod
    def compute_overlaps(cropA, cropB):
        abs_overlap = CropHelper.compute_absolute_overlap(cropA, cropB)

        if abs_overlap.min_width <= 0:
            return abs_overlap, None, None

        def _make_relative_area(abs_area, crop):
            area = copy.deepcopy(abs_area)
            area.rstart -= crop[0]
            area.rend -= crop[0]
            area.cstart -= crop[1]
            area.cend -= crop[1]
            return area

        return abs_overlap, _make_relative_area(abs_overlap, cropA), _make_relative_area(abs_overlap, cropB)

    @staticmethod
    def offset_crop(crop, margin):
        return (crop[0] + margin,
                crop[1] + margin,
                crop[2] - 2 * margin,
                crop[3] - 2 * margin)

    @staticmethod
    def offset_area(area, margin):
        return CropHelper.make_area(area.rstart + margin, area.cstart + margin,
                                    area.rend - area.rstart - margin * 2,
                                    area.cend - area.cstart - margin * 2)

    @staticmethod
    def pad_area_bounded(area, margin, max_dim):
        rows = area.rend - area.rstart
        cols = area.cend - area.cstart
        rmargin = min(margin, (max_dim - rows) // 2)
        cmargin = min(margin, (max_dim - cols) // 2)
        return CropHelper.make_area_direct(
            area.rstart - rmargin, area.cstart - cmargin,
            rend=area.rend + rmargin, cend=area.cend + cmargin)

    @staticmethod
    def clip_area(area, source_rows, source_cols):
        return CropHelper.make_area_direct(
            rstart=max(0, min(area.rstart, source_rows - 1)),
            cstart=max(0, min(area.cstart, source_cols - 1)),
            rend=max(0, min(area.rend, source_rows)),
            cend=max(0, min(area.cend, source_cols)))

    @staticmethod
    def make_area_relative(area, parent_area):
        rstart = max(area.rstart - parent_area.rstart, 0)
        cstart = max(area.cstart - parent_area.cstart, 0)

        rend = min(area.rend, parent_area.rend) - parent_area.rstart
        rows = rend - rstart

        cend = min(area.cend, parent_area.cend) - parent_area.cstart
        cols = cend - cstart
        return CropHelper.make_area(rstart, cstart, rows, cols)

    @staticmethod
    def expand_area(area, to_width, source_rows, source_cols):
        """
        Expands area (see make_area) to be exactly to_width heigh and wide.
        By default expands to be centered around original area center,
        but is also guaranteed to remain within source_rows, source_cols larger
        area of the canvas.
        Reasonable crops for areas that are larger are not guaranteed.

        @param area:
        @param to_width:
        @param source_rows:
        @param source_cols:
        @return:
        """
        rows = area.rend - area.rstart
        cols = area.cend - area.cstart
        if rows == to_width and cols == to_width:
            return area

        def _find_start(extra, start, max_val):
            if extra <= 0:
                # TODO: do better crop
                return start, start + to_width

            new_start = int(max(0, start - extra // 2))
            new_end = new_start + to_width
            if new_end > max_val:
                new_end = max_val
                new_start = new_end - to_width
            return new_start

        rextra = to_width - rows
        cextra = to_width - cols
        new_rstart = _find_start(rextra, area.rstart, source_rows)
        new_cstart = _find_start(cextra, area.cstart, source_cols)

        return CropHelper.make_area(new_rstart, new_cstart, to_width, to_width)

    @staticmethod
    def composite(im1, im2, area1, area2, alpha1=None):
        """
        Composite im2 into im1, such that area1 in im1 is now occupied by pixels in area2 of im2,
        where features of im1 are optionally blended using mask alpha1, same size as area1 and area2.

        @param im1: B x C x H x W
        @param im2: B x C x H x W
        @param area1: see make_area above; area within im1
        @param area2: see make_area above; area within im2, same size
        @param alpha1: h x w, same size as areas
        @return:
        """
        mask1 = torch.ones_like(im1[:1, :1, ...])
        mask1[..., area1.rstart:area1.rend, area1.cstart:area1.cend] = alpha1 if alpha1 is not None else 0

        res = mask1 * im1
        res[..., area1.rstart:area1.rend, area1.cstart:area1.cend] += \
            (1 - alpha1 if alpha1 is not None else 1.0) * im2[..., area2.rstart:area2.rend, area2.cstart:area2.cend]
        return res

    @staticmethod
    def gen_overlapping_square_crop(input_width, crop1, margin, min_overlap):
        width = crop1[2]
        radius = width - margin - min_overlap - 1
        ij = [0, 0]
        for x in range(2):
            rmin = max(0, crop1[x] - radius)
            rmax = min(crop1[x] + radius, input_width - width - 1)
            ij[x] = random.randint(rmin, rmax)

        return ij[0], ij[1], width, width


class RandomStitcher(nn.Module):
    def __init__(self,
                 crop_margin=10,
                 min_overlap=50):
        super().__init__()
        self.margin = crop_margin
        self.min_overlap = min_overlap

    def gen_overlapping_square_crop(self, input_width, crop1):
        return CropHelper.gen_overlapping_square_crop(input_width, crop1, self.margin, self.min_overlap)

    def offset_crop(self, crop):
        return CropHelper.offset_crop(crop, self.margin)

    @staticmethod
    def gen_random_positions(batch, width):
        return torch.randint(0, width - 1, (batch, 2))

    def generate_with_stitching(self, G, z, c, geom_feature1, geom_feature2, crop1, crop2, positions1=None):
        """
        This is batched.

        @param z:
        @param c:
        @param geom_feature1: list of multi-level features, as accepted by Generator for crop1
        @param geom_feature2: list of multi-level features, as accepted by Generator for crop2
        @param crop1: tuple (row_start, col_start, height, width)

        @return: (fake1, fake2, fake1_composite, geom_patch, fake1_patch, fake2_patch, fake2_geom)

        May make sense to use a dictionary, not tuple.
        - fake1: first generator result
        - fake2: raw second generator result (patch is in another place)
        - fake1_composite: fake1 with patch of fake2 composited in the right place
        - geom_crop: random geometry crop
        - fake1_crop: patch of fake1 that corresponds to geom_crop
        - fake2_crop: patch of fake2 that corresponds to geom_crop
        - fake2_geom: the full geometry used to generate fake2 (for debugging)

        """
        img_resolution = G.img_resolution
        batch = z.shape[0]

        # Step 0: generate random position
        if positions1 is None:
            # TODO: make width configurable, it need not equal image resolution
            positions1 = RandomStitcher.gen_random_positions(batch, width=img_resolution).to(z.device)
        positions2 = positions1 + CropHelper.position_delta(crop1, crop2).unsqueeze(0).to(z.device)

        # Step 1: run generator as usual, both sets of features and positions
        fake1 = G(z, c, geom_feature=geom_feature1, positions=positions1, style_mixing_prob=0)
        fake2 = G(z, c, geom_feature=geom_feature2, positions=positions2, style_mixing_prob=0)

        # Composite fake2 patch into fake1
        abs_area, area1, area2 = CropHelper.compute_overlaps(crop1, self.offset_crop(crop2))
        fake1_composite = CropHelper.composite(fake1, fake2, area1, area2)

        # Composite fake1 patch into fake2
        abs_area, area1, area2 = CropHelper.compute_overlaps(self.offset_crop(crop1), crop2)
        fake2_composite = CropHelper.composite(fake2, fake1, area2, area1)

        patch1 = fake1[..., area1.rstart:area1.rend, area1.cstart:area1.cend]
        patch2 = fake2[..., area2.rstart:area2.rend, area2.cstart:area2.cend]

        return {
            'fake1': fake1,
            'fake2': fake2,
            'fake1_composite': fake1_composite,
            'fake2_composite': fake2_composite,
            'positions1': positions1,
            'positions2': positions2,
            'patch1': patch1,
            'patch2': patch2
        }
