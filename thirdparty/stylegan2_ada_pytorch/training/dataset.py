# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import re
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import torchvision
import logging

try:
    import pyspng
except ImportError:
    pyspng = None


logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        original_resolution = None,    # Resolution of original images on disk
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        transform = None
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._original_resolution = original_resolution
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self._transform = transform

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def print_info(self):
        print(f'Name:                {self.name}')
        print(f'Num images:          {len(self)}')
        print(f'Original resolution: {self.original_resolution}')
        print(f'Image shape:         {self.image_shape}')
        print()

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        # Can be any type
        if self._transform is not None:
            image = self._transform(image)
        assert list(image.shape) == self.image_shape
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert len(image.shape) == 3  # CHW
        if self._xflip[idx]:
            image = image[:, :, ::-1]
        return image, self.get_label(idx)  # TODO: why was this image.copy()?

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def original_resolution(self):
        if self._original_resolution is not None:
            return self._original_resolution
        else:
            return self.resolution

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class EnsureChannelDimension(object):
    def __init__(self):
        pass

    def __call__(self, img_tensor):
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0)
        return img_tensor


class PilToNp(object):
    def __init__(self):
        pass

    def __call__(self, pil_img):
        image = np.array(pil_img)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image


class ImageFolderDataset(Dataset):
    def __init__(self,
                 path,                   # Path to directory or zip.
                 resolution      = None, # Ensure specific resolution, None = highest available.
                 resize_mode     = None,  # If resolution != image resolution, can crop or resize
                 regexp          = ".*",
                 name            = None,
                 **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if
                                    (self._file_ext(fname) in PIL.Image.EXTENSION and self._matches_regexp(fname, regexp)))
        if len(self._image_fnames) == 0:
            raise IOError('No image files matching "{}" found in the specified path'.format(regexp))

        if name is None:
            name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._get_raw_image_chw_shape())

        original_resolution = None
        transforms = [PilToNp()]
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            info_str = f'Native resolution ({raw_shape[2]}x{raw_shape[3]}) does not match desired value {resolution}: '
            original_resolution = raw_shape[-1]
            if resize_mode == 'crop':
                assert resolution < raw_shape[2] and resolution < raw_shape[3]
                transforms.insert(0, torchvision.transforms.RandomCrop(resolution))
                logger.info(f'{info_str} using crop transform')
            elif resize_mode == 'resize':
                transforms.insert(0, torchvision.transforms.Resize(resolution))
                logger.info(f'{info_str} using resize transform')
            else:
                raise RuntimeError(f'{info_str} and unknown resize mode {resize_mode}')
            raw_shape[-1] = resolution
            raw_shape[-2] = resolution
        super().__init__(name=name, raw_shape=raw_shape, original_resolution=original_resolution,
                         transform=torchvision.transforms.Compose(transforms), **super_kwargs)

    @staticmethod
    def _matches_regexp(fname, regexp):
        if regexp is None:
            return True
        return re.match(regexp, fname) is not None

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _get_raw_image_chw_shape(self):
        fname = self._image_fnames[0]
        with self._open_file(fname) as f:
            image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]  # HW => HWC
        image = image.transpose(2, 0, 1)  # HWC => CHW
        return image.shape

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png' and self._transform is None:
                image = pyspng.load(f.read())

                if image.ndim == 2:
                    image = image[:, :, np.newaxis]  # HW => HWC
                image = image.transpose(2, 0, 1)  # HWC => CHW
            else:
                image = PIL.Image.open(f)
                image.load()
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
