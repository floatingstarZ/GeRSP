# Copyright (c) OpenMMLab. All rights reserved.
import os
import os.path as osp

import numpy as np

from ..builder import DATASOURCES
from .base import BaseDataSource


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


@DATASOURCES.register_module()
class ImgFolder(BaseDataSource):
    """`ImageNet <http://www.image-net.org>`_ Dataset.

    This implementation is modified from
    https://github.com/pytorch/vision/blob/master/torchvision/datasets/imagenet.py
    """  # noqa: E501

    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif')

    def load_annotations(self):

        samples = []
        img_dir = osp.expanduser(self.data_prefix)
        for f_pth in os.listdir(img_dir):
            if has_file_allowed_extension(f_pth, self.IMG_EXTENSIONS):
                item = (f_pth, 0)
                samples.append(item)

        self.samples = samples

        data_infos = []
        for i, (filename, gt_label) in enumerate(self.samples):
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': filename}
            info['gt_label'] = np.array(gt_label, dtype=np.int64)
            info['idx'] = int(i)
            data_infos.append(info)
        print('#' * 100)
        print('Load Images: %d' % len(self.samples))
        print('#' * 100)

        return data_infos
