# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple

from collections import OrderedDict
from typing import (Any, Callable, Dict, List, Optional, Tuple, Union,
                    no_type_check)

import torch
from torch.utils.data import DataLoader

import mmcv
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.epoch_based_runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info
from torch.optim import Optimizer

@RUNNERS.register_module()
class EpochBasedRunnerContinuousLearning(EpochBasedRunner):

    @no_type_check
    def resume(self,
               checkpoint: str,
               resume_optimizer: bool = True,
               map_location: Union[str, Callable] = 'default') -> None:
        if map_location == 'default':
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                checkpoint = self.load_checkpoint(
                    checkpoint,
                    map_location=lambda storage, loc: storage.cuda(device_id))
            else:
                checkpoint = self.load_checkpoint(checkpoint)
        else:
            checkpoint = self.load_checkpoint(
                checkpoint, map_location=map_location)

        last_epoch = checkpoint['meta']['epoch']
        last_iter = checkpoint['meta']['iter']
        self.logger.info('#' * 100)
        self.logger.info('EpochBasedRunnerContinuousLearning '
                         'resumed epoch %d, iter %d', last_epoch, last_iter)
        self.logger.info('EpochBasedRunnerContinuousLearning '
                         'All Meta Data Removed')
        self.logger.info('EpochBasedRunnerContinuousLearning '
                         'Start from epoch %d, iter %d', self.epoch, self.iter)
        self.logger.info('#' * 100)
