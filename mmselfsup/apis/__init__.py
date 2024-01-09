# Copyright (c) OpenMMLab. All rights reserved.
from .train import init_random_seed, set_random_seed, train_model

__all__ = ['init_random_seed', 'set_random_seed', 'train_model']

#########################################################
from .epoch_based_runner_continuous_learning import EpochBasedRunnerContinuousLearning
__all__.extend(
    ['EpochBasedRunnerContinuousLearning'
     ])








