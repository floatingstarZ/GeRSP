# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class SimSiamHook(Hook):
    """Hook for SimSiam.

    This hook is for SimSiam to fix learning rate of predictor.

    Args:
        fix_pred_lr (bool): whether to fix the lr of predictor or not.
        lr (float): the value of fixed lr.
        adjust_by_epoch (bool, optional): whether to set lr by epoch or iter.
            Defaults to True.
    """

    def __init__(self, fix_pred_lr, lr, adjust_by_epoch=True, **kwargs):
        self.fix_pred_lr = fix_pred_lr
        self.lr = lr
        self.adjust_by_epoch = adjust_by_epoch

    def before_train_iter(self, runner):
        if self.adjust_by_epoch:
            return
        else:
            if self.fix_pred_lr:
                for param_group in runner.optimizer.param_groups:
                    if 'fix_lr' in param_group and param_group['fix_lr']:
                        param_group['lr'] = self.lr

    def before_train_epoch(self, runner):
        """fix lr of predictor."""
        if self.fix_pred_lr:
            for param_group in runner.optimizer.param_groups:
                if 'fix_lr' in param_group and param_group['fix_lr']:
                    param_group['lr'] = self.lr
