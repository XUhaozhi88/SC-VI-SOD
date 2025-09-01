# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class CheckNanHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def __init__(self, interval: int = 1) -> None:
        self.interval = interval

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Regularly check whether the loss is valid every n iterations.

        Args:
            runner (:obj:`Runner`): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict, Optional): Data from dataloader.
                Defaults to None.
            outputs (dict, Optional): Outputs from model. Defaults to None.
        """
        # if self.every_n_train_iters(runner, self.interval):
        #     assert torch.isfinite(outputs['loss']), \
        #         runner.logger.info('loss become infinite or NaN!')
            
        # module = runner.model
        # # unused_params = []
        # for name, module in module.named_modules():
        #     """检查每个层的输入和输出是否含有 NaN 值"""
        #     # 检查输入
        #     if torch.isnan(input).any():
        #         print(f"NaN detected in input of {module.__class__.__name__}")
            
        #     # 检查输出
        #     if torch.isnan(output).any():
        #         print(f"NaN detected in output of {module.__class__.__name__}")
