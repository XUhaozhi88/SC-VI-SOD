# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class FindUnusedParametersHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def _after_iter(self,
                runner,
                batch_idx: int,
                data_batch = None,
                outputs=None,
                mode: str = 'train'):
        """Regularly check whether the loss is valid every n iterations.

        Args:
            runner (:obj:`Runner`): The runner of the training process.
        """
        module = runner.model
        unused_params = []
        for name, module in module.named_modules():
            if 'Norm' in module.__class__.__name__:
                continue
            elif 'Embedding' in module.__class__.__name__:
                continue
            elif 'backbone' in name:
                continue
            elif 'neck' in name:
                continue
            else:
                if hasattr(module, 'weight') and module.weight.grad is None:
                    unused_params.append(f"{name}.weight")
                if hasattr(module, 'bias') and module.bias is not None and module.bias.grad is None:
                    unused_params.append(f"{name}.bias")

        print('_________________________________________________________')
        for unused_param in unused_params:
            print(unused_param)
