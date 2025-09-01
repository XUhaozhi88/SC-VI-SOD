# Copyright (c) OpenMMLab. All rights reserved.
import math
import random
from numbers import Number
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.utils import is_seq_of
from mmengine.model.utils import stack_batch
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.registry import MODELS

from .data_preprocessor import DetDataPreprocessor
import cv2
import os
import numpy

@MODELS.register_module()
class MultiModalDetDataPreprocessor(DetDataPreprocessor):

    def __init__(self, 
                 *args, 
                 mean,
                 std,
                 train_mod_list: list = ["img", "ir_img"],
                 val_mod_list: list = ["img", "ir_img"],
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_all = mean
        self.std_all = std
        self._enable_normalize = True

        self.train_mod_list = train_mod_list
        self.val_mod_list = val_mod_list

    def forward_base(self, data: dict, training: bool = False) -> Union[dict, list]:
        data = self.cast_data(data)  # type: ignore
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_inputs = []
            for _batch_input in _batch_inputs:
                # channel transform
                if self._channel_conversion:
                    _batch_input = _batch_input[[2, 1, 0], ...]
                # Convert to float after channel conversion to ensure
                # efficiency
                _batch_input = _batch_input.float()
                # Normalization.
                if self._enable_normalize:
                    if self.mean.shape[0] == 3:
                        assert _batch_input.dim(
                        ) == 3 and _batch_input.shape[0] == 3, (
                            'If the mean has 3 values, the input tensor '
                            'should in shape of (3, H, W), but got the tensor '
                            f'with shape {_batch_input.shape}')
                    _batch_input = (_batch_input - self.mean) / self.std
                batch_inputs.append(_batch_input)
            # Pad and stack Tensor.
            batch_inputs = stack_batch(batch_inputs, self.pad_size_divisor,
                                       self.pad_value)
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            if self._channel_conversion:
                _batch_inputs = _batch_inputs[:, [2, 1, 0], ...]
            # Convert to float after channel conversion to ensure
            # efficiency
            _batch_inputs = _batch_inputs.float()
            if self._enable_normalize:
                _batch_inputs = (_batch_inputs - self.mean) / self.std
            h, w = _batch_inputs.shape[2:]
            target_h = math.ceil(
                h / self.pad_size_divisor) * self.pad_size_divisor
            target_w = math.ceil(
                w / self.pad_size_divisor) * self.pad_size_divisor
            pad_h = target_h - h
            pad_w = target_w - w
            batch_inputs = F.pad(_batch_inputs, (0, pad_w, 0, pad_h),
                                 'constant', self.pad_value)
        else:
            raise TypeError('Output of `cast_data` should be a dict of '
                            'list/tuple with inputs and data_samples, '
                            f'but got {type(data)}: {data}')
        data['inputs'] = batch_inputs
        data.setdefault('data_samples', None)
        return data

    def forward_single(self, data: dict, training: bool = False):
        batch_pad_shape = self._get_pad_shape(data)
        data = self.forward_base(data=data, training=training)
        # data = super(DetDataPreprocessor, self).forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return inputs, data_samples

    def forward(self, data: dict, training: bool = False) -> dict:
        mod_list = self.train_mod_list if self.training else self.val_mod_list

        # img = data['inputs'][0]
        # ir = data['ir_img_inputs'][0]
        # all = torch.cat((img, ir), dim=2)
        # all = all.permute(1, 2, 0).numpy().astype(numpy.uint8)
        # cv2.cvtColor(all, cv2.COLOR_RGB2GRAY)
        # savePath = os.path.join('/workspace/mmdetection/results/show/', 
        #                         str(data['data_samples'][0].img_id) + '.jpg')
        # cv2.imwrite(filename=savePath, img=all)
        
        for mod in mod_list:
            self.mean = torch.tensor(self.mean_all[mod], device=self.device).view(-1, 1, 1)
            self.std = torch.tensor(self.std_all[mod], device=self.device).view(-1, 1, 1)
            if mod == 'img':
                inputs, data_samples = self.forward_single(data=data, training=training)
            else:
                data_new = copy.deepcopy(data)
                extra_inputs, extra_data_samples = data_new[f'{mod}_inputs'], data['ir_data_samples']
                if mod == 'con_img':
                    inputs_new = []
                    data_samples_new = []
                    for extra_input in extra_inputs:
                        data_new['inputs'] = list(extra_input)
                        input_new, data_sample_new = self.forward_single(data=data_new, training=training)
                        inputs_new.append(input_new)
                        data_samples_new.append(data_sample_new)
                    data[f'{mod}_inputs'] = inputs_new
                    data[f'{mod}_inputs'] = inputs_new
                else:
                    data_new['inputs'] = extra_inputs
                    data[f'{mod}_inputs'], data[f'{mod}_data_samples'] = self.forward_single(data=data_new, training=training)
        
        extra_inputs = [data[f'{mod}_inputs'] for mod in mod_list if mod != 'img']
        
        if training:
            extra_data_samples = data[f'ir_img_data_samples']
            # extra_data_samples = [data[f'{mod}_data_samples'] for mod in mod_list if mod != 'img']
            return {'inputs': inputs, 'data_samples': data_samples, 
                    'extra_inputs': extra_inputs, 'extra_data_samples': extra_data_samples}
        else:            
            return {'inputs': inputs, 'data_samples': data_samples, 'extra_inputs': extra_inputs}

        # if data_samples is not None:
        #     # NOTE the batched image size information may be useful, e.g.
        #     # in DETR, this is needed for the construction of masks, which is
        #     # then used for the transformer_head.
        #     batch_input_shape = tuple(inputs[0].size()[-2:])
        #     for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
        #         data_sample.set_metainfo({
        #             'batch_input_shape': batch_input_shape,
        #             'pad_shape': pad_shape
        #         })
        #     extra_batch_input_shape = tuple(extra_inputs[0].size()[-2:])
        #     for extra_data_sample, extra_pad_shape in zip(extra_data_samples, extra_batch_input_shape):
        #         extra_data_sample.set_metainfo({
        #             'batch_input_shape': extra_batch_input_shape,
        #             'pad_shape': extra_pad_shape
        #         })

        #     if self.boxtype2tensor:
        #         samplelist_boxtype2tensor(data_samples)
        #         samplelist_boxtype2tensor(extra_data_samples)

        #     if self.pad_mask and training:
        #         self.pad_gt_masks(data_samples)
        #         self.pad_gt_masks(extra_data_samples)

        #     if self.pad_seg and training:
        #         self.pad_gt_sem_seg(data_samples)
        #         self.pad_gt_sem_seg(extra_data_samples)

        # if training and self.batch_augments is not None:
        #     for batch_aug in self.batch_augments:
        #         inputs_raw, data_samples = batch_aug(inputs_raw, data_samples)
        #         inputs_raw, data_samples = batch_aug(inputs_raw, extra_data_samples)
        #         extra_inputs_new = []
        #         for extra_input in extra_inputs:
        #             if isinstance(extra_input, list):
        #                 extra_input = [batch_aug(extra_input1, data_samples) for extra_input1 in extra_input]
        #             else:
        #                 extra_input = batch_aug(extra_input, data_samples)
        #             extra_inputs_new.append(extra_input)
        #             extra_inputs = extra_inputs_new
        
        return {'inputs': inputs, 'data_samples': data_samples, 
                'extra_inputs': extra_inputs, 'extra_data_samples': extra_data_samples}