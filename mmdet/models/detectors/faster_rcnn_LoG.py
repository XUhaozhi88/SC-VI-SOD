# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import torch
from torch import Tensor, nn

from typing import List

from mmcv.cnn import ConvModule
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .faster_rcnn import FasterRCNN

import cv2
import numpy


@MODELS.register_module()
class FasterRCNN_LoG(FasterRCNN):

    def cam(self):
        pass

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(batch_inputs)

        # If there are no pre-defined proposals, use RPN to get proposals
        if batch_data_samples[0].get('proposals', None) is None:
            rpn_results_list = self.rpn_head.predict(
                x, batch_data_samples, rescale=False)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(
            x, rpn_results_list, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples