# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence

import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmengine.structures import InstanceData, PixelData

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample, ReIDDataSample, TrackDataSample
from mmdet.structures.bbox import BaseBoxes



"""
@author：ty Zhao
"""
from .formatting import PackDetInputs
@TRANSFORMS.register_module()
class PackPairedImagesDetInputs(PackDetInputs):
    def __init__(self, meta_keys=('img_id', 'img_path', 'img_lwir_path',
                                  'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction')):
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:            dict:

            - 'inputs' (obj:`torch.Tensor`): The forward data of models.
            - 'data_sample' (obj:`DetDataSample`): The annotation info of the
                sample.
        """

        packed_results = dict()
        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            # To improve the computational speed by by 3-5 times, apply:
            # If image is not contiguous, use
            # `numpy.transpose()` followed by `numpy.ascontiguousarray()`
            # If image is already contiguous, use
            # `torch.permute()` followed by `torch.contiguous()`
            # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
            # for more details
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()

            packed_results['inputs'] = img

        if 'img_lwir' in results:
            img_lwir = results['img_lwir']
            if len(img_lwir.shape) < 3:
                img_lwir = np.expand_dims(img_lwir, -1)

            if not img_lwir.flags.c_contiguous:
                img_lwir = np.ascontiguousarray(img_lwir.transpose(2, 0, 1))
                img_lwir = to_tensor(img_lwir)
            else:
                img_lwir = to_tensor(img_lwir).permute(2, 0, 1).contiguous()
            packed_results['inputs_lwir'] = img_lwir

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        data_sample = DetDataSample()
        instance_data = InstanceData()
        ignore_instance_data = InstanceData()
        # import pdb
        # pdb.set_trace()
        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = results[key][valid_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = results[key][ignore_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][valid_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        results[key][ignore_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        data_sample.ignored_instances = ignore_instance_data

        if 'proposals' in results:
            proposals = InstanceData(
                bboxes=to_tensor(results['proposals']),
                scores=to_tensor(results['proposals_scores']))
            data_sample.proposals = proposals

        if 'gt_seg_map' in results:
            gt_sem_seg_data = dict(
                sem_seg=to_tensor(results['gt_seg_map'][None, ...].copy()))
            gt_sem_seg_data = PixelData(**gt_sem_seg_data)
            if 'ignore_index' in results:
                metainfo = dict(ignore_index=results['ignore_index'])
                gt_sem_seg_data.set_metainfo(metainfo)
            data_sample.gt_sem_seg = gt_sem_seg_data

        img_meta = {}
        for key in self.meta_keys:
            assert key in results, f'`{key}` is not found in `results`, ' \
                                   f'the valid keys are {list(results)}.'
            img_meta[key] = results[key]

        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample

        return packed_results  # 输入模型的字典?