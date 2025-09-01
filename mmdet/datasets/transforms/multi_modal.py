# Copyright (c) XHZ. All rights reserved.
import torch
from typing import Dict, List, Optional, Tuple, Union
import os
import random
import glob

import mmcv
import numpy as np
from mmcv.transforms import (LoadImageFromFile, to_tensor, BaseTransform,
                             RandomChoice)
from mmcv.transforms.utils import cache_randomness
from mmengine.fileio import get
from mmengine.structures import InstanceData, PixelData

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample
from mmdet.structures.bbox import autocast_box_type, BaseBoxes

from .transforms import Resize, RandomFlip, imrescale, RandomCrop
from .formatting import PackDetInputs


@TRANSFORMS.register_module()
class LoadMultiModalImages(LoadImageFromFile):
    def __init__(
            self, 
            mod_path_mapping_dict: dict = {
                "RGBT-Tiny": {
                    "img": None,
                    "ir_img": None
                }
            },
            mod_list: list = ["img", "ir_img"],
            is_replace_img=False,
            replace_str=['/00/', '/01/'],
            *args,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod_path_mapping_dict = mod_path_mapping_dict
        self.mod_list = mod_list
        self.is_replace_img = is_replace_img
        self.replace_str = replace_str
    
    def transform(self, results: dict) -> Optional[dict]:
        results = super().transform(results)

        filename = results['img_path']

        img_fields_list = []
        valid_image_list = []
        for dataset_key, dataset_dict in self.mod_path_mapping_dict.items():
            if dataset_key in filename:
                for mod_key, mod_value in dataset_dict.items():
                    org_key = mod_value['org_key']
                    target_key = mod_value['target_key']

                    if org_key in filename:
                        curr_filename = filename.replace(org_key, target_key)
                        if mod_key == 'img':
                            continue
                            # img_bytes = get(curr_filename, backend_args=self.backend_args)
                            # img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
                        elif mod_key == 'ir_img':
                            curr_filename = curr_filename.replace(self.replace_str[0], self.replace_str[1])
                            img_bytes = get(curr_filename, backend_args=self.backend_args)
                            img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)
                            results['ir_img_path'] = curr_filename
                        else:
                            img_bytes = get(curr_filename, backend_args=self.backend_args)
                            img = mmcv.imfrombytes(img_bytes, flag=self.color_type, backend=self.imdecode_backend)

                        if self.is_replace_img:
                            mod_key = 'img'

                    img_fields_list.append(mod_key)
                    valid_image_list.append(mod_key)

                    results[mod_key] = img.astype(np.float32) if self.to_float32 else img
                    results[f'{mod_key}_shape'] = img.shape[:2]
                    results[f'{mod_key}_ori_shape'] = img.shape[:2]
                    results[f'{mod_key}_height'] = img.shape[0]
                    results[f'{mod_key}_width'] = img.shape[1]
        return results
    

@TRANSFORMS.register_module()
class LoadContrastImages(BaseTransform):
    def __init__(self, con_mod='ir_img', interval=30, mode='easy') -> None:
        self.con_mod = con_mod
        self.interval = interval
        self.mode = mode

    def easy_select(self, img_path, cite, prefix):        
        # 选择不是同一个视频序列的图像作为负样本
        img_pkg = img_path.split('/')[-3]
        images_pkgs = os.path.join('/', *img_path.split('/')[:-3])
        images_pkgs = os.listdir(images_pkgs)
        images_pkgs.remove(img_pkg)
        images_pkg = random.choice(images_pkgs) # 随机选取视频序列
        images_path = glob.glob(os.path.join('/', *img_path.split('/')[:-3], images_pkg, cite, f'*.{prefix}'))
        image_path = random.choice(images_path) # 随机选取图片
        return mmcv.imread(image_path)
    
    def hard_select(self, img_path, cite, prefix):
        # 选择同一个视频序列的图像（间隔一定距离）作为负样本   
        images_path = glob.glob(os.path.join('/', *img_path.split('/')[:-2], cite, f'*.{prefix}'))
        img_path_id = int(os.path.basename(img_path).split('.')[0])
        while True:
            image_path = random.choice(images_path) # 随机选取图片
            image_path_id = int(os.path.basename(image_path).split('.')[0])
            if abs(img_path_id - image_path_id) >= self.interval: break
        return mmcv.imread(image_path)

    def transform(self, results: dict) -> None:
        # 1. choose video sequence randomly, without current sequence
        # 2. choose image in random video sequence as contrastive image
        if self.con_mod == 'img':
            img_path = results['img_path']
            cite = '00'
        elif self.con_mod == 'ir_img':
            img_path = results['ir_img_path']
            cite = '01'
        
        prefix = img_path.split('.')[-1]    
        results['con_img'] = []
        if isinstance(self.mode, str):
            self.mode = [self.mode]
        for mode in self.mode:
            if mode == 'easy':
                con_image = self.easy_select(img_path, cite, prefix)
            elif mode == 'hard':    
                con_image = self.hard_select(img_path, cite, prefix)
            else:
                raise AssertionError
            results['con_img'].append(con_image)
        return results


@TRANSFORMS.register_module()
class MultiModalResize(Resize):
    def __init__(self, *args, 
                 mod_list, 
                 is_fixscale=False,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod_list = mod_list
        self.is_fixscale = is_fixscale

    def _resize_img_single(self, img, scale):   
        if self.keep_ratio:     
            h, w = img.shape[:2]
            if self.is_fixscale:
                imrescale_func = imrescale
            else:
                imrescale_func = mmcv.imrescale
            img, _ = imrescale_func(
                img=img,
                scale=scale,
                interpolation=self.interpolation,
                return_scale=True,
                backend=self.backend)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                img=img,
                size=scale,
                interpolation=self.interpolation,
                return_scale=True,
                backend=self.backend)
            
        return img, (w_scale, h_scale)

    def _resize_img(self, results: dict) -> None:
        for mod in self.mod_list:
            if results.get(mod, None) is not None:
                img = results[mod]
                scale = results['scale']
                if isinstance(img, list):
                    img = [self._resize_img_single(img1, scale)[0] for img1 in img]
                else:
                    img, scale_factor = self._resize_img_single(img, scale)
                results[mod] = img
                if mod == 'img':
                    results['img_shape'] = img.shape[:2]
                    results['scale_factor'] = scale_factor
                    results['keep_ratio'] = self.keep_ratio

    def _resize_img_old(self, results: dict) -> None:
        for mod in self.mod_list:
            if results.get(mod, None) is not None:
                if self.keep_ratio:
                    img, scale_factor = mmcv.imrescale(
                        img=results[mod],
                        scale=results['scale'],
                        interpolation=self.interpolation,
                        return_scale=True,
                        backend=self.backend)
                    # the w_scale and h_scale has minor difference
                    # a real fix should be done in the mmcv.imrescale in the future
                    new_h, new_w = img.shape[:2]
                    h, w = results[mod].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = mmcv.imresize(
                        img=results[mod],
                        size=results['scale'],
                        interpolation=self.interpolation,
                        return_scale=True,
                        backend=self.backend)
                results[mod] = img
                if mod == 'img':
                    results['img_shape'] = img.shape[:2]
                    results['scale_factor'] = (w_scale, h_scale)
                    results['keep_ratio'] = self.keep_ratio


@TRANSFORMS.register_module()
class MultiModalRandomChoice(RandomChoice):
    def __init__(self, *args, mod_list, **kwargs):
        super().__init__(*args, **kwargs)
        self.mod_list = mod_list

    def __iter__(self):
        return iter(self.transforms)

    @cache_randomness
    def random_pipeline_index(self) -> int:
        """Return a random transform index."""
        indices = np.arange(len(self.transforms))
        return np.random.choice(indices, p=self.prob)

    def transform(self, results: Dict) -> Optional[Dict]:
        """Randomly choose a transform to apply."""
        idx = self.random_pipeline_index()
        for mod in self.mod_list:
            pass
        return self.transforms[idx](results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms = {self.transforms}'
        repr_str += f'prob = {self.prob})'
        return repr_str


@TRANSFORMS.register_module()
class MultiModalRandomCrop(RandomCrop):
    def __init__(self, *args, mod_list, **kwargs):
        super().__init__(*args, **kwargs)
        self.mod_list = mod_list

    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool) -> Union[dict, None]:
        assert crop_size[0] > 0 and crop_size[1] > 0

        img = results['img']
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
            dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

        # crop the image        
        for mod in self.mod_list:
            img = results[mod]
            img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]            
            results[mod] = img
            if mod == 'img':
                img_shape = img.shape
                results['img_shape'] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offset_w, -offset_h])
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            valid_inds = bboxes.is_inside(img_shape[:2]).numpy()
            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not allow_negative_crop):
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                        type(results['gt_bboxes']))

            # We should remove the instance ids corresponding to invalid boxes.
            if results.get('gt_instances_ids', None) is not None:
                results['gt_instances_ids'] = \
                    results['gt_instances_ids'][valid_inds]

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                                          crop_x1:crop_x2]

        return results

@TRANSFORMS.register_module()
class MultiModalRandomFlip(RandomFlip):
    def __init__(self, *args, mod_list, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod_list = mod_list

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        for mod in self.mod_list:
            img = results[mod]
            if isinstance(img, list):
                img = [mmcv.imflip(img1, direction=results['flip_direction']) for img1 in img]
            else:
                img = mmcv.imflip(img, direction=results['flip_direction'])
            results[mod] = img
        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'].flip_(img_shape, results['flip_direction'])

        # flip masks
        if results.get('gt_masks', None) is not None:
            results['gt_masks'] = results['gt_masks'].flip(
                results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

        # record homography matrix for flip
        self._record_homography_matrix(results)


@TRANSFORMS.register_module()
class PackMultiModalDetInputs(PackDetInputs):
    def __init__(self, *args, mod_list, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.mod_list = mod_list

    def transform(self, results: dict) -> dict:
        '''
            This method is a trick. We concatenate RGB img and IR img as a Tensor.
        '''
        def process(img):            
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                img = to_tensor(img)
            else:
                img = to_tensor(img).permute(2, 0, 1).contiguous()
            return img

        packed_results = self.transform_mmdet(results)
        for mod in self.mod_list:
            if mod == 'img':
                continue
            else:
                img = results[mod]
                if isinstance(img, list):
                    img = [process(img1) for img1 in img]
                else:
                    img = process(img)
                packed_results[f'{mod}_inputs'] = img
                # packed_results['inputs'] = torch.cat((packed_results['inputs'], img), dim=0)

        return packed_results
    
    def transform_mmdet(self, results: dict) -> dict:
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

        if 'gt_ignore_flags' in results:
            valid_idx = np.where(results['gt_ignore_flags'] == 0)[0]
            ignore_idx = np.where(results['gt_ignore_flags'] == 1)[0]

        valid_instances = [results['instances'][i] for i in valid_idx.tolist()]
        ignore_instances = [results['instances'][i] for i in ignore_idx.tolist()]
        valid_rgb_idx = []
        valid_ir_idx = []
        for i, instance in enumerate(valid_instances):
            if instance['is_ir']:
                valid_ir_idx.append(i)
            else:
                valid_rgb_idx.append(i)
        valid_rgb_idx = np.array(valid_rgb_idx, dtype=np.int64)
        valid_ir_idx = np.array(valid_ir_idx, dtype=np.int64)

        ignore_rgb_idx = []
        ignore_ir_idx = []
        for i, instance in enumerate(ignore_instances):
            if instance['is_ir']:
                ignore_ir_idx.append(i)
            else:
                ignore_rgb_idx.append(i)
        ignore_rgb_idx = np.array(ignore_rgb_idx, dtype=np.int64)
        ignore_ir_idx = np.array(ignore_ir_idx, dtype=np.int64)

        data_sample = DetDataSample()
        ir_data_sample = DetDataSample()
        instance_data = InstanceData()
        ir_instance_data = InstanceData()
        ignore_instance_data = InstanceData()
        ir_ignore_instance_data = InstanceData()

        for i, key in enumerate(self.mapping_table.keys()):
            if key not in results:
                continue
            valid = results[key][valid_idx]
            ignore = results[key][ignore_idx]
            if key == 'gt_masks' or isinstance(results[key], BaseBoxes):
                if 'gt_ignore_flags' in results:
                    instance_data[
                        self.mapping_table[key]] = valid[valid_rgb_idx]
                    ir_instance_data[
                        self.mapping_table[key]] = valid[valid_ir_idx]
                    ignore_instance_data[
                        self.mapping_table[key]] = ignore[ignore_rgb_idx]
                    ir_ignore_instance_data[
                        self.mapping_table[key]] = ignore[ignore_ir_idx]
                else:
                    instance_data[self.mapping_table[key]] = results[key]
            else:
                if 'gt_ignore_flags' in results:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        valid[valid_rgb_idx])
                    ir_instance_data[self.mapping_table[key]] = to_tensor(
                        valid[valid_ir_idx])
                    ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        ignore[ignore_rgb_idx])
                    ir_ignore_instance_data[self.mapping_table[key]] = to_tensor(
                        ignore[ignore_ir_idx])
                else:
                    instance_data[self.mapping_table[key]] = to_tensor(
                        results[key])
        data_sample.gt_instances = instance_data
        ir_data_sample.gt_instances = ir_instance_data
        data_sample.ignored_instances = ignore_instance_data
        ir_data_sample.ignored_instances = ir_ignore_instance_data

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
            if key in results:
                img_meta[key] = results[key]
        data_sample.set_metainfo(img_meta)
        packed_results['data_samples'] = data_sample
        packed_results['ir_data_samples'] = ir_data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(meta_keys={self.meta_keys})'
        return repr_str