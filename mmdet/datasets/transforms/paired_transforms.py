# Copyright (c) OpenMMLab. All rights reserved.
import copy
import inspect
import math
import warnings
from typing import List, Optional, Sequence, Tuple, Union,Dict
import mmengine
import cv2
import mmcv
import numpy
import numpy as np
from mmcv.image import imresize
from mmcv.image.geometric import _scale_size
from mmcv.transforms import BaseTransform
from mmcv.transforms import Pad as MMCV_Pad
from mmcv.transforms import RandomFlip as MMCV_RandomFlip
from mmcv.transforms import Resize as MMCV_Resize
from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from mmengine.dataset import BaseDataset
from mmengine.utils import is_str
from numpy import random

from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
from mmdet.utils import log_img_scale

try:
    from imagecorruptions import corrupt
except ImportError:
    corrupt = None

try:
    import albumentations
    from albumentations import Compose
except ImportError:
    albumentations = None
    Compose = None

from .wrappers import Compose

Number = Union[int, float]




"""
@author：ty Zhao
"""
from .transforms import Resize
@TRANSFORMS.register_module()
class PairedImagesResize(Resize):
    def _resize_img(self, results: dict) -> None:
        if results.get('img', None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)

                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h

            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio

        if results.get('img_lwir', None) is not None:
            if self.keep_ratio:
                img_lwir, scale_factor = mmcv.imrescale(
                    results['img_lwir'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img_lwir.shape[:2]
                h, w = results['img_lwir'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:

                img_lwir, w_scale, h_scale = mmcv.imresize(
                    results['img_lwir'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img_lwir'] = img_lwir
            # results['img_shape'] = img.shape[:2]
            # results['scale_factor'] = (w_scale, h_scale)
            #
            # results['keep_ratio'] = self.keep_ratio


from mmcv.transforms import RandomResize as MMCV_RandomResize

"""
@author：ty Zhao
"""
@TRANSFORMS.register_module()
class PairedImagesRandomResize(MMCV_RandomResize):
    def __init__(
            self,
            scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
            ratio_range: Tuple[float, float] = None,
            resize_type: str = 'PairedImagesResize',  # 原来是 Resize, 相当于是 wrapper
            **resize_kwargs,
    ) -> None:
        self.scale = scale
        self.ratio_range = ratio_range

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Reisize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})


"""
@author：ty Zhao
"""
from .transforms import RandomFlip
@TRANSFORMS.register_module()
class PairedImageRandomFlip(RandomFlip):
    """Flip the image & bbox & mask & segmentation map. Added or Updated keys:
    flip, flip_direction, img, gt_bboxes, and gt_seg_map. There are 3 flip
    modes:

     - ``prob`` is float, ``direction`` is string: the image will be
         ``direction``ly flipped with probability of ``prob`` .
         E.g., ``prob=0.5``, ``direction='horizontal'``,
         then image will be horizontally flipped with probability of 0.5.
     - ``prob`` is float, ``direction`` is list of string: the image will
         be ``direction[i]``ly flipped with probability of
         ``prob/len(direction)``.
         E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
         then image will be horizontally flipped with probability of 0.25,
         vertically with probability of 0.25.
     - ``prob`` is list of float, ``direction`` is list of string:
         given ``len(prob) == len(direction)``, the image will
         be ``direction[i]``ly flipped with probability of ``prob[i]``.
         E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
         'vertical']``, then image will be horizontally flipped with
         probability of 0.3, vertically with probability of 0.5.


    Required Keys:
    - img
    - img_lwir
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:
    - img
    - img_lwir
    - gt_bboxes
    - gt_masks
    - gt_seg_map

    Added Keys:
    - flip
    - flip_direction
    - homography_matrix

    Args:
         prob (float | list[float], optional): The flipping probability.
             Defaults to None.
         direction(str | list[str]): The flipping direction. Options
             If input is a list, the length must equal ``prob``. Each
             element in ``prob`` indicates the flip probability of
             corresponding direction. Defaults to 'horizontal'.
    """

    @autocast_box_type()
    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, and semantic segmentation map."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])
        results['img_lwir'] = mmcv.imflip(
            results['img_lwir'], direction=results['flip_direction'])

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


"""
@author：ty Zhao
"""
from .transforms import Pad
@TRANSFORMS.register_module()
class PairedImagesPad(Pad):
    def _pad_img(self, results: dict) -> None:
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)  # 使用某个值去 pad

        size = None
        if self.pad_to_square:
            max_size = max(results['img'].shape[:2])
            size = (max_size, max_size)
        if self.size_divisor is not None:
            if size is None:
                size = (results['img'].shape[0], results['img'].shape[1])
            pad_h = int(np.ceil(
                size[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(
                size[1] / self.size_divisor)) * self.size_divisor
            size = (pad_h, pad_w)
        elif self.size is not None:
            size = self.size[::-1]
        if isinstance(pad_val, int) and results['img'].ndim == 3:
            pad_val = tuple(pad_val for _ in range(results['img'].shape[2]))
        padded_img = mmcv.impad(
            results['img'],
            shape=size,
            pad_val=pad_val,
            padding_mode=self.padding_mode)

        if isinstance(pad_val, int) and results['img_lwir'].ndim == 3:
            pad_val = tuple(pad_val for _ in range(results['img_lwir'].shape[2]))

        padded_img_lwir = mmcv.impad(
            results['img_lwir'],
            shape=size,
            pad_val=pad_val,
            padding_mode=self.padding_mode)

        results['img'] = padded_img
        results['img_lwir'] = padded_img_lwir
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        results['img_shape'] = padded_img.shape[:2]


"""
@author：ty Zhao
"""
from .transforms import RandomAffine
@TRANSFORMS.register_module()
class PairedImageRandomAffine(RandomAffine):
    """Random affine transform data augmentation.

    This operation randomly generates affine transform matrix which including
    rotation, translation, shear and scaling transforms.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_ignore_flags (optional)

    Args:
        max_rotate_degree (float): Maximum degrees of rotation transform.
            Defaults to 10.
        max_translate_ratio (float): Maximum ratio of translation.
            Defaults to 0.1.
        scaling_ratio_range (tuple[float]): Min and max ratio of
            scaling transform. Defaults to (0.5, 1.5).
        max_shear_degree (float): Maximum degrees of shear
            transform. Defaults to 2.
        border (tuple[int]): Distance from width and height sides of input
            image to adjust output shape. Only used in mosaic dataset.
            Defaults to (0, 0).
        border_val (tuple[int]): Border padding values of 3 channels.
            Defaults to (114, 114, 114).
        bbox_clip_border (bool, optional): Whether to clip the objects outside
            the border of the image. In some dataset like MOT17, the gt bboxes
            are allowed to cross the border of images. Therefore, we don't
            need to clip the gt bboxes in these cases. Defaults to True.
    """

    def __init__(self,
                 max_rotate_degree: float = 10.0,
                 max_translate_ratio: float = 0.1,
                 scaling_ratio_range: Tuple[float, float] = (0.5, 1.5),
                 max_shear_degree: float = 2.0,
                 border: Tuple[int, int] = (0, 0),
                 border_val: Tuple[int, int, int] = (114, 114, 114),
                 bbox_clip_border: bool = True) -> None:
        assert 0 <= max_translate_ratio <= 1
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.max_rotate_degree = max_rotate_degree
        self.max_translate_ratio = max_translate_ratio
        self.scaling_ratio_range = scaling_ratio_range
        self.max_shear_degree = max_shear_degree
        self.border = border
        self.border_val = border_val
        self.bbox_clip_border = bbox_clip_border

    @cache_randomness
    def _get_random_homography_matrix(self, height, width):
        # Rotation
        rotation_degree = random.uniform(-self.max_rotate_degree,
                                         self.max_rotate_degree)
        rotation_matrix = self._get_rotation_matrix(rotation_degree)

        # Scaling
        scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                       self.scaling_ratio_range[1])
        scaling_matrix = self._get_scaling_matrix(scaling_ratio)

        # Shear
        x_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        y_degree = random.uniform(-self.max_shear_degree,
                                  self.max_shear_degree)
        shear_matrix = self._get_shear_matrix(x_degree, y_degree)

        # Translation
        trans_x = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * width
        trans_y = random.uniform(-self.max_translate_ratio,
                                 self.max_translate_ratio) * height
        translate_matrix = self._get_translation_matrix(trans_x, trans_y)

        warp_matrix = (
            translate_matrix @ shear_matrix @ rotation_matrix @ scaling_matrix)
        return warp_matrix

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        img = results['img']

        height = img.shape[0] + self.border[1] * 2
        width = img.shape[1] + self.border[0] * 2

        warp_matrix = self._get_random_homography_matrix(height, width)

        img = cv2.warpPerspective(
            img,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img'] = img
        results['img_shape'] = img.shape[:2]

        bboxes = results['gt_bboxes']
        num_bboxes = len(bboxes)
        if num_bboxes:
            bboxes.project_(warp_matrix)
            if self.bbox_clip_border:
                bboxes.clip_([height, width])
            # remove outside bbox
            valid_index = bboxes.is_inside([height, width]).numpy()
            results['gt_bboxes'] = bboxes[valid_index]
            results['gt_bboxes_labels'] = results['gt_bboxes_labels'][
                valid_index]
            results['gt_ignore_flags'] = results['gt_ignore_flags'][
                valid_index]

            if 'gt_masks' in results:
                raise NotImplementedError('RandomAffine only supports bbox.')

        img_lwir = results['img_lwir']

        img_lwir = cv2.warpPerspective(
            img_lwir,
            warp_matrix,
            dsize=(width, height),
            borderValue=self.border_val)
        results['img_lwir'] = img_lwir


        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(max_rotate_degree={self.max_rotate_degree}, '
        repr_str += f'max_translate_ratio={self.max_translate_ratio}, '
        repr_str += f'scaling_ratio_range={self.scaling_ratio_range}, '
        repr_str += f'max_shear_degree={self.max_shear_degree}, '
        repr_str += f'border={self.border}, '
        repr_str += f'border_val={self.border_val}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str



"""
@author：ty Zhao
"""
from mmcv.transforms import MultiScaleFlipAug
@TRANSFORMS.register_module()
class PairedImageMultiScaleFlipAug(MultiScaleFlipAug):
    """Test-time augmentation with multiple scales and flipping.

    An example configuration is as followed:

    .. code-block::

        dict(
            type='MultiScaleFlipAug',
            scales=[(1333, 400), (1333, 800)],
            flip=True,
            transforms=[
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=1),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])

    ``results`` will be resized using all the sizes in ``scales``.
    If ``flip`` is True, then flipped results will also be added into output
    list.

    For the above configuration, there are four combinations of resize
    and flip:

    - Resize to (1333, 400) + no flip
    - Resize to (1333, 400) + flip
    - Resize to (1333, 800) + no flip
    - resize to (1333, 800) + flip

    The four results are then transformed with ``transforms`` argument.
    After that, results are wrapped into lists of the same length as below:

    .. code-block::

        dict(
            inputs=[...],
            data_samples=[...]
        )

    Where the length of ``inputs`` and ``data_samples`` are both 4.

    Required Keys:

    - Depending on the requirements of the ``transforms`` parameter.

    Modified Keys:

    - All output keys of each transform.

    Args:
        transforms (list[dict]): Transforms to be applied to each resized
            and flipped data.
        scales (tuple | list[tuple] | None): Images scales for resizing.
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        allow_flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal", "vertical" and "diagonal". If
            flip_direction is a list, multiple flip augmentations will be
            applied. It has no effect when flip == False. Defaults to
            "horizontal".
        resize_cfg (dict): Base config for resizing. Defaults to
            ``dict(type='Resize', keep_ratio=True)``.
        flip_cfg (dict): Base config for flipping. Defaults to
            ``dict(type='RandomFlip')``.
    """

    def __init__(
        self,
        transforms: List[dict],
        scales: Optional[Union[Tuple, List[Tuple]]] = None,
        scale_factor: Optional[Union[float, List[float]]] = None,
        allow_flip: bool = False,
        flip_direction: Union[str, List[str]] = 'horizontal',
        resize_cfg: dict = dict(type='Resize', keep_ratio=True),
        flip_cfg: dict = dict(type='RandomFlip')
    ) -> None:

        self.transforms = Compose(transforms)  # type: ignore

        if scales is not None:
            self.scales = scales if isinstance(scales, list) else [scales]
            self.scale_key = 'scale'
            assert mmengine.is_list_of(self.scales, tuple)
        else:
            # if ``scales`` and ``scale_factor`` both be ``None``
            if scale_factor is None:
                self.scales = [1.]  # type: ignore
            elif isinstance(scale_factor, list):
                self.scales = scale_factor  # type: ignore
            else:
                self.scales = [scale_factor]  # type: ignore

            self.scale_key = 'scale_factor'

        self.allow_flip = allow_flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmengine.is_list_of(self.flip_direction, str)
        if not self.allow_flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        self.resize_cfg = resize_cfg.copy()
        self.flip_cfg = flip_cfg

    def transform(self, results: dict) -> Dict:
        """Apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: The augmented data, where each value is wrapped
            into a list.
        """

        data_samples = []
        flip_args = [(False, '')]
        if self.allow_flip:
            flip_args += [(True, direction)
                          for direction in self.flip_direction]
        for scale in self.scales:
            for flip, direction in flip_args:
                _resize_cfg = self.resize_cfg.copy()
                _resize_cfg.update({self.scale_key: scale})
                _resize_flip = [_resize_cfg]

                if flip:
                    _flip_cfg = self.flip_cfg.copy()
                    _flip_cfg.update(prob=1.0, direction=direction)
                    _resize_flip.append(_flip_cfg)
                else:
                    results['flip'] = False
                    results['flip_direction'] = None
                resize_flip = Compose(_resize_flip)

                _results = resize_flip(results.copy())

                packed_results = self.transforms(_results)  # type: ignore
                #
                # inputs.append(packed_results['inputs'])  # type: ignore
                # inputs_lwir.append(packed_results['inputs_lwir'])
                inputs = packed_results['inputs']
                inputs_lwir = packed_results['inputs_lwir']
                data_samples = packed_results['data_samples']  # type: ignore
                # print(len(inputs_lwir))

        return {'inputs':inputs,'inputs_lwir':inputs_lwir ,'data_samples':data_samples}

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}'
        repr_str += f', scales={self.scales}'
        repr_str += f', allow_flip={self.allow_flip}'
        repr_str += f', flip_direction={self.flip_direction})'
        return repr_str

