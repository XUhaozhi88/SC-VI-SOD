# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Tuple, Union

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile
from mmengine.fileio import get
from mmengine.structures import BaseDataElement
import mmengine.fileio as fileio
from mmdet.registry import TRANSFORMS
from mmdet.structures.bbox import get_box_type
from mmdet.structures.bbox.box_type import autocast_box_type
from mmdet.structures.mask import BitmapMasks, PolygonMasks
"""
@author：ty Zhao
"""
@TRANSFORMS.register_module()
class LoadPairedImageFromFile(LoadImageFromFile):
    """Load an image from file.

    Required Keys:

    - img_path
    - img_lwir_path

    Modified Keys:

    - img
    - img_lwir
    - img_shape
    - ori_shape

    """
    def transform(self, results):
        """Functions to load RGB image and LWIR image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['img_path']
        filename_lwir = results['img_lwir_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename_lwir )
                img_lwir_bytes = file_client.get(filename_lwir )
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
                img_lwir_bytes = fileio.get(
                    filename_lwir, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
            lwir_img = mmcv.imfrombytes(
                img_lwir_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        assert img is not None, f'failed to load image: {filename}'
        assert lwir_img is not None, f'failed to load LWIR image: {filename_lwir}'

       
        if self.to_float32:
            img = img.astype(np.float32)
            lwir_img = lwir_img.astype(np.float32)

        results['img'] = img
        results['img_lwir'] = lwir_img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]

        return results

"""
@author：ty Zhao
"""
@TRANSFORMS.register_module()
class LoadLwirImageFromFile(LoadImageFromFile):
    """Load an image from file.

    Required Keys:

    - img_lwir_path

    Modified Keys:

    - img
    - img_shape
    - ori_shape

    """

    def transform(self, results):
        """Functions to load RGB image and LWIR image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        filename_lwir = results['img_lwir_path']
        try:
            if self.file_client_args is not None:

                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename_lwir)
                img_lwir_bytes = file_client.get(filename_lwir)
            else:
                img_lwir_bytes = fileio.get(
                    filename_lwir, backend_args=self.backend_args)
            lwir_img = mmcv.imfrombytes(
                img_lwir_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        assert lwir_img is not None, f'failed to load LWIR image: {filename_lwir}'

        if self.to_float32:
            lwir_img = lwir_img.astype(np.float32)

        results['img'] =  lwir_img
        results['img_shape'] = lwir_img.shape[:2]
        results['ori_shape'] = lwir_img.shape[:2]
        return results