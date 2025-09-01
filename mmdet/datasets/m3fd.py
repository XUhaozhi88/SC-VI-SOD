# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
from typing import List, Optional, Union

from mmengine.fileio import get_local_path, join_path
from mmengine.utils import is_abs

from mmdet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class M3FDDataset(BaseDetDataset):
    """Dataset for M3FD."""

    METAINFO = {
        'classes':
        ('People', 'Car', 'Lamp', 'Bus', 'Motorcycle', 'Truck'), # 6
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100)]
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

    def __init__(self,
                 *args,
                 ann_file: Union[str, dict],
                 seg_map_suffix: str = '.png',
                 proposal_file: Optional[str] = None,
                 file_client_args: dict = None,
                 backend_args: dict = None,
                 return_classes: bool = False,
                 caption_prompt: Optional[dict] = None,
                 **kwargs) -> None:
        self.seg_map_suffix = seg_map_suffix
        self.proposal_file = proposal_file
        self.backend_args = backend_args
        self.return_classes = return_classes
        self.caption_prompt = caption_prompt
        if self.caption_prompt is not None:
            assert self.return_classes, \
                'return_classes must be True when using caption_prompt'
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )        
        super().__init__(*args, ann_file=ann_file, **kwargs)

    def _join_prefix(self):
        # Automatically join annotation file path with `self.root` if
        # `self.ann_file` is not an absolute path.
        for mod, ann_file in self.ann_file.items():
            if ann_file and not is_abs(ann_file) and self.data_root:
                self.ann_file[mod] = join_path(self.data_root, ann_file)
        # Automatically join data directory with `self.root` if path value in
        # `self.data_prefix` is not an absolute path.
        for data_key, prefix in self.data_prefix.items():
            if not isinstance(prefix, str):
                raise TypeError('prefix should be a string, but got '
                                f'{type(prefix)}')
            if not is_abs(prefix) and self.data_root:
                self.data_prefix[data_key] = join_path(self.data_root, prefix)
            else:
                self.data_prefix[data_key] = prefix

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        # rgb "id": 190001 / "file_name": "visible/{train_test}" + "190001.jpg"
        # ir  "id": 190001 / "file_name": "infrared/{train_test}" + "190001.jpg"

        for mod, ann_file in self.ann_file.items():
            with get_local_path(ann_file, backend_args=self.backend_args) as local_path:
                if mod == 'img':
                        self.coco = self.COCOAPI(local_path)
                elif mod == 'ir_img':
                        self.ir_coco = self.COCOAPI(local_path)
                else:
                    raise AssertionError        
        
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=self.metainfo['classes'])
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            # img
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            # ir
            ir_img_id = img_id
            ir_raw_img_info = self.ir_coco.load_imgs([ir_img_id])[0]
            ir_raw_img_info['ir_img_id'] = ir_img_id
            
            ir_ann_ids = self.ir_coco.get_ann_ids(img_ids=[ir_img_id])
            ir_raw_ann_info = self.ir_coco.load_anns(ir_ann_ids)
            for ann_info in ir_raw_ann_info:
                ann_info['is_ir'] = True
            # raw_ann_info.extend(ir_raw_ann_info)
            # total_ann_ids.extend(ir_ann_ids)

            assert raw_img_info['file_name'].replace('visible', 'infrared') == ir_raw_img_info['file_name']
            parsed_data_info = self.parse_data_info({
                'raw_ann_info': raw_ann_info,
                'raw_img_info': raw_img_info,
                # 'ir_img_raw_ann_info': ir_raw_ann_info,
                # 'ir_img_raw_img_info': ir_raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list
    
    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].rsplit('.', 1)[0] + self.seg_map_suffix)
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']
            
            # mark annotations of ir images
            instance['is_ir'] = ann.get('is_ir', False)

            instances.append(instance)
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos
