# Copyright (c) OpenMMLab. All rights reserved.
from .coco_api import COCO, COCOeval, COCOPanoptic
from .cocoeval_mp import COCOevalMP

from .cocoeval_small import COCOevalSmall
from .cocoeval_safit import COCOevalSAFit


__all__ = ['COCO', 'COCOeval', 'COCOPanoptic', 'COCOevalMP', 'COCOevalSmall', 'COCOevalSAFit']
