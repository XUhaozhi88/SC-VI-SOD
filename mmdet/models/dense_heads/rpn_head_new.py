# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.ops import batched_nms
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (cat_boxes, empty_box_as, get_box_tensor,
                                   get_box_wh, scale_boxes)
from ..utils import images_to_levels, multi_apply, unpack_gt_instances
from mmdet.utils import InstanceList, MultiConfig, OptInstanceList
from .rpn_head import RPNHead


@MODELS.register_module()
class RPNHeadNew(RPNHead):

    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        super()._init_layers()
        reg_dim = self.bbox_coder.encode_size
        self.rpn_reg = nn.Conv2d(self.feat_channels,
                                 self.num_base_priors * reg_dim, 1)
                                #  self.num_base_priors * reg_dim * 2, 1)
        self.rpn_reg_var = nn.Conv2d(self.feat_channels, self.num_base_priors * reg_dim, 1)
        nn.init.constant_(self.rpn_reg_var.bias, 0.0)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level \
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale \
                    level, the channels number is num_base_priors * 4.
        """
        x = self.rpn_conv(x)
        x = F.relu(x)
        rpn_cls_score = self.rpn_cls(x)
        rpn_bbox_pred = self.rpn_reg(x)
        rpn_bbox_var = torch.exp(self.rpn_reg_var(x))
        return rpn_cls_score, rpn_bbox_pred, rpn_bbox_var

    def loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor, bbox_var: Tensor,
                            anchors: Tensor, labels: Tensor,
                            label_weights: Tensor, bbox_targets: Tensor,
                            bbox_weights: Tensor, avg_factor: int) -> tuple:
        """Calculate the loss of a single scale level based on the features
        extracted by the detection head.

        Args:
            cls_score (Tensor): Box scores for each scale level
                Has shape (N, num_anchors * num_classes, H, W).
            bbox_pred (Tensor): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W).
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor
                weight shape (N, num_total_anchors, 4).
            bbox_weights (Tensor): BBox regression loss weights of each anchor
                with shape (N, num_total_anchors, 4).
            avg_factor (int): Average factor that is used to average the loss.

        Returns:
            tuple: loss components.
        """
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=avg_factor)
        # regression loss
        target_dim = bbox_targets.size(-1)
        bbox_targets = bbox_targets.reshape(-1, target_dim)
        bbox_weights = bbox_weights.reshape(-1, target_dim)
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1,
                                                 self.bbox_coder.encode_size)# * 2)
        # bbox_var = torch.exp(bbox_pred[:, 4:])
        # bbox_pred = bbox_pred[:, :4]
        bbox_var = bbox_var.permute(0, 2, 3,
                                      1).reshape(-1,
                                                 self.bbox_coder.encode_size)
        if self.reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            anchors = anchors.reshape(-1, anchors.size(-1))
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)
            bbox_pred = get_box_tensor(bbox_pred)
        loss_bbox = self.loss_bbox(
            bbox_pred, bbox_var, bbox_targets, bbox_weights, avg_factor=avg_factor)
        return loss_cls, loss_bbox

    def loss_by_feat(self,
                     cls_scores: List[Tensor],
                     bbox_preds: List[Tensor],
                     bbox_vars: List[Tensor],
                     batch_gt_instances: InstanceList,
                     batch_img_metas: List[dict],
                     batch_gt_instances_ignore: OptInstanceList = None) \
            -> dict:
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels

        device = cls_scores[0].device

        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, batch_img_metas, device=device)
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         avg_factor) = cls_reg_targets

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(cat_boxes(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        losses_cls, losses_bbox = multi_apply(
            self.loss_by_feat_single,
            cls_scores,
            bbox_preds,
            bbox_vars,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            avg_factor=avg_factor)

        return dict(
            loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                score_factor_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                img_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                with_nms: bool = True) -> InstanceData:
        """Transform a single image's features extracted from the head into
        bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Be compatible with
                BaseDenseHead. Not used in RPNHead.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            img_meta (dict): Image meta info.
            cfg (ConfigDict, optional): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        level_ids = []
        for level_idx, (cls_score, bbox_pred, priors) in \
                enumerate(zip(cls_score_list, bbox_pred_list,
                              mlvl_priors)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            reg_dim = self.bbox_coder.encode_size
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, reg_dim)
            # bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, reg_dim * 2)[:, :4]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                # remind that we set FG labels to [0] since mmdet v2.0
                # BG cat_id: 1
                scores = cls_score.softmax(-1)[:, :-1]

            scores = torch.squeeze(scores)
            if 0 < nms_pre < scores.shape[0]:
                # sort is faster than topk
                # _, topk_inds = scores.topk(cfg.nms_pre)
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[:nms_pre]
                scores = ranked_scores[:nms_pre]
                bbox_pred = bbox_pred[topk_inds, :]
                priors = priors[topk_inds]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)

            # use level id to implement the separate level nms
            level_ids.append(
                scores.new_full((scores.size(0), ),
                                level_idx,
                                dtype=torch.long))

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = bboxes
        results.scores = torch.cat(mlvl_scores)
        results.level_ids = torch.cat(level_ids)

        return self._bbox_post_process(
            results=results, cfg=cfg, rescale=rescale, img_meta=img_meta)

    def loss_and_predict(
        self,
        x: Tuple[Tensor],
        batch_data_samples: SampleList,
        proposal_cfg: Optional[ConfigDict] = None
    ) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        outputs = unpack_gt_instances(batch_data_samples)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = outputs

        outs = self(x)

        loss_inputs = outs + (batch_gt_instances, batch_img_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        
        predictions = self.predict_by_feat(
            *outs[:2], batch_img_metas=batch_img_metas, cfg=proposal_cfg)
        return losses, predictions