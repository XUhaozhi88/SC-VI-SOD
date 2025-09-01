# Author: Xu Haozhi.
# Time:   2025.05.14

import copy
from typing import Dict, Optional, Tuple, Union

import torch
import torchvision
import torch.nn as nn
from torch.nn import functional as F
from mmengine.runner.amp import autocast
from mmengine.structures import InstanceData
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
from torch import Tensor

from mmcv.ops import batched_nms
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType
from ..layers import SinePositionalEncoding, CdnQueryGenerator
# from ..layers.transformer.dino_layers_fusion_20250512 import (DinoTransformerEncoder_Parallel, 
#                                                               DinoTransformerDecoder_Parallel)
from ..layers.transformer.dino_layers_fusion_20250513 import DinoTransformerEncoder_Parallel
from ..layers.transformer.dino_layers_fusion_20250512 import DinoTransformerDecoder_Parallel
from ..layers.transformer.dino_layers_fusion_20250528 import DinoTransformerDecoder_MultiModal
from .dino import DINO
from .deformable_detr import DeformableDETR


@MODELS.register_module()
class DINO_Parallel_0514(DINO):
    def __init__(self,
                 *args,
                 mod_list=["img", "ir_img"],
                 use_autocast=False,                 
                 mode='loss',
                 extra_return=["ir",],
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 dn_cfg: OptConfigType = None,
                 with_box_refine: bool = False,
                 as_two_stage: bool = False,
                 num_feature_levels: int = 4,
                 extra_num_queries: int = 0,
                 **kwargs) -> None:

        self.mod_list = mod_list        
        # self.num_mod = len(mod_list)
        # self.extra_num_queries = extra_num_queries

        self._special_tokens = '. '
        self.use_autocast = use_autocast
        self.mode = mode
        self.extra_return = extra_return

        # deformable detr init
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels

        if bbox_head is not None:
            assert 'share_pred_layer' not in bbox_head and \
                   'num_pred_layer' not in bbox_head and \
                   'as_two_stage' not in bbox_head, \
                'The two keyword args `share_pred_layer`, `num_pred_layer`, ' \
                'and `as_two_stage are set in `detector.__init__()`, users ' \
                'should not set them in `bbox_head` config.'
            # The last prediction layer is used to generate proposal
            # from encode feature map when `as_two_stage` is `True`.
            # And all the prediction layers should share parameters
            # when `with_box_refine` is `True`.
            bbox_head['share_pred_layer'] = not with_box_refine
            bbox_head['num_pred_layer'] = (decoder['num_layers'] + 1) \
                if self.as_two_stage else decoder['num_layers']
            bbox_head['as_two_stage'] = as_two_stage
            
            bbox_head['num_pred_layer'] += len(extra_return)  # for supervise instance of rgb/ir image

        super(DeformableDETR, self).__init__(*args, decoder=decoder, bbox_head=bbox_head, **kwargs)

        # dino init
        assert self.as_two_stage, 'as_two_stage must be True for DINO'
        assert self.with_box_refine, 'with_box_refine must be True for DINO'

        if dn_cfg is not None:
            assert 'num_classes' not in dn_cfg and \
                   'num_queries' not in dn_cfg and \
                   'hidden_dim' not in dn_cfg, \
                'The three keyword args `num_classes`, `embed_dims`, and ' \
                '`num_matching_queries` are set in `detector.__init__()`, ' \
                'users should not set them in `dn_cfg` config.'
            dn_cfg['num_classes'] = self.bbox_head.num_classes
            dn_cfg['embed_dims'] = self.embed_dims
            dn_cfg['num_matching_queries'] = self.num_queries
        self.dn_query_generator = CdnQueryGenerator(**dn_cfg)
        
        # 初始化结构
        backbone = nn.ModuleList()
        if self.with_neck: neck = nn.ModuleList()
        for _ in range(len(self.mod_list)):
            backbone.append(copy.deepcopy(self.backbone))
            if self.with_neck: neck.append(copy.deepcopy(self.neck))
        self.backbone = backbone
        if self.with_neck: self.neck = neck

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = nn.ModuleList(
            SinePositionalEncoding(**self.positional_encoding) for _ in (self.mod_list))
        self.encoder = DinoTransformerEncoder_Parallel(**self.encoder)
        # self.decoder = DinoTransformerDecoder_Parallel(**self.decoder)
        self.decoder = DinoTransformerDecoder_MultiModal(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding[0].num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))

        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

    def pre_transformer(
            self,
            idx: int,
            mlvl_feats: Tuple[Tensor],
            batch_data_samples: OptSampleList = None) -> Tuple[Dict]:
        batch_size = mlvl_feats[0].size(0)

        # construct binary masks for the transformer.
        assert batch_data_samples is not None
        batch_input_shape = batch_data_samples[0].batch_input_shape
        
        input_img_h, input_img_w = batch_input_shape
        img_shape_list = [sample.img_shape for sample in batch_data_samples]
        same_shape_flag = all([
            s[0] == input_img_h and s[1] == input_img_w for s in img_shape_list
        ])
        # support torch2onnx without feeding masks
        if torch.onnx.is_in_onnx_export() or same_shape_flag:
            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(None)
                mlvl_pos_embeds.append(self.positional_encoding[idx](None, input=feat))
        else:
            masks = mlvl_feats[0].new_ones(
                (batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0
            # NOTE following the official DETR repo, non-zero
            # values representing ignored positions, while
            # zero values means valid positions.

            mlvl_masks = []
            mlvl_pos_embeds = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:]).to(
                        torch.bool).squeeze(0))
                mlvl_pos_embeds.append(self.positional_encoding[idx](mlvl_masks[-1]))

        feat_flatten = []
        lvl_pos_embed_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            pos_embed = pos_embed.view(batch_size, c, -1).permute(0, 2, 1)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            # lvl_pos_embed = pos_embed + self.level_embed[idx][lvl].view(1, 1, -1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        # (bs, num_feat_points), where num_feat_points = sum_lvl(h_lvl*w_lvl)
        if mask_flatten[0] is not None:
            mask_flatten = torch.cat(mask_flatten, 1)
        else:
            mask_flatten = None

        # (num_level, 2)
        spatial_shapes = torch.cat(spatial_shapes).view(-1, 2)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1, )),  # (num_level)
            spatial_shapes.prod(1).cumsum(0)[:-1]))
        if mlvl_masks[0] is not None:
            valid_ratios = torch.stack(  # (bs, num_level, 2)
                [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        else:
            valid_ratios = mlvl_feats[0].new_ones(batch_size, len(mlvl_feats), 2)

        encoder_inputs_dict = dict(
            feat=feat_flatten,
            feat_mask=mask_flatten,
            feat_pos=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        decoder_inputs_dict = dict(
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return encoder_inputs_dict, decoder_inputs_dict

    def forward_encoder(self, img : Dict, ir_img: Dict) -> Dict:
        
        memory = self.encoder(
            # rgb
            query=img['feat'],
            query_pos=img['feat_pos'],
            # ir
            ir_query=ir_img['feat'],
            ir_query_pos=ir_img['feat_pos'],
            # all
            key_padding_mask=img['feat_mask'],  # for self_attn
            spatial_shapes=img['spatial_shapes'],
            level_start_index=img['level_start_index'],
            valid_ratios=img['valid_ratios']) 

        encoder_outputs_dict = dict()
        encoder_outputs_dict['rgb'] = dict(
            memory=memory['rgb'],
            memory_mask=img['feat_mask'],
            spatial_shapes=img['spatial_shapes'])
        if 'ir' in self.extra_return:
            encoder_outputs_dict['ir'] = dict(
                memory=memory['ir'])
        return encoder_outputs_dict
    
    def pre_decoder(
        self,
        # rgb
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        # ir
        ir_memory: Tensor,
        ir_extra_idx: int,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, n, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        ir_output_memory, ir_output_proposals = self.gen_encoder_output_proposals(
            ir_memory, memory_mask, spatial_shapes)        

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory)
        ir_enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers + ir_extra_idx](ir_output_memory)        

        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals
        ir_enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers + ir_extra_idx](ir_output_memory) + ir_output_proposals
        
        ## diff
        # gaussin nll loss
        enc_outputs_var = torch.exp(
            self.bbox_head.var_branches[0](output_memory))
        ir_enc_outputs_var = torch.exp(
            self.bbox_head.var_branches[ir_extra_idx](ir_output_memory))               

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        # topk_indices = torch.topk(
        #     enc_outputs_class.max(-1)[0], k=self.num_queries // (1 + len(self.extra_return)), dim=1)[1]
        # ir_topk_indices = torch.topk(
        #     ir_enc_outputs_class.max(-1)[0], k=self.num_queries // (1 + len(self.extra_return)), dim=1)[1]
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        ir_topk_indices = torch.topk(
            ir_enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        # score logits
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        ir_topk_score = torch.gather(
            ir_enc_outputs_class, 1,
            ir_topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))        
        # box coordinates
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        ir_topk_coords_unact = torch.gather(
            ir_enc_outputs_coord_unact, 1,
            ir_topk_indices.unsqueeze(-1).repeat(1, 1, 4))        
        # diff
        # box variances
        topk_var = torch.gather(
            enc_outputs_var, 1, topk_indices.unsqueeze(-1))
        ir_topk_var = torch.gather(
            ir_enc_outputs_var, 1, ir_topk_indices.unsqueeze(-1))
        
        topk_coords = topk_coords_unact.sigmoid()
        ir_topk_coords = ir_topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()
        ir_topk_coords_unact = ir_topk_coords_unact.detach()

        ## 组合
        # topk_coords = torch.cat((topk_coords, ir_topk_coords), dim=1)
        # 1. simple concat
        # topk_coords_unact = torch.cat((topk_coords_unact, ir_topk_coords_unact), dim=1)
        # 2. twice topk
        # topk_score_unact = topk_score.detach()  # 可以不detach做个测试
        # ir_topk_score_uncat = ir_topk_score.detach()
        # topk_score_cat = torch.cat((topk_score_unact, ir_topk_score_uncat), dim=1)
        # topk_indices_cat = torch.topk(topk_score_cat.max(-1)[0], k=self.num_queries, dim=1)[1]
        # topk_coords_unact = torch.gather(
        #     topk_coords_unact, 1, 
        #     topk_indices_cat.unsqueeze(-1).repeat(1, 1, 4))
        # 3. probability fusion
        topk_score_unact = topk_score.detach()  # 可以不detach做个测试
        ir_topk_score_uncat = ir_topk_score.detach()
        topk_var_unact = topk_var.detach()  # 可以不detach做个测试
        ir_topk_var_unact = ir_topk_var.detach()
        topk_coords_unact = self.prob_fusion(
            scores_logits=topk_score_unact, coords=topk_coords_unact, variances=topk_var_unact,
            ir_scores_logits=ir_topk_score_uncat, ir_coords=ir_topk_coords_unact, ir_variances=ir_topk_var_unact,
            batch_data_samples=batch_data_samples)
        
        # 下面都和dino里面的操作一样
        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = \
                self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact],
                                         dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            ir_memory=ir_memory,
            reference_points=reference_points,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            enc_outputs_var=topk_var,
            extra_enc_outputs=[dict(enc_outputs_class=ir_topk_score, 
                                    enc_outputs_coord=ir_topk_coords, 
                                    enc_outputs_var=ir_topk_var)],
            dn_meta=dn_meta) if self.training else dict()
        return decoder_inputs_dict, head_inputs_dict
    
    def predict_single_image(self,
                             cls_score: Tensor, bbox_pred: Tensor, coord: Tensor, variance: Tensor,
                             img_meta: dict, rescale: bool = True, max_per_img = None):
        '''
        same as _predict_by_feat_single in detr_head
        '''
        
        assert len(cls_score) == len(bbox_pred)  # num_queries
        if max_per_img is None: max_per_img = cls_score.shape[0]   # 900
        img_shape = img_meta['img_shape']
        # exclude background
        if self.bbox_head.loss_cls.use_sigmoid:
            num_classes = cls_score.shape[-1]
            prob_score = cls_score.sigmoid()    # N, C
            scores, indexes = prob_score.view(-1).topk(max_per_img)  # (N,) (N,)
            det_labels = indexes % num_classes  # N,
            bbox_index = indexes // num_classes # N,
            bbox_pred = bbox_pred[bbox_index]
        else:
            prob_score = F.softmax(cls_score, dim=-1)[..., :-1]
            scores, det_labels = prob_score.max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        det_bboxes = bbox_cxcywh_to_xyxy(bbox_pred)
        det_bboxes[:, 0::2] = det_bboxes[:, 0::2] * img_shape[1]
        det_bboxes[:, 1::2] = det_bboxes[:, 1::2] * img_shape[0]
        det_bboxes[:, 0::2].clamp_(min=0, max=img_shape[1])
        det_bboxes[:, 1::2].clamp_(min=0, max=img_shape[0])
        if rescale:
            assert img_meta.get('scale_factor') is not None
            det_bboxes /= det_bboxes.new_tensor(
                img_meta['scale_factor']).repeat((1, 2))

        results = dict()
        results['coords'] = coord[bbox_index]  # N, 4
        results['bboxes'] = det_bboxes  # N, 4
        results['scores'] = scores  # N,
        results['class'] = det_labels  # N,
        results['cls_logits'] = cls_score[bbox_index]  # N, C
        results['probs'] = prob_score[bbox_index]   # N, C
        results['vars'] = variance[bbox_index] # N, 1
        return results

    def batched_nms(self, info_1, info_2, iou_thresh=0.5, top_k=900):
        # Raw Coords
        coords = torch.cat((info_1['coords'], info_2['coords']), dim=0)
        # Boxes
        boxes = torch.cat((info_1['bboxes'], info_2['bboxes']), dim=0)
        # Scores
        scores = torch.cat((info_1['scores'], info_2['scores']), dim=0)
        # Classes
        classes = torch.cat((info_1['class'], info_2['class']), dim=0)

        # 按得分降序排序
        order = torch.argsort(scores, descending=True)
        coords = coords[order]
        boxes = boxes[order]
        scores = scores[order]
        classes = classes[order]

        max_coordinate = boxes.max()
        offsets = classes.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

        # 进行NMS
        selected_coords = []
        while len(boxes_for_nms) > 0:
            current_box = boxes_for_nms[[0]]  # 选择得分最高的框
            selected_coords.append(coords[0])

            # 计算当前框与其他框的IOU
            ious = torchvision.ops.box_iou(current_box, boxes_for_nms[1:])
            
            # 筛选IOU小于阈值的框
            keep_mask = ious.squeeze() >= iou_thresh

            # 更新排序框集
            boxes_for_nms = boxes_for_nms[1:][keep_mask]
            coords = coords[1:][keep_mask]

            # 如果选出的框数量已经达到了top_k，停止
            if len(selected_coords) >= top_k:
                break

        # 确保返回的框数量达到top_k
        selected_coords = torch.stack(selected_coords, dim=0)
        if len(selected_coords) < top_k:
            repeating = top_k // len(selected_coords)
            padding = top_k % len(selected_coords)
            if repeating > 0:
                selected_coords = torch.cat([selected_coords for _ in range(repeating)], dim=0)                
            selected_coords = torch.cat([selected_coords, selected_coords[:padding]], dim=0)

        return selected_coords

    def modal_aware_nms(self, info_1, info_2, iou_thresh=0.5, top_k=900):
        # Raw Coords
        coords = torch.cat((info_1['coords'], info_2['coords']), dim=0)
        # Boxes
        boxes = torch.cat((info_1['bboxes'], info_2['bboxes']), dim=0)
        # Scores
        scores = torch.cat((info_1['scores'], info_2['scores']), dim=0)
        # Classes
        classes = torch.cat((info_1['class'], info_2['class']), dim=0)
        set_ids = torch.cat([
            torch.zeros(len(info_1['scores']), dtype=torch.long),
            torch.ones(len(info_2['scores']), dtype=torch.long)]).to(coords.device)

        # 按得分降序排序
        order = torch.argsort(scores, descending=True)
        coords = coords[order]
        boxes = boxes[order]
        scores = scores[order]
        classes = classes[order]
        set_ids = set_ids[order]

        max_coordinate = boxes.max()
        offsets = classes.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

        # 进行NMS
        selected_coords = []
        while len(boxes_for_nms) > 0:
            current_box = boxes_for_nms[[0]]  # 选择得分最高的框
            set_id = set_ids[0] # 确定属于那个set
            selected_coords.append(coords[0])

            # 计算当前框与其他框的IOU
            ious = torchvision.ops.box_iou(current_box, boxes_for_nms[1:])
            
            # 筛选IOU小于阈值的框
            out_mask1 = ious.squeeze() >= iou_thresh
            out_mask2 = set_ids[1:] != set_id   # 判断是不是不同模态
            keep_mask = ~(out_mask1 & out_mask2)

            # 更新排序框集
            boxes_for_nms = boxes_for_nms[1:][keep_mask]
            set_ids = set_ids[1:][keep_mask]
            coords = coords[1:][keep_mask]

            # 如果选出的框数量已经达到了top_k，停止
            if len(selected_coords) >= top_k:
                break

        # 确保返回的框数量达到top_k
        selected_coords = torch.stack(selected_coords, dim=0)
        if len(selected_coords) < top_k:
            repeating = top_k // len(selected_coords)
            padding = top_k % len(selected_coords)
            if repeating > 0:
                selected_coords = torch.cat([selected_coords for _ in range(repeating)], dim=0)                
            selected_coords = torch.cat([selected_coords, selected_coords[:padding]], dim=0)

        return selected_coords

    def bayesian_nms(self, info_1, info_2, iou_thresh=0.5, top_k=900, method='iou'):
        def xyxy2xyw2h2(bbox):
            _bbox = bbox_xyxy_to_cxcywh(bbox)
            _bbox[:,2] = _bbox[:,2] / 2
            _bbox[:,3] = _bbox[:,3] / 2
            return _bbox
        
        def sigmoid(x, k):
            return 1/(1+torch.exp(-(1/k) * x))
        
        # Raw Coords
        coords = torch.cat((info_1['coords'], info_2['coords']), dim=0)
        # Boxes
        boxes = torch.cat((info_1['bboxes'], info_2['bboxes']), dim=0)
        # Scores
        scores = torch.cat((info_1['scores'], info_2['scores']), dim=0)
        # Classes
        classes = torch.cat((info_1['class'], info_2['class']), dim=0)
        # Probs 
        # probs = torch.cat((info_1['probs'], info_2['probs']), dim=0)
        # Variances
        vars = torch.cat((info_1['vars'], info_2['vars']), dim=0)
        set_ids = torch.cat([
            torch.zeros(len(info_1['scores']), dtype=torch.long),
            torch.ones(len(info_2['scores']), dtype=torch.long)]).to(coords.device)

        # 按得分降序排序
        order = torch.argsort(scores, descending=True)
        coords = coords[order]
        boxes = boxes[order]
        scores = scores[order]
        classes = classes[order]
        set_ids = set_ids[order]

        max_coordinate = boxes.max()
        offsets = classes.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]

        # 进行NMS
        selected_coords = []
        while len(boxes_for_nms) > 0:
            current_box = boxes_for_nms[[0]]  # 选择得分最高的框
            set_id = set_ids[0] # 确定属于那个set

            # 计算当前框与其他框的IOU
            if method == 'iou':
                ious = torchvision.ops.box_iou(current_box, boxes_for_nms[1:])
            elif method == 'nwd':
                C = 32
                pred, target = xyxy2xyw2h2(current_box), xyxy2xyw2h2(boxes_for_nms[1:])
                ious = torch.exp(-torch.norm((pred - target), p=2, dim=1) / C)
            elif method == 'safit':
                C = 32
                boxes_for_nms_1 = boxes_for_nms[1:]
                pred, target = xyxy2xyw2h2(current_box), xyxy2xyw2h2(boxes_for_nms_1)
                nwd = torch.exp(-torch.norm((pred - target), p=2, dim=1) / 32)
                ious1 = torchvision.ops.box_iou(current_box, boxes_for_nms[1:])
                area = (boxes_for_nms_1[:, 2] - boxes_for_nms_1[:, 0]) * \
                    (boxes_for_nms_1[:, 3] - boxes_for_nms_1[:, 1]) # 计算面积
                ious = sigmoid(torch.sqrt(area) - C , C) * ious1 + \
                    (1 - sigmoid(torch.sqrt(area) - C, C)) * nwd
            else:
                raise ValueError(f'Unknown method: {method}')
            
            # 筛选IOU小于阈值的框
            out_mask1 = ious.squeeze() >= iou_thresh    # 判断是不是重合
            out_mask2 = set_ids[1:] != set_id   # 判断是不是不同模态
            match_mask = out_mask1 & out_mask2
            keep_mask = ~(out_mask1 & out_mask2)

            # match框集
            match_coords = torch.cat((coords[[0]], coords[1:][match_mask]), dim=0)
            match_vars = torch.cat((vars[[0]], vars[1:][match_mask]), dim=0)
            weights  = 1. / match_vars
            weighted_sum = (match_coords * weights).sum(dim=0)
            keep_coords = weighted_sum / weights.sum(dim=0)
            selected_coords.append(keep_coords)

            # 更新排序框集
            boxes_for_nms = boxes_for_nms[1:][keep_mask]
            set_ids = set_ids[1:][keep_mask]
            coords = coords[1:][keep_mask]
            vars = vars[1:][keep_mask]

            # 如果选出的框数量已经达到了top_k，停止
            if len(selected_coords) >= top_k:
                break

        # 确保返回的框数量达到top_k
        selected_coords = torch.stack(selected_coords, dim=0)
        if len(selected_coords) < top_k:
            repeating = top_k // len(selected_coords)
            padding = top_k % len(selected_coords)
            if repeating > 0:
                selected_coords = torch.cat([selected_coords for _ in range(repeating)], dim=0)                
            selected_coords = torch.cat([selected_coords, selected_coords[:padding]], dim=0)

        return selected_coords
    
    def prob_fusion(self, 
                    # rgb
                    scores_logits, coords, variances,
                    # ir
                    ir_scores_logits, ir_coords, ir_variances,
                    # all
                    batch_data_samples):
        bs = scores_logits.shape[0]
        bbox_preds = coords.sigmoid()
        ir_bbox_preds = ir_coords.sigmoid()
        out_coords = []
        for i in range(bs):
            img_meta = batch_data_samples[i].metainfo
            cls_score, bbox_pred, coord, variance = \
                scores_logits[i], bbox_preds[i], coords[i], variances[i]
            ir_cls_score, ir_bbox_pred, ir_coord, ir_variance = \
                ir_scores_logits[i], ir_bbox_preds[i], ir_coords[i], ir_variances[i]
            det = self.predict_single_image(
                cls_score, bbox_pred, coord, variance, img_meta)
            ir_det = self.predict_single_image(
                ir_cls_score, ir_bbox_pred, ir_coord, ir_variance, img_meta)
            
            # fusion
            out_coords.append(
                # 普通nms
                # self.batched_nms(det, ir_det, iou_thresh=0.5, top_k=900)
                # 模态间nms
                # self.modal_aware_nms(det, ir_det, iou_thresh=0.5, top_k=900)
                # probEn
                self.bayesian_nms(det, ir_det, iou_thresh=0.5, top_k=900, method='iou')
                # self.bayesian_nms(det, ir_det, iou_thresh=0.5, top_k=900, method='nwd')
                # self.bayesian_nms(det, ir_det, iou_thresh=0.5, top_k=900, method='safit')
            )
        return torch.stack(out_coords, dim=0)
            

    def forward_decoder(self,                        
                        query: Tensor,
                        memory: Tensor, # rgb
                        ir_memory: Tensor,  # ir
                        memory_mask: Tensor,
                        reference_points: Tensor,
                        spatial_shapes: Tensor,
                        level_start_index: Tensor,
                        valid_ratios: Tensor,
                        dn_mask: Optional[Tensor] = None,
                        **kwargs) -> Dict:
        inter_states, references = self.decoder(            
            query=query,
            value=memory,   # rgb
            ir_value=ir_memory, # ir
            key_padding_mask=memory_mask,
            self_attn_mask=dn_mask,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            reg_branches=self.bbox_head.reg_branches,
            **kwargs)

        if len(query) == self.num_queries:
            # NOTE: This is to make sure label_embeding can be involved to
            # produce loss even if there is no denoising query (no ground truth
            # target in this GPU), otherwise, this will raise runtime error in
            # distributed training.
            inter_states[0] += \
                self.dn_query_generator.label_embedding.weight[0, 0] * 0.0

        decoder_outputs_dict = dict(
            hidden_states=inter_states, references=list(references))
        return decoder_outputs_dict
    
    def forward_transformer(
        self,
        img_feats,#: list[Tuple[Tensor]],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:

        encoder_inputs_dict, decoder_inputs_dict = {}, {}
        for i, mod_name in enumerate(self.mod_list):
            encoder_inputs_dict[mod_name], decoder_inputs_dict[mod_name] = \
            self.pre_transformer(i, img_feats[i], batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)    

        decoder_inputs_dict = decoder_inputs_dict['img']       
        
        if self.extra_return is None:            
            tmp_dec_in, head_inputs_dict = super().pre_decoder(
                **encoder_outputs_dict['rgb'], batch_data_samples=batch_data_samples)
        else:
            pre_decoder_input=encoder_outputs_dict['rgb']
            for extra_idx, mod in enumerate(self.extra_return, 1):
                pre_decoder_input.update(
                    {
                        f'{mod}_memory': encoder_outputs_dict[mod]['memory'],
                        f'{mod}_extra_idx': extra_idx,
                    }
                )
            tmp_dec_in, head_inputs_dict = self.pre_decoder(
                **pre_decoder_input, batch_data_samples=batch_data_samples)        
               
        decoder_inputs_dict.update(tmp_dec_in)
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict
            
    def extract_feat(self, batch_inputs: Tensor,
                     batch_extra_inputs) -> Tuple[Tensor]:
        x_V = self.backbone[0](batch_inputs)
        if isinstance(batch_extra_inputs, list):
            batch_extra_inputs = batch_extra_inputs[0]
        assert isinstance(batch_extra_inputs, Tensor)
        x_I = self.backbone[1](batch_extra_inputs)

        if self.with_neck:
            x_V = self.neck[0](x_V)
            x_I = self.neck[1](x_I)
        # x = [x_V1 + x_I1 for x_V1, x_I1 in zip(x_V, x_I)]   # 简单的sum融合
        return x_V, x_I   # fusion, rgb, ir
        
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,             
             batch_extra_inputs: Tensor,
             batch_extra_data_samples: SampleList) -> Union[dict, list]:

        if self.use_autocast:
            with autocast(enabled=True):
                img_feats = self.extract_feat(batch_inputs, batch_extra_inputs)
        else:
            img_feats = self.extract_feat(batch_inputs, batch_extra_inputs)
        head_inputs_dict = self.forward_transformer(img_feats, batch_data_samples)

        losses = self.bbox_head.loss(
            **head_inputs_dict, batch_data_samples=batch_data_samples)
        return losses

    def predict(self, batch_inputs,
                batch_data_samples, batch_extra_inputs, rescale: bool = True):

        # image feature extraction
        img_feats = self.extract_feat(batch_inputs, batch_extra_inputs)

        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples = None, 
            batch_extra_inputs = None):        
        img_feats = self.extract_feat(batch_inputs, batch_extra_inputs)
        head_inputs_dict = self.forward_transformer(img_feats,
                                                    batch_data_samples)
        results = self.bbox_head.forward(**head_inputs_dict)
        return results

    def forward(self,
                inputs: torch.Tensor,
                data_samples=None,
                mode: str = 'tensor',
                **kwargs):
        extra_inputs = kwargs.get('extra_inputs', None)
        extra_data_samples = kwargs.get('extra_data_samples', None)

        if mode == 'tensor':
            extra_inputs = inputs.clone()

        if extra_inputs is None:
            super().forward(inputs, data_samples, mode)
        else:
            if mode == 'loss':
                return self.loss(inputs, data_samples, extra_inputs, extra_data_samples)
            elif mode == 'predict':
                return self.predict(inputs, data_samples, extra_inputs)
            elif mode == 'tensor':
                return self._forward(inputs, data_samples, extra_inputs)
            else:
                raise RuntimeError(f'Invalid mode "{mode}". '
                                'Only supports loss, predict and tensor mode')
