import copy
from typing import Tuple, Union, Dict, Optional

import torch
import torchvision
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .base import BaseDetector
from .dino import DINO
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.structures.bbox import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh

from mmdet.utils import InstanceList, OptConfigType, OptMultiConfig
from ..layers.transformer import inverse_sigmoid, DinoTransformerDecoderNew
from ..layers import SinePositionalEncoding, DetrTransformerEncoder, DeformableDetrTransformerEncoder
from ..layers.transformer.dino_layers_fusion_20250513 import DinoTransformerEncoder_Parallel
from ..layers.transformer.dino_layers_fusion_20250512 import DinoTransformerDecoder_Parallel

from .codetr_new import CoDETR_New

@MODELS.register_module()
class CoDETR_parallel_20250731(CoDETR_New):

    def __init__(
            self,
            *args,
            mod_list=["img", "ir_img"],
            mode='loss',
            extra_return=["ir",],
            decoder: OptConfigType = None,
            bbox_head: OptConfigType = None,

            use_lsj=True,

            # bbox_head,
            # rpn_head=None,  # two-stage rpn
            # roi_head=[None],  # two-stage
            # OneStage_head=[None],  # one-stage
            # train_cfg=[None, None],
            # test_cfg=[None, None],
            # # Control whether to consider positive samples
            # # from the auxiliary head as additional positive queries.
            # mixed_selection=True,
            # with_pos_coord=True,
            # with_coord_feat=True,
            # use_lsj=True,
            # eval_module='detr',
            # # Evaluate the Nth head.
            # eval_index=0,
            # num_co_heads=1,
            **kwargs):
        #################################################
        self.mod_list = mod_list
        self.mode = mode
        self.extra_return = extra_return
        #################################################

        self.bbox_head_raw = copy.deepcopy(bbox_head)

        # super().__init__(*args, **kwargs)
        super().__init__(*args, 
                         bbox_head=bbox_head,
                         decoder=decoder,
                         use_lsj=use_lsj,
                         train_cfg=bbox_head.train_cfg, 
                         test_cfg=bbox_head.test_cfg, 
                         **kwargs)

        if self.bbox_head_raw is not None:
            assert 'share_pred_layer' not in self.bbox_head_raw and \
                   'num_pred_layer' not in self.bbox_head_raw and \
                   'as_two_stage' not in self.bbox_head_raw, \
                'The two keyword args `share_pred_layer`, `num_pred_layer`, ' \
                'and `as_two_stage are set in `detector.__init__()`, users ' \
                'should not set them in `bbox_head` config.'
            # The last prediction layer is used to generate proposal
            # from encode feature map when `as_two_stage` is `True`.
            # And all the prediction layers should share parameters
            # when `with_box_refine` is `True`.
            self.bbox_head_raw['share_pred_layer'] = not self.with_box_refine
            self.bbox_head_raw['num_pred_layer'] = (decoder['num_layers'] + 1) \
                if self.as_two_stage else decoder['num_layers']
            self.bbox_head_raw['as_two_stage'] = self.as_two_stage
            
            self.bbox_head_raw['num_pred_layer'] += len(extra_return)  # for supervise instance of rgb/ir image
        
        self.bbox_head = MODELS.build(self.bbox_head_raw)

        #################################################
        # 初始化结构
        for i in range(len(self.mod_list)):
            self.add_module(f'backbone{i}', copy.deepcopy(self.backbone))
            self.add_module(f'downsample{i}', copy.deepcopy(self.downsample))
            if self.with_neck: self.add_module(f'neck{i}', copy.deepcopy(self.neck))
        del self.backbone
        del self.downsample
        if self.with_neck: del self.neck

        self.ir_rpn_head = copy.deepcopy(self.rpn_head)
        self.ir_roi_head = copy.deepcopy(self.roi_head)
        self.ir_OneStage_head = copy.deepcopy(self.OneStage_head)
        #################################################

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck"""
        if hasattr(self, 'neck') and self.neck is not None:
            return True
        elif hasattr(self, 'neck0') and self.neck0 is not None:
            return True
        else:
            return False
    
    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = nn.ModuleList(
            SinePositionalEncoding(**self.positional_encoding) for _ in (self.mod_list))
        # self.encoder = DeformableDetrTransformerEncoder(**self.encoder)
        # self.decoder = DinoTransformerDecoderNew(**self.decoder)
        #################################################
        self.encoder = DinoTransformerEncoder_Parallel(**self.encoder)
        self.decoder = DinoTransformerDecoder_Parallel(**self.decoder)
        #################################################
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        # NOTE In DINO, the query_embedding only contains content
        # queries, while in Deformable DETR, the query_embedding
        # contains both content and spatial queries, and in DETR,
        # it only contains spatial queries.

        num_feats = self.positional_encoding[0].num_feats   # diff
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        if self.with_pos_coord & (self.num_co_heads > 0):
            # bug: this code should be 'self.head_pos_embed =
            # nn.Embedding(self.num_co_heads, self.embed_dims)',
            # we keep this bug for reproducing our results with ResNet-50.
            # You can fix this bug when reproducing results with
            # swin transformer.
            # self.head_pos_embed = nn.Embedding(self.num_co_heads, 1, 1, self.embed_dims)  # deformable detr
            self.aux_pos_trans_fc = nn.ModuleList()
            self.aux_pos_trans_norm = nn.ModuleList()
            self.pos_feats_trans = nn.ModuleList()
            self.pos_feats_norm = nn.ModuleList()
            for _ in range(self.num_co_heads):
                self.aux_pos_trans_fc.append(nn.Linear(self.embed_dims * 2, self.embed_dims))
                self.aux_pos_trans_norm.append(nn.LayerNorm(self.embed_dims))
                if self.with_coord_feat:
                    self.pos_feats_trans.append(nn.Linear(self.embed_dims, self.embed_dims))
                    self.pos_feats_norm.append(nn.LayerNorm(self.embed_dims))

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
        # enc_outputs_var = torch.exp(
        #     self.bbox_head.var_branches[0](output_memory))
        # ir_enc_outputs_var = torch.exp(
        #     self.bbox_head.var_branches[ir_extra_idx](ir_output_memory))               
        enc_outputs_var = self.bbox_head.var_branches[0](output_memory)
        ir_enc_outputs_var = self.bbox_head.var_branches[ir_extra_idx](ir_output_memory)            

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
        #     torch.cat((topk_coords_unact, ir_topk_coords_unact), dim=1), 1, 
        #     topk_indices_cat.unsqueeze(-1).repeat(1, 1, 4))
        # 3. probability fusion
        topk_score_unact = topk_score#.detach()  # 可以不detach做个测试
        ir_topk_score_uncat = ir_topk_score#.detach()
        topk_var_unact = topk_var#.detach()  # 可以不detach做个测试
        ir_topk_var_unact = ir_topk_var#.detach()
        topk_coords_unact = self.prob_fusion(
            scores_logits=topk_score_unact, coords=topk_coords_unact, variances=topk_var_unact.exp(),
            ir_scores_logits=ir_topk_score_uncat, ir_coords=ir_topk_coords_unact, ir_variances=ir_topk_var_unact.exp(),
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

        if method == 'iou':
            ious = torchvision.ops.box_iou(boxes_for_nms, boxes_for_nms)
        elif method == 'nwd':
            C = 32
            target = xyxy2xyw2h2(boxes_for_nms)
            ious = torch.exp(
                -torch.norm((target[:, None, :] - target[None, :, :]), p=2, dim=2) 
                / C)
        elif method == 'safit':
            C = 32
            target = xyxy2xyw2h2(boxes_for_nms)
            nwd = torch.exp(
                -torch.norm((target[:, None, :] - target[None, :, :]), p=2, dim=2) 
                / C)
            ious1 = torchvision.ops.box_iou(boxes_for_nms, boxes_for_nms)
            area = 4 * target[:, [2]] * target[:, [3]]  # 计算面积
            ious = sigmoid(torch.sqrt(area) - C , C) * ious1 + \
                (1 - sigmoid(torch.sqrt(area) - C, C)) * nwd
        else:
            raise ValueError(f'Unknown method: {method}') 
    
        # new
        coords = torch.nan_to_num(coords, nan=0.0, posinf=1e5, neginf=-1e5)

        iou_mask = ious >= iou_thresh   # N, N
        set_id_mask = set_ids.unsqueeze(1) != set_ids.unsqueeze(0)  # N, N
        match_mask = iou_mask & set_id_mask # N, N
        self_mask = torch.eye(len(coords), len(coords), dtype=torch.bool, device=coords.device)   # N, N
        total_mask = (match_mask | self_mask).float() # N, N
        # diff clamp
        vars_for_calc = vars.clone() if vars.size(-1) != 2 else torch.repeat_interleave(vars, 2, dim=-1)
        with torch.no_grad():
            vars_for_calc.clamp_(min=1e-4)
        weights = 1.0 / vars_for_calc     # N, 1
        weighted_coords = coords * weights  # N, 4
        # 4. 计算加权和的分子与分母
        numerator = torch.matmul(total_mask, weighted_coords)
        denominator = torch.matmul(total_mask, weights)
        safe_denominator = torch.clamp(denominator, min=1e-6)
        # f. 计算最终的加权平均坐标
        selected_coords = numerator / safe_denominator
        return selected_coords[:top_k].detach()
     
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
                # self.bayesian_nms_with_score(det, ir_det, iou_thresh=0.5, top_k=900, method='iou')
                # self.bayesian_nms_with_score(det, ir_det, iou_thresh=0.5, top_k=900, method='nwd')
                # self.bayesian_nms_with_score(det, ir_det, iou_thresh=0.5, top_k=900, method='safit')
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
    
    def forward_transformer(self,
                            img_feats: Tuple[Tensor],
                            batch_data_samples: OptSampleList = None):
        
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

        if self.training:
            spatial_shapes = encoder_inputs_dict['img']['spatial_shapes']
            memory = encoder_outputs_dict['rgb']['memory']
            ir_memory = encoder_outputs_dict['ir']['memory']
            outs = []
            ir_outs = []
            bs, _, c = memory.shape
            start = 0
            for lvl in range(spatial_shapes.shape[0]):
                h, w = spatial_shapes[lvl]
                end = start + h * w
                feat = memory[:, start:end].permute(0, 2, 1).contiguous()
                ir_feat = ir_memory[:, start:end].permute(0, 2, 1).contiguous()
                start = end
                outs.append(feat.reshape(bs, c, h, w))
                ir_outs.append(ir_feat.reshape(bs, c, h, w))
            downsample = getattr(self, f'downsample{0}')
            outs.append(downsample(outs[-1]))
            downsample = getattr(self, f'downsample{1}')
            ir_outs.append(downsample(ir_outs[-1]))
            return head_inputs_dict, outs, ir_outs
        else:
            return head_inputs_dict
    
    def forward_transformer_aux(self,
                                img_feats: Tuple[Tensor],
                                batch_data_samples,
                                aux_targets,
                                head_idx):
        aux_coords, aux_labels, aux_targets, aux_label_weights, \
                aux_bbox_weights, aux_feats, attn_masks = aux_targets
        head_inputs_dict=dict(
            aux_labels=aux_labels,
            aux_targets=aux_targets,
            aux_label_weights=aux_label_weights,
            aux_bbox_weights=aux_bbox_weights)

        decoder_inputs_dict = {}
        for i, mod_name in enumerate(self.mod_list):
            decoder_inputs_dict[mod_name] = self.pre_transformer_aux(img_feats[i], batch_data_samples)

        tmp_dec_in = self.pre_decoder_aux(pos_anchors=aux_coords, pos_feats=aux_feats, head_idx=head_idx)
        # decoder_inputs_dict.update(tmp_dec_in)
        # decoder_inputs_dict.update(dict(attn_masks=attn_masks)) deformable detr
        decoder_outputs_dict = self.forward_decoder(**tmp_dec_in, **decoder_inputs_dict['img'], ir_memory=decoder_inputs_dict['ir_img']['memory'])
        head_inputs_dict.update(decoder_outputs_dict)        
        return head_inputs_dict

    @staticmethod
    def upd_loss(losses, idx, weight=1, mod=''):
        new_losses = dict()
        for k, v in losses.items():
            new_k = '{}_{}{}'.format(mod, k, idx)
            if isinstance(v, list) or isinstance(v, tuple):
                new_losses[new_k] = [i * weight for i in v]
            else:
                new_losses[new_k] = v * weight
        return new_losses    
    
    def TwoStage_forward(self, x, ir_x, batch_data_samples):
        losses = dict()
        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.rpn_head.train_cfg.get(
                'rpn_proposal', self.rpn_head.test_cfg.get('rpn'))

            rpn_data_samples = copy.deepcopy(batch_data_samples)
            # set cat_id of gt_labels to 0 in RPN
            for data_sample in rpn_data_samples:
                data_sample.gt_instances.labels = \
                    torch.zeros_like(data_sample.gt_instances.labels)

            rpn_losses, proposal_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)
            ir_rpn_losses, ir_proposal_list = self.ir_rpn_head.loss_and_predict(
                ir_x, rpn_data_samples, proposal_cfg=proposal_cfg)
            
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
            keys = ir_rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    ir_rpn_losses[f'rpn_{key}'] = ir_rpn_losses.pop(key)
            keys = ir_rpn_losses.keys()
            for key in list(keys):
                ir_rpn_losses[f'ir_{key}'] = ir_rpn_losses.pop(key)
            losses.update(ir_rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            proposal_list = [
                data_sample.proposals for data_sample in batch_data_samples]
            ir_proposal_list = [
                data_sample.proposals for data_sample in batch_data_samples]

        positive_coords, ir_positive_coords = [], []
        for i in range(len(self.roi_head)):
            roi_losses = self.roi_head[i].loss(x, proposal_list, batch_data_samples)
            ir_roi_losses = self.ir_roi_head[i].loss(ir_x, ir_proposal_list, batch_data_samples)
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))    # ori_coords, ori_labels, ori_bbox_targets, ori_bbox_feats
                ir_positive_coords.append(ir_roi_losses.pop('pos_coords'))
            else:
                if 'pos_coords' in roi_losses.keys():
                    roi_losses.pop('pos_coords')
                if 'pos_coords' in ir_roi_losses.keys():
                    ir_roi_losses.pop('pos_coords')
            roi_losses = self.upd_loss(roi_losses, idx=i)
            ir_roi_losses = self.upd_loss(ir_roi_losses, idx=i, mod='ir')
            losses.update(roi_losses)
            losses.update(ir_roi_losses)
        return losses, positive_coords, ir_positive_coords
    
    def OneStage_forward(self, x, ir_x, batch_data_samples):
        losses = dict()
        positive_coords, ir_positive_coords = [], []
        for i in range(len(self.OneStage_head)):
            bbox_losses = self.OneStage_head[i].loss(x, batch_data_samples)
            ir_bbox_losses = self.ir_OneStage_head[i].loss(ir_x, batch_data_samples)
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                ir_pos_coords = ir_bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)  # ori_anchors, ori_labels, ori_bbox_targets
                ir_positive_coords.append(ir_pos_coords)
            else:
                if 'pos_coords' in bbox_losses.keys():
                    bbox_losses.pop('pos_coords')
                if 'pos_coords' in ir_bbox_losses.keys():
                    ir_bbox_losses.pop('pos_coords')
            bbox_losses = self.upd_loss(bbox_losses, idx=i + len(self.roi_head))
            ir_bbox_losses = self.upd_loss(ir_bbox_losses, idx=i + len(self.roi_head), mod='ir')
            losses.update(bbox_losses)
            losses.update(ir_bbox_losses)
        return losses, positive_coords, ir_positive_coords
    
    def extract_feat(self, batch_inputs: Tensor,
                     batch_extra_inputs) -> Tuple[Tensor]:
        backbone = getattr(self, f'backbone{0}')
        x_V = backbone(batch_inputs)
        if isinstance(batch_extra_inputs, list):
            batch_extra_inputs = batch_extra_inputs[0]
        assert isinstance(batch_extra_inputs, Tensor)        
        backbone = getattr(self, f'backbone{1}')
        x_I = backbone(batch_extra_inputs)

        if self.with_neck:
            neck = getattr(self, f'neck{0}')
            x_V = neck(x_V)
            neck = getattr(self, f'neck{1}')
            x_I = neck(x_I)
        # x = [x_V1 + x_I1 for x_V1, x_I1 in zip(x_V, x_I)]   # 简单的sum融合
        return x_V, x_I   # fusion, rgb, ir
    
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList,
             batch_extra_inputs: Tensor,
             batch_extra_data_samples: SampleList) -> Union[dict, list]:
        batch_input_shape = batch_data_samples[0].batch_input_shape
        if self.use_lsj:
            for data_samples in batch_data_samples:
                img_metas = data_samples.metainfo
                input_img_h, input_img_w = batch_input_shape
                img_metas['img_shape'] = [input_img_h, input_img_w]

        x, ir_x = self.extract_feat(batch_inputs, batch_extra_inputs)

        losses = dict()

        # DETR encoder and decoder forward
        if self.with_bbox_head:
            head_inputs_dict, x, ir_x = self.forward_transformer([x, ir_x], batch_data_samples)
            # bbox_losses, x = self.bbox_head.loss(x, batch_data_samples)
            # 这里的x是encoder之后的query重建成的featmap
            bbox_losses = self.bbox_head.loss(
                **head_inputs_dict, batch_data_samples=batch_data_samples)
            losses.update(bbox_losses)

        positive_coords, ir_positive_coords = [], []
        # Two Stage rpn, roi forward
        TwoStage_losses, TwoStage_positive_coords, TwoStage_ir_positive_coords = self.TwoStage_forward(x, ir_x, batch_data_samples)
        losses.update(TwoStage_losses)
        positive_coords += TwoStage_positive_coords
        ir_positive_coords += TwoStage_ir_positive_coords

        # One Stage bbox head forward
        OneStage_losses, OneStage_positive_coords, OneStage_ir_positive_coords = self.OneStage_forward(x, ir_x, batch_data_samples)
        losses.update(OneStage_losses)
        positive_coords += OneStage_positive_coords
        ir_positive_coords += OneStage_ir_positive_coords

        if self.with_pos_coord and len(positive_coords) > 0:
            for i in range(len(positive_coords)):
                batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
                aux_targets = self.bbox_head.get_aux_targets(pos_coords=positive_coords[i], img_metas=batch_img_metas, mlvl_feats=x)
                ir_aux_targets = self.bbox_head.get_aux_targets(pos_coords=ir_positive_coords[i], img_metas=batch_img_metas, mlvl_feats=ir_x)
                #########################################################################
                aux_coords, aux_labels, au_targets, aux_label_weights, aux_bbox_weights, aux_feats, attn_masks = aux_targets 
                ir_aux_coords, ir_aux_labels, ir_au_targets, ir_aux_label_weights, ir_aux_bbox_weights, ir_aux_feats, ir_attn_masks = ir_aux_targets
                aux_coords = torch.cat((aux_coords, ir_aux_coords), dim=1)
                aux_labels = torch.cat((aux_labels, ir_aux_labels), dim=1)
                au_targets = torch.cat((au_targets, ir_au_targets), dim=1)
                aux_label_weights = torch.cat((aux_label_weights, ir_aux_label_weights), dim=1)
                aux_bbox_weights = torch.cat((aux_bbox_weights, ir_aux_bbox_weights), dim=1)
                aux_feats = torch.cat((aux_feats, ir_aux_feats), dim=1)
                if attn_masks is not None:
                    attn_masks = torch.cat((attn_masks, ir_attn_masks), dim=1)
                aux_targets = (aux_coords, aux_labels, au_targets, aux_label_weights, aux_bbox_weights, aux_feats, attn_masks)
                #########################################################################
                head_inputs_dict = self.forward_transformer_aux((x[:-1], ir_x[:-1]), batch_data_samples, aux_targets, i)
                bbox_losses = self.bbox_head.loss_aux(**head_inputs_dict, batch_data_samples=batch_data_samples)
                bbox_losses = self.upd_loss(bbox_losses, idx=i)
                losses.update(bbox_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                batch_extra_inputs: Tensor,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs, has shape (bs, dim, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the input images.
            Each DetDataSample usually contain 'pred_instances'. And the
            `pred_instances` usually contains following keys.

            - scores (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        assert self.eval_module in ['detr', 'one-stage', 'two-stage']

        if self.use_lsj:
            for data_samples in batch_data_samples:
                img_metas = data_samples.metainfo
                input_img_h, input_img_w = img_metas['batch_input_shape']
                img_metas['img_shape'] = [input_img_h, input_img_w]

        img_feats = self.extract_feat(batch_inputs, batch_extra_inputs)
        if self.with_bbox and self.eval_module == 'one-stage':
            results_list = self.predict_OneStage_head(
                img_feats, batch_data_samples, rescale=rescale)
        elif self.with_roi_head and self.eval_module == 'two-stage':
            results_list = self.predict_TwoStage_head(
                img_feats, batch_data_samples, rescale=rescale)
        else:
            results_list = self.predict_bbox_head(
                img_feats, batch_data_samples, rescale=rescale)

        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def predict_bbox_head(self,
                           mlvl_feats: Tuple[Tensor],
                           batch_data_samples: SampleList,
                           rescale: bool = True) -> InstanceList:        
        head_inputs_dict = self.forward_transformer(mlvl_feats,
                                                    batch_data_samples)
        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)
        return results_list

    def predict_TwoStage_head(self,
                         mlvl_feats: Tuple[Tensor],
                         batch_data_samples: SampleList,
                         rescale: bool = True) -> InstanceList:
        assert self.with_bbox, 'Bbox head must be implemented.'
        if self.with_bbox_head:
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]
            results = self.forward_transformer(mlvl_feats, batch_img_metas)
            mlvl_feats = results[-1]
        rpn_results_list = self.rpn_head.predict(
            mlvl_feats, batch_data_samples, rescale=False)
        return self.roi_head[self.eval_index].predict(
            mlvl_feats, rpn_results_list, batch_data_samples, rescale=rescale)

    def predict_OneStage_head(self,
                          mlvl_feats: Tuple[Tensor],
                          batch_data_samples: SampleList,
                          rescale: bool = True) -> InstanceList:
        assert self.with_bbox, 'Bbox head must be implemented.'
        if self.with_bbox_head:
            batch_img_metas = [
                data_samples.metainfo for data_samples in batch_data_samples
            ]
            results = self.forward_transformer(mlvl_feats, batch_img_metas)
            mlvl_feats = results[-1]
        return self.OneStage_head[self.eval_index].predict(
            mlvl_feats, batch_data_samples, rescale=rescale)

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
