import copy
from typing import Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .base import BaseDetector
from .dino import DINO
from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
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

            bbox_head,
            rpn_head=None,  # two-stage rpn
            roi_head=[None],  # two-stage
            OneStage_head=[None],  # one-stage
            train_cfg=[None, None],
            test_cfg=[None, None],
            # Control whether to consider positive samples
            # from the auxiliary head as additional positive queries.
            mixed_selection=True,
            with_pos_coord=True,
            with_coord_feat=True,
            use_lsj=True,
            eval_module='detr',
            # Evaluate the Nth head.
            eval_index=0,
            num_co_heads=1,
            **kwargs):
        #################################################
        self.mod_list = mod_list
        self.mode = mode
        self.extra_return = extra_return
        #################################################

        # codetr
        self.with_pos_coord = with_pos_coord
        self.with_coord_feat = with_coord_feat
        self.num_co_heads = num_co_heads
        super(DINO, self).__init__(*args, bbox_head=bbox_head, train_cfg=bbox_head.train_cfg, test_cfg=bbox_head.test_cfg, **kwargs)
        self.mixed_selection = mixed_selection
        self.use_lsj = use_lsj

        assert eval_module in ['detr', 'one-stage', 'two-stage']
        self.eval_module = eval_module

        # Module index for evaluation
        self.eval_index = eval_index
        head_idx = 0
        if self.bbox_head is not None:
            # self.bbox_head = MODELS.build(bbox_head)
            # self.bbox_head.init_weights()
            head_idx += 1
        else:
            raise 'bbox head do not initilize'

        if rpn_head is not None:
            self.rpn_head = MODELS.build(rpn_head)
            # self.rpn_head.init_weights()

        self.roi_head = nn.ModuleList()
        for i in range(len(roi_head)):
            if roi_head[i]:
                self.roi_head.append(MODELS.build(roi_head[i]))
                # self.roi_head[-1].init_weights()

        self.OneStage_head = nn.ModuleList()
        for i in range(len(OneStage_head)):
            if OneStage_head[i]:
                self.OneStage_head.append(MODELS.build(OneStage_head[i]))
                # self.OneStage_head[-1].init_weights()

        self.head_idx = head_idx
        # self.train_cfg = train_cfg
        # self.test_cfg = test_cfg

        self.downsample = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=3, stride=2, padding=1), 
            nn.GroupNorm(32, self.embed_dims))
        
        #################################################
        # 初始化结构
        for i in range(len(self.mod_list)):
            self.add_module(f'backbone{i}', copy.deepcopy(self.backbone))
            if self.with_neck: self.add_module(f'neck{i}', copy.deepcopy(self.neck))
        del self.backbone
        if self.with_neck: del self.neck
        #################################################
    
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

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_bbox_head(self):
        """bool: whether the detector has a dino head"""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None and len(
            self.roi_head) > 0

    @property
    def with_shared_head(self):
        """bool: whether the detector has a shared head in the RoI Head"""
        return hasattr(self, 'roi_head') and self.roi_head[0].with_shared_head

    @property
    def with_bbox(self):
        """bool: whether the detector has a bbox head"""
        return ((hasattr(self, 'roi_head') and self.roi_head is not None
                 and len(self.roi_head) > 0)
                or (hasattr(self, 'OneStage_head') and self.OneStage_head is not None
                    and len(self.OneStage_head) > 0))

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
            spatial_shapes = encoder_inputs_dict['spatial_shapes']
            memory = encoder_outputs_dict['memory']
            outs = []
            bs, _, c = memory.shape
            start = 0
            for lvl in range(spatial_shapes.shape[0]):
                h, w = spatial_shapes[lvl]
                end = start + h * w
                feat = memory[:, start:end].permute(0, 2, 1).contiguous()
                start = end
                outs.append(feat.reshape(bs, c, h, w))
            outs.append(self.downsample(outs[-1]))
            return head_inputs_dict, outs
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

        decoder_inputs_dict = self.pre_transformer_aux(img_feats, batch_data_samples)     
        tmp_dec_in = self.pre_decoder_aux(pos_anchors=aux_coords, pos_feats=aux_feats, head_idx=head_idx)
        decoder_inputs_dict.update(tmp_dec_in)
        # decoder_inputs_dict.update(dict(attn_masks=attn_masks)) deformable detr
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)        
        return head_inputs_dict

    def pre_transformer_aux(self,
                            mlvl_feats: Tuple[Tensor],
                            batch_data_samples):
        '''Only for decoder'''
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
            for feat in mlvl_feats:
                mlvl_masks.append(None)
        else:
            masks = mlvl_feats[0].new_ones((batch_size, input_img_h, input_img_w))
            for img_id in range(batch_size):
                img_h, img_w = img_shape_list[img_id]
                masks[img_id, :img_h, :img_w] = 0

            mlvl_masks = []
            for feat in mlvl_feats:
                mlvl_masks.append(
                    F.interpolate(masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0))
    
        feat_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for feat, mask in zip(mlvl_feats, mlvl_masks):
            batch_size, c, h, w = feat.shape
            spatial_shape = torch._shape_as_tensor(feat)[2:].to(feat.device)
            # [bs, c, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl, c]
            feat = feat.view(batch_size, c, -1).permute(0, 2, 1)
            # [bs, h_lvl, w_lvl] -> [bs, h_lvl*w_lvl]
            if mask is not None:
                mask = mask.flatten(1)

            feat_flatten.append(feat)
            mask_flatten.append(mask)
            spatial_shapes.append(spatial_shape)

        # (bs, num_feat_points, dim)
        feat_flatten = torch.cat(feat_flatten, 1)
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

        decoder_inputs_dict = dict(
            memory=feat_flatten,
            memory_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios)
        return decoder_inputs_dict
    
    def pre_decoder_aux(self,
                        pos_anchors: Tensor,
                        pos_feats=None,
                        head_idx=0):
        topk_coords_unact = inverse_sigmoid(pos_anchors)
        reference_points = pos_anchors

        # # deformable detr
        # init_reference_out = reference_points
        # if self.num_co_heads > 0:
        #     pos_trans_out = self.aux_pos_trans_fc[head_idx](self.get_proposal_pos_embed(topk_coords_unact))
        #     pos_trans_out = self.aux_pos_trans_norm[head_idx](pos_trans_out)
        #     query_pos, query = torch.split(pos_trans_out, c, dim=2)
        #     if self.with_coord_feat:
        #         query = query + self.pos_feats_norm[head_idx](
        #             self.pos_feats_trans[head_idx](pos_feats))
        #         query_pos = query_pos + self.head_pos_embed.weight[head_idx]
        # decoder_inputs_dict = dict(
        #     query=query,
        #     query_pos=query_pos,    
        #     reference_points=reference_points)
        
        # dino
        if self.num_co_heads > 0:
            pos_trans_out = self.aux_pos_trans_fc[head_idx](self.get_proposal_pos_embed(topk_coords_unact))
            pos_trans_out = self.aux_pos_trans_norm[head_idx](pos_trans_out)
            query = pos_trans_out   # 这种query的生成方式和dino（用query embedding）不一样，和deformdetr有点像，但是也不同
            if self.with_coord_feat:
                query = query + self.pos_feats_norm[head_idx](
                    self.pos_feats_trans[head_idx](pos_feats))
        decoder_inputs_dict = dict(
            query=query, 
            reference_points=reference_points)
        return decoder_inputs_dict

    @staticmethod
    def upd_loss(losses, idx, weight=1):
        new_losses = dict()
        for k, v in losses.items():
            new_k = '{}{}'.format(k, idx)
            if isinstance(v, list) or isinstance(v, tuple):
                new_losses[new_k] = [i * weight for i in v]
            else:
                new_losses[new_k] = v * weight
        return new_losses
    
    def TwoStage_forward(self, x, batch_data_samples):
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
            # avoid get same name with roi_head loss
            keys = rpn_losses.keys()
            for key in list(keys):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            assert batch_data_samples[0].get('proposals', None) is not None
            # use pre-defined proposals in InstanceData for the second stage
            # to extract ROI features.
            proposal_list = [
                data_sample.proposals for data_sample in batch_data_samples]

        positive_coords = []
        for i in range(len(self.roi_head)):
            roi_losses = self.roi_head[i].loss(x, proposal_list, batch_data_samples)
            if self.with_pos_coord:
                positive_coords.append(roi_losses.pop('pos_coords'))    # ori_coords, ori_labels, ori_bbox_targets, ori_bbox_feats
            else:
                if 'pos_coords' in roi_losses.keys():
                    roi_losses.pop('pos_coords')
            roi_losses = self.upd_loss(roi_losses, idx=i)
            losses.update(roi_losses)
        return losses, positive_coords
    
    def OneStage_forward(self, x, batch_data_samples):
        losses = dict()
        positive_coords = []
        for i in range(len(self.OneStage_head)):
            bbox_losses = self.OneStage_head[i].loss(x, batch_data_samples)
            if self.with_pos_coord:
                pos_coords = bbox_losses.pop('pos_coords')
                positive_coords.append(pos_coords)  # ori_anchors, ori_labels, ori_bbox_targets
            else:
                if 'pos_coords' in bbox_losses.keys():
                    bbox_losses.pop('pos_coords')
            bbox_losses = self.upd_loss(bbox_losses, idx=i + len(self.roi_head))
            losses.update(bbox_losses)
        return losses, positive_coords
    
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

        x = self.extract_feat(batch_inputs)

        losses = dict()

        # DETR encoder and decoder forward
        if self.with_bbox_head:
            head_inputs_dict, x = self.forward_transformer(x, batch_data_samples)
            # bbox_losses, x = self.bbox_head.loss(x, batch_data_samples)
            # 这里的x是encoder之后的query重建成的featmap
            bbox_losses = self.bbox_head.loss(
                **head_inputs_dict, batch_data_samples=batch_data_samples)
            losses.update(bbox_losses)

        positive_coords = []
        # Two Stage rpn, roi forward
        TwoStage_losses, TwoStage_positive_coords = self.TwoStage_forward(x, batch_data_samples)
        losses.update(TwoStage_losses)
        positive_coords += TwoStage_positive_coords

        # One Stage bbox head forward
        OneStage_losses, OneStage_positive_coords = self.OneStage_forward(x, batch_data_samples)
        losses.update(OneStage_losses)
        positive_coords += OneStage_positive_coords

        if self.with_pos_coord and len(positive_coords) > 0:
            for i in range(len(positive_coords)):
                batch_img_metas = [data_sample.metainfo for data_sample in batch_data_samples]
                aux_targets = self.bbox_head.get_aux_targets(pos_coords=positive_coords[i], img_metas=batch_img_metas, mlvl_feats=x)
                head_inputs_dict = self.forward_transformer_aux(x[:-1], batch_data_samples, aux_targets, i)
                bbox_losses = self.bbox_head.loss_aux(**head_inputs_dict, batch_data_samples=batch_data_samples)
                bbox_losses = self.upd_loss(bbox_losses, idx=i)
                losses.update(bbox_losses)

        return losses

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
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

        img_feats = self.extract_feat(batch_inputs)
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
