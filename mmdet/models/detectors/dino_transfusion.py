# Copyright (c) OpenMMLab. All rights reserved.
import copy
import re
import warnings
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from mmengine.runner.amp import autocast
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType
from ..layers import SinePositionalEncoding, CdnQueryGenerator, DinoTransformerDecoder
from ..layers.transformer.dino_layers_fusion import (DinoTransformerEncoder_FusionNew,
    DinoTransformerEncoder_Fusion, DinoTransformerDecoder_New)
from .dino import DINO
from .deformable_detr import DeformableDETR


@MODELS.register_module()
class DINO_TransFusion(DINO):
    def __init__(self,
                 *args,
                 mod_list=["img", "ir_img"],
                #  mod_loss_list=[],
                 mod_loss_list=["img", "ir_img"],
                 use_autocast=False,                 
                 mode='loss',
                 extra_return=None,
                 decoder: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 dn_cfg: OptConfigType = None,
                 with_box_refine: bool = False,
                 as_two_stage: bool = False,
                 num_feature_levels: int = 4,
                 extra_num_queries: int = 0,
                 **kwargs) -> None:

        self.mod_list = mod_list        
        self.num_mod = len(mod_list)
        # self.num_mod = range(len(mod_list))
        self.mod_loss_list = mod_loss_list
        self.extra_num_queries = extra_num_queries

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
            SinePositionalEncoding(**self.positional_encoding) for _ in (['fusion'] + self.mod_list))
        self.encoder = DinoTransformerEncoder_FusionNew(**self.encoder)
        # self.encoder = DinoTransformerEncoder_Fusion(**self.encoder)
        self.decoder = DinoTransformerDecoder_New(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding[0].num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))

        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # if 'img' in self.mod_loss_list:
        #     self.rgb_query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        #     nn.init.xavier_uniform_(self.rgb_query_embedding.weight)
        # if 'ir_img' in self.mod_loss_list:
        #     self.ir_query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        #     nn.init.xavier_uniform_(self.ir_query_embedding.weight)

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

    def forward_encoder(self, fusion : Dict, img : Dict, ir_img: Dict) -> Dict:
        
        memory = self.encoder(
            # fusion
            query=fusion['feat'],
            query_pos=fusion['feat_pos'],
            key_padding_mask=fusion['feat_mask'],  # for self_attn
            spatial_shapes=fusion['spatial_shapes'],
            level_start_index=fusion['level_start_index'],
            valid_ratios=fusion['valid_ratios'],
            # rgb
            rgb_query=img['feat'],
            rgb_query_pos=img['feat_pos'],
            rgb_key_padding_mask=img['feat_mask'],  # for self_attn & rgb2fusion
            rgb_spatial_shapes=img['spatial_shapes'],
            rgb_level_start_index=img['level_start_index'],
            rgb_valid_ratios=img['valid_ratios'],
            # ir
            ir_query=ir_img['feat'],
            ir_query_pos=ir_img['feat_pos'],
            ir_key_padding_mask=ir_img['feat_mask'],  # for self_attn & rgb2fusion
            ir_spatial_shapes=ir_img['spatial_shapes'],
            ir_level_start_index=ir_img['level_start_index'],
            ir_valid_ratios=ir_img['valid_ratios']) 

        encoder_outputs_dict = dict()
        encoder_outputs_dict['fusion'] = dict(
            memory=memory['fusion'],
            memory_mask=fusion['feat_mask'],
            spatial_shapes=fusion['spatial_shapes'])
        if 'rgb' in self.extra_return:
            encoder_outputs_dict['rgb'] = dict(
                memory=memory['rgb'],
                memory_mask=img['feat_mask'],
                spatial_shapes=img['spatial_shapes'])
        if 'ir' in self.extra_return:
            encoder_outputs_dict['ir'] = dict(
                memory=memory['ir'],
                memory_mask=ir_img['feat_mask'],
                spatial_shapes=ir_img['spatial_shapes'])
        return encoder_outputs_dict
    
    def pre_decoder_new(
        self,
        # fusion
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        # rgb
        rgb_memory: Tensor,
        rgb_memory_mask: Tensor,
        rgb_spatial_shapes: Tensor,
        rgb_extra_idx: int,
        # ir
        ir_memory: Tensor,
        ir_memory_mask: Tensor,
        ir_spatial_shapes: Tensor,
        ir_extra_idx: int,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, n, c = memory.shape
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        rgb_output_memory, rgb_output_proposals = self.gen_encoder_output_proposals(
            rgb_memory, rgb_memory_mask, rgb_spatial_shapes)
        ir_output_memory, ir_output_proposals = self.gen_encoder_output_proposals(
            ir_memory, ir_memory_mask, ir_spatial_shapes)        

        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers](output_memory)
        rgb_enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers + rgb_extra_idx](rgb_output_memory)
        ir_enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers + ir_extra_idx](ir_output_memory)        

        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers](output_memory) + output_proposals
        rgb_enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers + rgb_extra_idx](rgb_output_memory) + rgb_output_proposals
        ir_enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers + ir_extra_idx](ir_output_memory) + ir_output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries - self.extra_num_queries * self.num_mod, dim=1)[1]
        rgb_topk_indices = torch.topk(
            rgb_enc_outputs_class.max(-1)[0], k=self.extra_num_queries, dim=1)[1]
        ir_topk_indices = torch.topk(
            ir_enc_outputs_class.max(-1)[0], k=self.extra_num_queries, dim=1)[1]
        
        # topk_score = torch.gather(
        #     enc_outputs_class, 1,
        #     topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        rgb_topk_score = torch.gather(
            rgb_enc_outputs_class, 1,
            rgb_topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        ir_topk_score = torch.gather(
            ir_enc_outputs_class, 1,
            ir_topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        
        # get remaining fused feats
        full_indices = torch.arange(n, device=memory.device).unsqueeze(0).repeat(bs, 1)
        remaining_indices = []
        for i in range(bs):
            mask = ~torch.isin(full_indices[i], topk_indices[i])
            remaining_indices.append(full_indices[i][mask])
        remaining_indices = torch.stack(remaining_indices, dim=0)
        remaining_feats = \
            torch.gather(output_memory, 1, 
                         remaining_indices.unsqueeze(-1).repeat(1, 1, cls_out_features)) 
        
        # get similarity matrix
        rgb_logits = torch.matmul(rgb_topk_score, remaining_feats.transpose(1, 2))
        ir_logits = torch.matmul(ir_topk_score, remaining_feats.transpose(1, 2))
        # vis_logits = torch.einsum('bic,btc->bit', rgb_topk_score, remaining_feats)
        # inf_logits = torch.einsum('bic,btc->bit', ir_topk_score, remaining_feats)

        # get index
        rgb_logits_indices = rgb_logits.max(-1)[1]  # bs, vis_num
        ir_logits_indices = ir_logits.max(-1)[1]  # bs, inf_num
        topk_indices = torch.cat((topk_indices, rgb_logits_indices, ir_logits_indices), dim=1)
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))

        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        rgb_topk_coords_unact = torch.gather(
            rgb_enc_outputs_coord_unact, 1,
            rgb_topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        ir_topk_coords_unact = torch.gather(
            ir_enc_outputs_coord_unact, 1,
            ir_topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        
        topk_coords = topk_coords_unact.sigmoid()
        rgb_topk_coords = rgb_topk_coords_unact.sigmoid()
        ir_topk_coords = ir_topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

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
            reference_points=reference_points,
            dn_mask=dn_mask)
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        if self.training:
            head_inputs_dict = dict(
                enc_outputs_class=topk_score,
                enc_outputs_coord=topk_coords,
                dn_meta=dn_meta)
            head_inputs_dict['rgb_enc_outputs'] = dict(
                enc_outputs_class=rgb_topk_score,
                enc_outputs_coord=rgb_topk_coords)
            head_inputs_dict['ir_enc_outputs'] = dict(
                enc_outputs_class=ir_topk_score,
                enc_outputs_coord=ir_topk_coords)
        else:
            head_inputs_dict = dict()
        return decoder_inputs_dict, head_inputs_dict
    
    def extra_pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        extra_idx: int,
    ) -> Tuple[Dict]:
        cls_out_features = self.bbox_head.cls_branches[
            self.decoder.num_layers + extra_idx].out_features

        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, memory_mask, spatial_shapes)
        enc_outputs_class = self.bbox_head.cls_branches[
            self.decoder.num_layers + extra_idx](output_memory)
        enc_outputs_coord_unact = self.bbox_head.reg_branches[
            self.decoder.num_layers + extra_idx](output_memory) + output_proposals
        
        topk_indices = torch.topk(
            enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(
            enc_outputs_class, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(
            enc_outputs_coord_unact, 1,
            topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()

        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords)
        return head_inputs_dict

    def forward_transformer(
        self,
        img_feats,#: list[Tuple[Tensor]],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:

        encoder_inputs_dict, decoder_inputs_dict = {}, {}
        for i, mod_name in enumerate(['fusion'] + self.mod_list):
            encoder_inputs_dict[mod_name], decoder_inputs_dict[mod_name] = \
            self.pre_transformer(i, img_feats[i], batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)    

        decoder_inputs_dict = decoder_inputs_dict['fusion']       
        
        if self.extra_return is None:            
            tmp_dec_in, head_inputs_dict = self.pre_decoder(
                **encoder_outputs_dict['fusion'], batch_data_samples=batch_data_samples)
        else:
            pre_decoder_input=encoder_outputs_dict['fusion']
            for extra_idx, mod in enumerate(self.extra_return, 1):
                pre_decoder_input.update(
                    {
                        f'{mod}_memory': encoder_outputs_dict[mod]['memory'],
                        f'{mod}_memory_mask': encoder_outputs_dict[mod]['memory_mask'],
                        f'{mod}_spatial_shapes': encoder_outputs_dict[mod]['spatial_shapes'],
                        f'{mod}_extra_idx': extra_idx,
                    }
                )
            tmp_dec_in, head_inputs_dict = self.pre_decoder_new(
                **pre_decoder_input, batch_data_samples=batch_data_samples)        
               
        decoder_inputs_dict.update(tmp_dec_in)
        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)
        return head_inputs_dict
    
    def forward_transformer_old(
        self,
        img_feats,#: list[Tuple[Tensor]],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:

        encoder_inputs_dict, decoder_inputs_dict = {}, {}
        for i, mod_name in enumerate(['fusion'] + self.mod_list):
            encoder_inputs_dict[mod_name], decoder_inputs_dict[mod_name] = \
            self.pre_transformer(i, img_feats[i], batch_data_samples)

        encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict)           
        
        decoder_inputs_dict = decoder_inputs_dict['fusion']
        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict['fusion'], batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)
        # extra loss
        if self.training and (self.extra_return is not None):
            for extra_idx, mod in enumerate(self.extra_return, 1):
                if mod == 'rgb':
                    extra_head_inputs_dict = self.extra_pre_decoder(**encoder_outputs_dict[mod], extra_idx=extra_idx)
                    head_inputs_dict['rgb_enc_outputs'] = extra_head_inputs_dict
                elif mod == 'ir':
                    extra_head_inputs_dict = self.extra_pre_decoder(**encoder_outputs_dict[mod], extra_idx=extra_idx)
                    head_inputs_dict['ir_enc_outputs'] = extra_head_inputs_dict            

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
        x = [x_V1 + x_I1 for x_V1, x_I1 in zip(x_V, x_I)]   # 简单的sum融合
        return [tuple(x), x_V, x_I]   # fusion, rgb, ir
        
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
