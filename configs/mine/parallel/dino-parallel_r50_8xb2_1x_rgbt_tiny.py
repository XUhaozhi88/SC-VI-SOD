_base_ = [
    '../_base_/datasets/rgbt_tiny.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
num_levels=4
mode = 'loss'
# mode = 'tensor'
num_cross_atten=1
# num_cross_atten=2
# extra_return=[]
extra_return=['ir',]  # this is for encoder output and head input
model = dict(
    type='DINO_Parallel',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    # use_autocast=True,
    mode=mode,
    extra_return=extra_return,
    data_preprocessor=dict(
        type='MultiModalDetDataPreprocessor',
        mean={"img": [95.70, 99.40, 99.62], "ir_img": [87.71, 87.71, 87.71]},
        std={"img": [42.79, 41.22, 43.68], "ir_img": [49.66, 49.66, 49.66]},
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        # with_cp=False if mode else True,
        with_cp=False if mode == 'tensor' else True,
        depth=50,
        num_stages=4,
        # out_indices=(2, 3),
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        bias=True,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_levels),
    encoder=dict(
        mode=mode,
        extra_return=extra_return,
        num_layers=6,
        # num_cp=6,
        # Fused img layer config
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0), # deformable attention
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),  
    ),
    decoder=dict(
        mode=mode,
        extra_return=extra_return,
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),  # simple attention
            cross_attn_cfg=dict(embed_dims=256, num_levels=4, dropout=0.0),  # deformable attention
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048,  # 1024 for DeformDETR
                ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128, normalize=True, offset=0.0, temperature=20),    
    bbox_head=dict(
        # type='GroundingDINOHead',
        # contrastive_cfg=dict(max_text_len=256, log_scale='auto', bias=True),
        type='DINOHead_Fusion',
        # type='DINOHead',
        num_classes=7,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None,
                       num_dn_queries=100)),  # TODO: half num_dn_queries
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                # dict(type='BinaryFocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ])),
    test_cfg=dict(max_per_img=300))

# dataset settings
train_pipeline = [
    dict(
        type='LoadMultiModalImages', 
        mod_path_mapping_dict={"RGBT-Tiny": {
            "img": {'org_key': 'images', 'target_key': 'images'}, 
            "ir_img": {'org_key': 'images', 'target_key': 'images'}}},
        mod_list=_base_.mod_list, 
        backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MultiModalRandomFlip', prob=0.5, mod_list=_base_.mod_list),
    # dict(
    #     type='RandomChoiceResize',
    #     resize_type='MultiModalResize',
    #     mod_list=_base_.mod_list, 
    #     scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
    #             #(608, 1333), (640, 1333), #(672, 1333), (704, 1333),
    #             #(736, 1333), (768, 1333), (800, 1333)
    #             ],
    #     keep_ratio=True),
    dict(
        # type='MultiModalRandomChoice',
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    resize_type='MultiModalResize',
                    mod_list=_base_.mod_list, 
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)
                            ],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    resize_type='MultiModalResize',
                    mod_list=_base_.mod_list, 
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    # type='RandomCrop',
                    type='MultiModalRandomCrop',     
                    mod_list=_base_.mod_list,                
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    resize_type='MultiModalResize',
                    mod_list=_base_.mod_list, 
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)
                            ],
                    keep_ratio=True)
            ]
        ]),
    dict(
        type='PackMultiModalDetInputs',
        mod_list=_base_.mod_list,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction'))
]

test_pipeline = [
    dict(type='LoadMultiModalImages', 
        mod_path_mapping_dict={"RGBT-Tiny": {
            "img": {'org_key': 'images', 'target_key': 'images'}, 
            "ir_img": {'org_key': 'images', 'target_key': 'images'}}},
        mod_list=_base_.mod_list, 
        backend_args=_base_.backend_args),
    dict(type='MultiModalResize', 
         scale=(1333, 800), keep_ratio=True, mod_list=_base_.mod_list, 
         is_fixscale=True),   # equal to FixScaleResize
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackMultiModalDetInputs',
        mod_list=_base_.mod_list,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False),
        pipeline=train_pipeline,
        return_classes=True))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, return_classes=True))
test_dataloader = val_dataloader

# We did not adopt the official 24e optimizer strategy
# because the results indicate that the current strategy is superior.
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1)
    }))
# learning policy
max_epochs = 12
train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16, enable=True)
# find_unused_parameters=True
