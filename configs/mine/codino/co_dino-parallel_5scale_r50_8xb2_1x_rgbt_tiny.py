_base_ = [
    '../_base_/datasets/ssj_scp_rgbt_tiny.py', 
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
num_dec_layer = 6
loss_lambda = 2.0
num_classes = 7
num_levels = 5
mode = 'loss'
# mode = 'tensor'
extra_return=['ir',]  # this is for encoder output and head input

image_size = (1024, 1024)
# image_size = (512, 512)
batch_augments = [
    dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
]
model = dict(
    type='CoDETR_parallel_20250731',
    # If using the lsj augmentation,
    # it is recommended to set it to True.
    use_lsj=False,
    # detr: 52.1
    # one-stage: 49.4
    # two-stage: 47.9
    eval_module='detr',  # in ['detr', 'one-stage', 'two-stage']
    num_queries=900,  # num_matching_queries
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=num_levels,
    with_coord_feat=False,
    num_co_heads=2,  # ATSS Aux Head + Faster RCNN Aux Head
    mode=mode,
    extra_return=extra_return,
    data_preprocessor=dict(
        type='MultiModalDetDataPreprocessor',
        mean={"img": [95.70, 99.40, 99.62], "ir_img": [87.71, 87.71, 87.71]},
        std={"img": [42.79, 41.22, 43.68], "ir_img": [49.66, 49.66, 49.66]},
        bgr_to_rgb=True,
        pad_mask=False),
    backbone=dict(
        type='ResNet',
        with_cp=False if mode == 'tensor' else True,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[256, 512, 1024, 2048],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_levels),
    encoder=dict(
        mode=mode,
        extra_return=extra_return,
        num_layers=6,
        num_cp=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=num_levels, dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0))),  # 0.1 for DeformDETR
    decoder=dict(
        mode=mode,
        extra_return=extra_return,
        num_layers=6,
        # num_cp=4,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),  # 0.1 for DeformDETR
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_levels, dropout=0.0),  # 0.1 for DeformDETR
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)),  # 0.1 for DeformDETR
        post_norm_cfg=None),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,  # -0.5 for DeformDETR
        temperature=20),  # 10000 for DeformDETR
    bbox_head=dict(
        type='CoDINOHead_Fusion_var',
        num_classes=num_classes,
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='QualityFocalLoss',    # CODINO
            beta=2.0,           # CODINO
            # type='FocalLoss',   # DINO
            # gamma=2.0,          # DINO
            # alpha=0.25,         # DINO
            use_sigmoid=True,
            loss_weight=1.0),  # 2.0 in DeformDETR
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_var=dict(type='GaussianNLLLoss', loss_weight=0.1),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)])),
        test_cfg=dict(
            # Deferent from the DINO, we use the NMS.
            max_per_img=300,
            # NMS can improve the mAP by 0.2.
            nms=dict(type='soft_nms', iou_threshold=0.8))),
    dn_cfg=dict(  # TODO: Move to model.train_cfg ?
        label_noise_scale=0.5,
        box_noise_scale=1.0,  # 0.4 for DN-DETR
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),  # TODO: half num_dn_queries
    rpn_head=dict(
        type='RPNHead',
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,    # Different from the Faster RCNN 
            scales_per_octave=3,    # Different from the Faster RCNN 
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128]),   # Different from the Faster RCNN, 这个要和encoder的输出层数一样（是encoder的输出reshape HxW之后作为特征图输入）
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0 * num_dec_layer * loss_lambda), # Different from the Faster RCNN 
        loss_bbox=dict(
            type='L1Loss', loss_weight=1.0 * num_dec_layer * loss_lambda),   # Different from the Faster RCNN 
        train_cfg=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            rpn_proposal=dict(
                nms_pre=4000,   # Different from the Faster RCNN 
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        test_cfg=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0)),
    roi_head=[
        dict(
            type='CoStandardRoIHead',   # Different from the Faster RCNN 
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64], # Different from the Faster RCNN , 这个要和encoder的输出层数一样
                finest_scale=56),   # Different from the Faster RCNN 
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,  # Different from the Faster RCNN 
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda), # Different from the Faster RCNN 
                loss_bbox=dict(
                    type='GIoULoss',    # Different from the Faster RCNN 
                    loss_weight=10.0 * num_dec_layer * loss_lambda)),
            train_cfg=dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            test_cfg=dict(
                score_thr=0.0,  # Different from the Faster RCNN 
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100))
    ],
    OneStage_head=[
        dict(
            type='CoATSSHead',  # Different from the ATSS
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),   # Different from the ATSS, 这个要和encoder的输出层数一样
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda), # Different from the ATSS
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda), # Different from the ATSS
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            train_cfg=dict(
                assigner=dict(type='ATSSAssigner', topk=9),
                allowed_border=-1,
                pos_weight=-1,
                debug=False),
            test_cfg=dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.0,  # Different from the ATSS
                nms=dict(type='nms', iou_threshold=0.6),
                max_per_img=100))
    ]
)

# train_pipeline, NOTE the img_scale and the Pad's size_divisor is different
# from the default setting in mmdet.
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
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    resize_type='MultiModalResize',
                    mod_list=_base_.mod_list, 
                    scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                            (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                            (736, 1333), (768, 1333), (800, 1333)],
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
                            (736, 1333), (768, 1333), (800, 1333)],
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
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        return_classes=True))

val_dataloader = dict(dataset=dict(pipeline=test_pipeline, return_classes=True))
test_dataloader = val_dataloader


optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=2e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}))

val_evaluator = dict(metric='bbox')
test_evaluator = val_evaluator


# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=16, enable=True)