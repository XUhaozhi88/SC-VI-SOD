_base_ = [
    '../_base_/datasets/rgbt_tiny.py', '../_base_/default_runtime.py'
]
model = dict(
    type='RetinaNet_Fusion_Simple',
    data_preprocessor=dict(
        type='MultiModalDetDataPreprocessor',
        mean={
            "img": [95.70, 99.40, 99.62], 
            "ir_img": [87.71, 87.71, 87.71]
        },
        std={
            "img": [42.79, 41.22, 43.68], 
            "ir_img": [49.66, 49.66, 49.66]
        },
        bgr_to_rgb=True,
        pad_size_divisor=1),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
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
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=7,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

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
train_dataloader = dict(
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False), pipeline=train_pipeline))

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)})
)  # custom_keys contains sampling_offsets and reference_points in DeformDETR  # noqa

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
