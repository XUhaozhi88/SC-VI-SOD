# dataset settings
dataset_type = 'RGBTTinyRGBDataset'
data_root = '/datasets/RGBT-Tiny/'
image_size = (1024, 1024)
# image_size = (512, 512)
# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection/coco/'

# Method 2: Use `backend_args`, `file_client_args` in versions before 3.0.0rc6
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection/',
#         'data/': 's3://openmmlab/datasets/detection/'
#     }))
backend_args = None

# Standard Scale Jittering (SSJ) resizes and crops an image
# with a resize range of 0.8 to 1.25 of the original image size.
load_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='RandomResize',
        scale=image_size,
        ratio_range=(0.8, 1.25),
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=image_size),
]
train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=100),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataset=dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations_coco/instances_00_train2017.json',
        # ann_file='annotations_coco/instances_00_train2017_0.2.json',
        # ann_file='annotations_coco/instances_00_train2017_small.json',
        data_prefix=dict(img='images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=load_pipeline,
        backend_args=backend_args),
    pipeline=train_pipeline
)

train_dataloader=dict(
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler'),
    dataset=train_dataset)

val_dataloader=dict(
        batch_size=1,
        num_workers=12,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='annotations_coco/instances_00_test2017.json',
            # ann_file='annotations_coco/instances_00_test2017_0.5.json',
            data_prefix=dict(img='images/'),
            test_mode=True,
            pipeline=test_pipeline,
            backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator=dict(
        # type=CocoMetric,
        # type='CocoSmallMetric',
        type='CocoSAFitMetric',
        ann_file=data_root + 'annotations_coco/instances_00_test2017.json',
        # ann_file=data_root + 'annotations_coco/instances_00_test2017_0.5.json',
        metric='bbox',
        format_only=False,
        backend_args=backend_args)
test_evaluator = val_evaluator
