# dataset settings
dataset_type = 'RGBTTinyDataset'
data_root = '/datasets/RGBT-Tiny/'

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
mod_list = ["img", "ir_img"]
image_size = (1024, 1024)
# image_size = (512, 512)

# Standard Scale Jittering (SSJ) resizes and crops an image
# with a resize range of 0.8 to 1.25 of the original image size.
load_pipeline = [
    dict(
        type='LoadMultiModalImages', 
        mod_path_mapping_dict={"RGBT-Tiny": {
            "img": {'org_key': 'images', 'target_key': 'images'}, 
            "ir_img": {'org_key': 'images', 'target_key': 'images'}}},
        mod_list=mod_list, 
        # replace_str=['/00/', '/01/'],   # raw, aim
        backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='MultiModalRandomResize',
        scale=image_size,
        ratio_range=(0.8, 1.25),
        keep_ratio=True),   # ?
    dict(
        type='MultiModalRandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),  # ?
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),    # ?
    dict(type='MultiModalRandomFlip', prob=0.5, mod_list=mod_list),
    dict(type='Pad', size=image_size),  # ?
]
train_pipeline = [
    dict(type='MultiModalCopyPaste', max_num_pasted=100),   # ?
    dict(type='PackMultiModalDetInputs')
]

test_pipeline = [
    dict(
        type='LoadMultiModalImages', 
        mod_path_mapping_dict={"RGBT-Tiny": {
            "img": {'org_key': 'images', 'target_key': 'images'}, 
            "ir_img": {'org_key': 'images', 'target_key': 'images'}}},
        mod_list=mod_list, 
        backend_args=backend_args),
    dict(type='MultiModalResize', scale=(1333, 800), keep_ratio=True, mod_list=mod_list),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackMultiModalDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader=dict(
    batch_size=1,
    num_workers=12,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=dict(
            img='annotations_coco/instances_00_train2017.json',
            # img='annotations_coco/instances_00_train2017_inf_ann.json',
            ir_img='annotations_coco/instances_01_train2017.json'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))

val_dataloader=dict(
        batch_size=1,
        num_workers=12,
        persistent_workers=True,
        drop_last=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img='images/'),
        ann_file=dict(
            img='annotations_coco/instances_00_test2017.json',
            # img='annotations_coco/instances_00_test2017_inf_ann.json',
            ir_img='annotations_coco/instances_01_test2017.json'),
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
