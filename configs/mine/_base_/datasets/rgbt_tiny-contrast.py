# dataset settings
dataset_type = 'RGBTTinyDataset'
data_root = '/workspace/mmdetection/datasets/RGBT-Tiny/'

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
train_mod_list = ["img", "ir_img", "con_img"]
val_mod_list = ["img", "ir_img"]

train_pipeline = [
    dict(
        type='LoadMultiModalImages', 
        mod_path_mapping_dict={"RGBT-Tiny": {
            "img": {'org_key': 'images', 'target_key': 'images'}, 
            "ir_img": {'org_key': 'images', 'target_key': 'images'}}},
        mod_list=val_mod_list, 
        backend_args=backend_args),
    dict(type='LoadContrastImages', con_mod='ir_img', interval=30, 
         mode=['hard']
        #  mode=['easy', 'hard']
         ),  # important
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='MultiModalResize', scale=(1333, 800), keep_ratio=True, mod_list=train_mod_list),
    dict(type='MultiModalRandomFlip', prob=0.5, mod_list=train_mod_list),
    dict(
        type='PackMultiModalDetInputs',
        mod_list=train_mod_list,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                            'scale_factor', 'flip', 'flip_direction'))
]
test_pipeline = [
    dict(type='LoadMultiModalImages', 
        mod_path_mapping_dict={"RGBT-Tiny": {
            "img": {'org_key': 'images', 'target_key': 'images'}, 
            "ir_img": {'org_key': 'images', 'target_key': 'images'}}},
        mod_list=val_mod_list, 
        backend_args=backend_args),
    dict(type='MultiModalResize', scale=(1333, 800), keep_ratio=True, mod_list=val_mod_list),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackMultiModalDetInputs',
        mod_list=val_mod_list,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=4,
    num_workers=32,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations_coco/instances_00_train2017.json',
        ann_file='annotations_coco/instances_00_train2017_0.2.json',
        data_prefix=dict(img='images/'),
        # ann_file=[
        #     'annotations_coco/instances_00_train2017.json',
        #     'annotations_coco/instances_01_train2017.json'],
        # data_prefix=dict(img=['images/', 'images/']),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args))
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        # ann_file='annotations_coco/instances_00_test2017.json',
        ann_file='annotations_coco/instances_00_test2017_0.5.json',
        data_prefix=dict(img='images/'),
        # ann_file=[
        #     'annotations_coco/instances_00_test2017.json',
        #     'annotations_coco/instances_01_test2017.json'],
        # data_prefix=dict(img=['images/', 'images/']),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    # ann_file=data_root + 'annotations_coco/instances_00_test2017.json',
    ann_file=data_root + 'annotations_coco/instances_00_test2017_0.5.json',
    # ann_file=[
    #     data_root + 'annotations_coco/instances_00_test2017.json',
    #     data_root + 'annotations_coco/instances_01_test2017.json',],
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
test_evaluator = val_evaluator

# inference on test dataset and
# format the output results for submission.
# test_dataloader = dict(
#     batch_size=1,
#     num_workers=2,
#     persistent_workers=True,
#     drop_last=False,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#         data_prefix=dict(img='test2017/'),
#         test_mode=True,
#         pipeline=test_pipeline))
# test_evaluator = dict(
#     type='CocoMetric',
#     metric='bbox',
#     format_only=True,
#     ann_file=data_root + 'annotations/image_info_test-dev2017.json',
#     outfile_prefix='./work_dirs/coco_detection/test')
