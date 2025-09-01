_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/rgbt_tiny-IR.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(        
    neck=dict(num_outs=5),
    rpn_head=dict(
        anchor_generator=dict(
            scales=[3],    # 缩小了anchor的尺寸，即特征图上大小为kxk，之后根据特征图缩放尺度，反向再缩放到图像尺度
            strides=[4, 8, 16, 32, 64])),
    roi_head=dict(
        bbox_head=dict(num_classes=7)))               