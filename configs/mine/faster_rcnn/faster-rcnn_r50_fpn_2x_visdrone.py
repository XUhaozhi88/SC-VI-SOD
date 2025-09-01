_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/visdrone.py',
    '../_base_/schedules/schedule_2x.py', '../_base_/default_runtime.py'
]

model = dict(
    # rpn_head=dict(anchor_generator=dict(scales=[5])),   # 缩小了anchor的尺寸，即特征图上大小为kxk，之后根据特征图缩放尺度，反向再缩放到图像尺度
    roi_head=dict(
        bbox_head=dict(
            num_classes=10
        )
    )
)