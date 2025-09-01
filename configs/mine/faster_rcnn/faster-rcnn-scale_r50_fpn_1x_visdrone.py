_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/visdrone.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    type="FasterRCNNScale",
    scale_thresholds = [8, 16, 32],
    loss_scale=dict(type='L1Loss', loss_weight=1.0),
    # rpn_head=dict(anchor_generator=dict(scales=[1])),
    roi_head=dict(bbox_head=dict(num_classes=10)
    )
)