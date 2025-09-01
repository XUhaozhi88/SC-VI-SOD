_base_ = [
    '../_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/rgbt_tiny-RGB.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(        
    # neck=dict(num_outs=5),
    # rpn_head=dict(
    #     anchor_generator=dict(
    #         scales=[1],
    #     )
    # ),
    # rpn_head=dict(
    #     anchor_generator=dict(
    #         scales=[3],    # 缩小了anchor的尺寸，即特征图上大小为kxk，之后根据特征图缩放尺度，反向再缩放到图像尺度
    #         strides=[4, 8, 16, 32, 64])
    # ),    
    # rpn_head=dict(
    #     type='RPNHeadNew',
    #     loss_bbox=dict(type='GaussianNLLLoss', loss_weight=1.0)),
    roi_head=dict(
        bbox_head=dict(
            # type='Shared2FCBBoxHeadNew',
            num_classes=7,
            # mode='mean', # 'mean', 'mean_var'
            # loss_cls=dict(_delete_=True, type='GaussianNLLLoss', loss_weight=1.0),
            # loss_bbox=dict(type='GaussianNLLLoss', loss_weight=1.0),
            # loss_cls_var=dict(type='GaussianNLLLoss', loss_weight=0.01),
            # loss_bbox_var=dict(type='GaussianNLLLoss', loss_weight=0.1),
        )
    )
)


# optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))

# custom_hooks = [
#     dict(
#         type='LossSwitchHook',
#         switch_epoch=5,  # 在第5个 epoch 切换损失函数
#         new_loss_config=dict(
#             type='GaussianNLLLoss',
#             epsilon=1e-6,
#             loss_weight=1.0
#         )
#     )
# ]
