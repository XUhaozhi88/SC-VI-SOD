_base_ = [
    './dino-fusion-4scale_r50_8xb2-12e_rgbt_tiny.py',
]
pretrained = '/workspace/mmdetection/mmpedestron_pretrained_best_best-d02b2b88.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='UNIXVit',
        stem_width=32,
        embed_dims=[64, 128, 320, 448],
        num_heads=[2, 4, 10, 14],
        mlp_ratios=[8, 8, 4, 3],
        norm_layer='LN',
        depths=[3, 4, 6, 3],
        drop_path_rate=0.3,
        score_embed_nums=3,
        num_scores=2,
        mod_nums=2,
        out_indices=(1, 2, 3),
        with_cp=True,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
        ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 320, 448],
        # in_channels=[64, 128, 320, 448],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4)
)


accumulative_counts=2
accumulative_counts=4 / _base_.train_dataloader['batch_size']
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    # type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,  # 0.0002 for DeformDETR
        weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=accumulative_counts,
    paramwise_cfg=dict(custom_keys={
        'absolute_pos_embed': dict(decay_mult=0.),
        'backbone': dict(lr_mult=0.1)
    }))
auto_scale_lr = dict(base_batch_size=16, enable=True)