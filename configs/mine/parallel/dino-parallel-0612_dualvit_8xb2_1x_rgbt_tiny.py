_base_ = [
    './dino-parallel-0612_r50_8xb2_1x_rgbt_tiny.py',
]
pretrained = '/workspace/mmdetection/dualvit_s_384.pth.tar'
# pretrained = '/workspace/mmdetection/swin_large_patch4_window12_384_22k.pth'
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='DualVit',
        stem_width=32,
        embed_dims=[64, 128, 320, 448],
        num_heads=[2, 4, 10, 14],
        mlp_ratios=[8, 8, 4, 3],
        norm_layer='LN',
        depths=[3, 4, 6, 3],
        drop_path_rate=0.15,
        # drop_path_rate=0.3,
        # score_embed_nums=3,
        # num_scores=2,
        # mod_nums=2,
        out_indices=(1, 2, 3),
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),    
    # backbone=dict(
    #     _delete_=True,
    #     type='SwinTransformer',
    #     pretrain_img_size=384,
    #     embed_dims=192,
    #     depths=[2, 2, 18, 2],
    #     num_heads=[6, 12, 24, 48],
    #     window_size=12,
    #     mlp_ratio=4,
    #     qkv_bias=True,
    #     qk_scale=None,
    #     drop_rate=0.,
    #     attn_drop_rate=0.,
    #     drop_path_rate=0.2,
    #     patch_norm=True,
    #     out_indices=(1, 2, 3),
    #     # Please only add indices that would be used
    #     # in FPN, otherwise some parameter will not be used
    #     with_cp=True,
    #     convert_weights=True,
    #     init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(
        type='ChannelMapper',
        in_channels=[128, 320, 448],    # dual vit s
        # in_channels=[384, 768, 1536], # swin l
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