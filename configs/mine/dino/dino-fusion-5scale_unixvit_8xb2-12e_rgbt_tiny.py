_base_ = './dino-fusion-4scale_r50_8xb2-12e_rgbt_tiny.py'

pretrained = '/workspace/mmdetection/pretrained/dualvit_s_384.pth.tar'  # noqa
num_levels = 5
model = dict(
    num_feature_levels=num_levels,    
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
        with_cp=True,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[64, 128, 320, 448], num_outs=num_levels),
    encoder=dict(num_cp=6, layer_cfg=dict(self_attn_cfg=dict(num_levels=num_levels))),
    decoder=dict(layer_cfg=dict(cross_attn_cfg=dict(num_levels=num_levels))))