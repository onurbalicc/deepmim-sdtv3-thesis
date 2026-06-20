_base_ = [
    '../_base_/models/fpn_snn_r50.py',
    # '../_base_/models/fpn_r50.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# checkpoint_file = '/raid/ligq/lzx/spikeformerv2/seg/checkpoint/checkpoint-199.pth'
checkpoint_file = '../Classification/Model_Large/06_clean_baseline_full_finetune_random_3gpu_50ep_100cls/checkpoint-49.pth'
# checkpoint_file='/raid/ligq/lzx/ckpt/sdtv2/T4/ckpt-55M.pth'  # direct_train
model = dict(
    data_preprocessor=data_preprocessor,
    type='EncoderDecoder',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        type='Spiking_vit_MetaFormer',
        img_size_h=512,
        img_size_w=512,
        patch_size=16,
        embed_dim=[128, 256, 512, 640],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=150,
        qkv_bias=False,
        depths=8,
        sr_ratios=1,
        T=4,  # T=1 & T=4
        decode_mode='snn',
        ),
    neck=dict(
        in_channels=[64, 128, 256, 640],
        out_channels=256,
        act_cfg=None,
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        ),
    decode_head=dict(
        in_channels=[256, 256, 256, 256],
        channels=256,
        num_classes=150,
        act_cfg=None,
        # init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        )
    )

# load_from = checkpoint_file
gpu_multiples = 1  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.9, 0.999),  weight_decay=0.05),  # default 0.001
    clip_grad=dict(max_norm=1.0),
    paramwise_cfg=dict(
        custom_keys={
            # 'backbone': dict(lr_mult=),
            'neck': dict(lr_mult=2.),
            'head': dict(lr_mult=2.)}   # default 2.
        ))
#
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
# policy='poly', power=0.9, min_lr=0.0, by_epoch=False
optimizer_config = dict()
# learning policy
lr_config = dict(warmup_iters=1500)
# runtime settings

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
train_dataloader = dict(batch_size=5)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=4000,
        save_best='mIoU',
        max_keep_ckpts=5),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
