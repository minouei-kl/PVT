_base_ = [
    './_base_/models/faster_rcnn_r50_fpn.py',
    './_base_/datasets/coco_detection.py',
    './_base_/schedules/schedule_1x.py', './_base_/default_runtime.py'
]


# optimizer
optimizer = dict(_delete_=True, type='AdamW', lr=0.0001, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=1, norm_type=2))
# learning policy
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=False,
        poly2mask=False),
    dict(
        type='Resize',
        img_scale=[(512, 640)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.0),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=8,
    train=dict(
        type='CocoDataset',
        classes=('table',),
        ann_file='/netscratch/minouei/alltables/trainmerge.json',
        img_prefix='/netscratch/minouei/alltables/images/',
        pipeline=train_pipeline),
    val=dict(
        type='CocoDataset',
        classes=('table',),
        ann_file='/netscratch/minouei/alltables/valmerge.json',
        img_prefix='/netscratch/minouei/alltables/images/',
        pipeline=test_pipeline),
    test=dict(
        type='CocoDataset',
        classes=('table',),
        ann_file='/netscratch/minouei/alltables/testmerge.json',
        img_prefix='/netscratch/minouei/alltables/images/',
        pipeline=test_pipeline))
lr_config = dict(policy='step', step=[1, 2])
runner = dict(type='EpochBasedRunner', max_epochs=3)
log_config = dict(
    interval=150,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
