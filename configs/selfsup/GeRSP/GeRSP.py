_base_ = [
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='MoCoGeRSP',
    queue_len=65536,
    feat_dim=128,
    momentum=0.999,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    num_classes=1000,
    head=dict(type='ContrastiveHead', temperature=0.07))
# runtime settings
# the max_keep_ckpts controls the max number of ckpt file in your work_dirs
# if it is 3, when CheckpointHook (in mmcv) saves the 4th ckpt
# it will remove the oldest one to keep the number of total ckpts as 3
checkpoint_config = dict(interval=1)

log_config = dict(
    interval=5,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

##########################################################################
prefetch = False
dataset_type = 'MultiViewDACPDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
    dict(type='RandomHorizontalFlip')
]
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

train_pipeline_ex = [
    dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='ColorJitter',
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.4),
    dict(type='RandomHorizontalFlip')
]
if not prefetch:
    train_pipeline_ex.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=32,  # total 32*8=256, 64*4=256
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type='ImageNet',
            data_prefix='./data/ImageNet/train',
            ann_file='./data/ImageNet/meta/train.txt',
        ),
        ex_data_source=dict(
            type='ImgFolder',
            data_prefix='./data/million_aid/test',
        ),
        num_views=[1],
        pipelines=[train_pipeline],
        ex_num_views=[2],
        ex_pipelines=[train_pipeline_ex],
        prefetch=prefetch,
    ))

optimizer = dict(type='SGD',
                 lr=0.007,
                 # lr=0.05,
                 weight_decay=5e-4,
                 momentum=0.9)
optimizer_config = dict()  # grad_clip, coalesce, bucket_size_mb

# learning policy
# see check_lr.py
lr_config = dict(policy='CosineRestart',
                 min_lr=0.001,
                 periods=[20] * 5,
                 restart_weights=[1] * 5
                 )

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
