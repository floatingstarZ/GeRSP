# # dataset settings
# data_source = 'ImageNet'
# dataset_type = 'MultiViewDACPDataset'
# img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# train_pipeline = [
#     dict(type='RandomResizedCrop', size=224, scale=(0.2, 1.)),
#     dict(type='RandomGrayscale', p=0.2),
#     dict(
#         type='ColorJitter',
#         brightness=0.4,
#         contrast=0.4,
#         saturation=0.4,
#         hue=0.4),
#     dict(type='RandomHorizontalFlip'),
# ]
#
# # prefetch
# prefetch = False
# if not prefetch:
#     train_pipeline.extend(
#         [dict(type='ToTensor'),
#          dict(type='Normalize', **img_norm_cfg)])
#
# # dataset summary
# data = dict(
#     samples_per_gpu=32,  # total 32*8=256
#     workers_per_gpu=4,
#     drop_last=True,
#     train=dict(
#         type=dataset_type,
#         data_source=dict(
#             type=data_source,
#             data_prefix='./data/ImageNet/train',
#             ann_file='./data/ImageNet/meta/train.txt',
#         ),
#         num_views=[2],
#         pipelines=[train_pipeline],
#         prefetch=prefetch,
#     ))
# dataset settings


data_source = 'ImageNet'
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
    dict(type='RandomHorizontalFlip'),
]

# prefetch
prefetch = False
if not prefetch:
    train_pipeline.extend(
        [dict(type='ToTensor'),
         dict(type='Normalize', **img_norm_cfg)])

# dataset summary
data = dict(
    samples_per_gpu=32,  # total 32*8=256
    workers_per_gpu=4,
    drop_last=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type=data_source,
            data_prefix='./data/ImageNet/train',
            ann_file='./data/ImageNet/meta/train.txt',
        ),
        num_views=[2],
        pipelines=[train_pipeline],
        prefetch=prefetch,
    ))
