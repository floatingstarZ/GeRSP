_base_ = [
    './base_settings.py',
    './base_dyhead.py',
]

# DACP: our method
ckpt_path = './results/GeRSP/CVT_epoch_100.pth'

# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='ResNetPretrain',
        ckpt_path=ckpt_path
        )
    )



