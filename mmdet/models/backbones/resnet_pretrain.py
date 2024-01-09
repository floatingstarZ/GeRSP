# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
from torch.nn.modules.batchnorm import _BatchNorm

from ..builder import BACKBONES
from ..utils import ResLayer
from torchvision.models import resnet50
import torch
from collections import OrderedDict


@BACKBONES.register_module()
class ResNetPretrain(nn.Module):

    def __init__(self, ckpt_path, ckpt_type='Regular'):
        super(ResNetPretrain, self).__init__()
        self.ckpt_path = ckpt_path
        if self.ckpt_path:
            print('#'*100)
            print("Load Checkpoint from :", ckpt_path)
            self.backbone = \
                self.load_network_encoder(ckpt_path=self.ckpt_path)
        else:
            self.backbone = resnet50(pretrained=False)
            self.backbone.fc = None
            print('#'*100)
            print("Using unpretrained weight")
        self.norm_eval = True

    def forward(self, x):

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        outs = []
        for i, layer_name in enumerate(['layer1', 'layer2', 'layer3', 'layer4']):
            res_layer = getattr(self.backbone, layer_name)
            x = res_layer(x)
            outs.append(x)
        return outs


    def load_network_encoder(self, ckpt_path):
        print('Using local  network params from ' + ckpt_path)
        teacher_dict = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
        transformed_dict = OrderedDict()
        print(teacher_dict.keys())
        for k in teacher_dict.keys():
            if k.startswith("backbone."):
                print(f'~~~ Load key from backbone: {k}')
                transformed_dict[k[9:]] = teacher_dict[k]
            elif k.startswith("teacher_network") or k.startswith("student_network_classifier"):
                print(f'Pass key(Teacher, Student Classifier): {k}')
                continue
            elif k.startswith("student_network."):
                print(f'~~~ Load key from student: {k}')
                transformed_dict[k[16:]] = teacher_dict[k]
            else:
                #continue
                print(f"Unexpected key encountered: {k}")

        # We don't need the fc layer in the pretrained weight
        final_weight = OrderedDict()
        for k in transformed_dict.keys():
            if not k.startswith("fc."):
                final_weight[k] = transformed_dict[k]
        teacher_network = resnet50(pretrained=False)
        teacher_network.fc = torch.nn.Identity()
        teacher_network.load_state_dict(final_weight, strict=True)
        return teacher_network

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(ResNetPretrain, self).train(mode)
        if mode and self.norm_eval:
            for m in self.backbone.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
