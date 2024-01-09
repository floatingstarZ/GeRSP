from pathlib import Path
from copy import deepcopy
from argparse import ArgumentParser
import warnings
warnings.simplefilter('ignore', UserWarning)

import torch
from torch import nn
from copy import deepcopy
from collections import OrderedDict
"""
For reference only
checkpoints from: https://github.com/NJU-LHRS/official-CMID
"""

if __name__ == '__main__':
    """
    """
    ckpt_root = '/gpfsdata/home/huangziyue/data/Projects/RS_Pretrain/results/CMID'
    ckpt_list = {
        'CMID_ResNet50_bk_200ep': 'CMID_ResNet50_bk_200ep_convert.pth',
    }
    ckpt_hash = {
        'module.encoder_q': 'backbone'
    }
    for ckpt_file in ckpt_list.keys():
        src_ckpt_pth = ckpt_root + '/' + ckpt_file
        converted_ckpt_pth = ckpt_root + '/' + ckpt_list[ckpt_file]
        state_dict = torch.load(src_ckpt_pth, map_location='cpu')

        new_state_dict = OrderedDict()
        for k in state_dict.keys():
            new_k = 'backbone.' + k
            new_state_dict[new_k] = state_dict[k]
        cvt_ckpt = dict(state_dict=new_state_dict)
        torch.save(cvt_ckpt, converted_ckpt_pth)
