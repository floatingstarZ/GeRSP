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
checkpoints from: https://github.com/sustainlab-group/geography-aware-ssl
"""

if __name__ == '__main__':
    """
    """
    ckpt_root = '/gpfsdata/home/huangziyue/data/Projects/RS_Pretrain/results/Geography-Aware'
    ckpt_list = {
        'moco_tp.pth.tar': 'GeoAware_moco_tp.pth',
        'moco_geo.pth.tar':'GeoAware_moco_geo.pth',
        'moco_geo+tp.pth.tar': 'GeoAware_moco_geo+tp.pth'
    }
    ckpt_hash = {
        'module.encoder_q': 'backbone'
    }
    for ckpt_file in ckpt_list.keys():
        src_ckpt_pth = ckpt_root + '/' + ckpt_file
        converted_ckpt_pth = ckpt_root + '/' + ckpt_list[ckpt_file]
        state_dict = torch.load(src_ckpt_pth, map_location='cpu')
        state_dict = state_dict['state_dict']

        new_state_dict = OrderedDict()
        for k in state_dict.keys():
            # ---- module.encoder_q as the backbone
            if 'module.encoder_q' not in k:
                continue
            hits = None
            for hash_k in ckpt_hash.keys():
                if hash_k in k:
                    hits = hash_k
                    break
            if hits is None:
                raise Exception('No hit: %s' % k)
            new_k = k.replace(hits, ckpt_hash[hits])
            new_state_dict[new_k] = state_dict[k]
        cvt_ckpt = dict(state_dict=new_state_dict)
        torch.save(cvt_ckpt, converted_ckpt_pth)

