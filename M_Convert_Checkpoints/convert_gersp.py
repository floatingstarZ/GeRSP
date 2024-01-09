import torch
import os
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

src_root = './results/GeRSP'
for src_ckpt_file in ['epoch_100.pth']:
    src_ckpt_pth = src_root + '/' + src_ckpt_file
    ckpt_name = Path(src_ckpt_pth).stem

    out_ckpt_pth = src_root + f'/CVT_{ckpt_name}.pth'
    src_model = torch.load(src_ckpt_pth, map_location='cpu')
    out_state_dict = OrderedDict()

    for name, parameters in src_model['state_dict'].items():
        if 'backbone' not in name:
            continue
        out_state_dict[name] = deepcopy(parameters)
    out_model = dict(
        state_dict=out_state_dict
    )
    torch.save(out_model, out_ckpt_pth)
    print(f'Save: {out_ckpt_pth}')




