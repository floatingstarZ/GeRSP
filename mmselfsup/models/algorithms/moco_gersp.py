# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel
import torch
from mmengine.dist import all_gather, get_data_device, get_comm_device, \
    cast_data_device, get_default_group, get_world_size
from torch import distributed as torch_dist

@torch.no_grad()
def concat_all_gather_diff_size(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Performs all_gather operation on the provided tensors.

    Args:
        tensor (torch.Tensor): Tensor to be broadcast from current process.

    Returns:
        torch.Tensor: The concatnated tensor.
    """


    world_size = get_world_size(group)
    # print('world_size', world_size)
    if world_size <= 1:
        return tensor

    group = get_default_group()
    input_device = get_data_device(tensor)
    backend_device = get_comm_device(group)
    data_on_device = cast_data_device(tensor, backend_device)

    local_size = tensor.size()[0]
    local_size = torch.tensor(local_size, device=input_device)
    all_sizes = [
        torch.empty_like(local_size, device=backend_device)
        for _ in range(world_size)
    ]
    torch_dist.all_gather(all_sizes, local_size)

    max_size = max(all_sizes)

    size_diff = torch.Size([max_size - local_size, *tensor.size()[1:]])
    padding = tensor.new_zeros(size_diff)
    pad_tensor = torch.cat([tensor, padding])

    pad_tensors_gather = all_gather(pad_tensor)

    tensors_gather = []
    for t, size in zip(pad_tensors_gather, all_sizes):
        tensors_gather.append(t[:size])

    output = torch.cat(tensors_gather)
    return output

def calculate_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class Projector(torch.nn.Module):
    def __init__(
            self,
            input_dim=2048,
            output_dim=128,
            hidden_dim=2048,
            num_layers=2
    ):
        super().__init__()
        if num_layers == 2:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        else:
            raise Exception(num_layers)

    def forward(self, x):
        return self.projector(x)


@ALGORITHMS.register_module()
class MoCoGeRSP(BaseModel):
    """MoCo.

    Implementation of `Momentum Contrast for Unsupervised Visual
    Representation Learning <https://arxiv.org/abs/1911.05722>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/moco/blob/master/moco/builder.py>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
    """

    def __init__(self,
                 backbone,
                 num_classes,
                 alpha=1,
                 head=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 init_cfg=None,
                 **kwargs):
        super(MoCoGeRSP, self).__init__(init_cfg)
        # -------remove neck in encoder_q
        self.encoder_q = build_backbone(backbone)
        self.proj_q = Projector(output_dim=feat_dim)
        self.pred_q = nn.Sequential(
            nn.Linear(2048, num_classes)
        )

        self.encoder_k = build_backbone(backbone)
        self.proj_k = Projector()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # -------head
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        for param_q, param_k in zip(self.proj_q.parameters(),
                                    self.proj_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.backbone = self.encoder_q
        self.neck = self.proj_q
        assert head is not None
        self.head = build_head(head)

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.alpha = alpha

        # -------classification head


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)  # concat_all_gather_diff_size(keys)# concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, gt_label, ex_img, **kwargs):
        """

        :param img:      from ImageNet,   one view
        :param gt_label: labels for ImageNet, [0, 1, 2, ...]
        :param ex_img:   from MillionAID, two view
        :param kwargs:
        :return:
        """
        # print('gt_label', gt_label)
        # print('len(img)', len(img))
        # print('len(ex_img)', len(ex_img))
        # raise Exception()
        assert isinstance(img, list)

        im_q = ex_img[0]
        im_k = ex_img[1]
        im_labeled = img[0]

        bs_unlabeled = im_q.shape[0]
        all_q = self.encoder_q(torch.cat([im_q, im_labeled]))
        assert len(all_q) == 1
        all_q = all_q[0]
        all_q = self.avgpool(all_q).flatten(1)
        emb_q = all_q[:bs_unlabeled]
        emb_labeled = all_q[bs_unlabeled:]
        labeled_label = gt_label[0]
        ############# ------- GET unlabeled loss

        # img_unlabeled -> 2FC -> hidden vec
        q = self.proj_q(emb_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            assert len(k) == 1
            k = k[0]
            k = self.avgpool(k).flatten(1)
            k = self.proj_k(k)
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)

        # # compute key features
        # with torch.no_grad():  # no gradient to keys
        #     k = self.encoder_k(im_k)  # keys: NxC
        #     assert len(k) == 1
        #     k = k[0]
        #     k = self.avgpool(k).flatten(1)
        #     k = self.proj_k(k)
        #     k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        losses = self.head(l_pos, l_neg)

        ############# ------- GET labeled loss
        ig_features = self.pred_q(emb_labeled)
        loss_labeled = F.cross_entropy(ig_features, labeled_label)
        acc1, acc5 = calculate_accuracy(ig_features, labeled_label, topk=(1, 5))
        losses['sup_loss'] = self.alpha * loss_labeled
        losses['sup_acc1'] = acc1.detach()
        losses['sup_acc5'] = acc5.detach()
        losses['sup_alpha'] = acc1.new_tensor(self.alpha)


        # update the queue
        self._dequeue_and_enqueue(k)

        return losses
