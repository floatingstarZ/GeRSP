# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.utils import build_from_cfg
from torchvision.transforms import Compose

from .base import BaseDataset
from .builder import DATASETS, PIPELINES, build_datasource
from .utils import to_numpy


@DATASETS.register_module()
class MultiViewDACPDataset(BaseDataset):
    """The dataset outputs multiple views of an image.

    The number of views in the output dict depends on `num_views`. The
    image can be processed by one pipeline or multiple piepelines.

    Args:
        data_source (dict): Data source defined in
            `mmselfsup.datasets.data_sources`.
        num_views (list): The number of different views.
        pipelines (list[list[dict]]): A list of pipelines, where each pipeline
            contains elements that represents an operation defined in
            `mmselfsup.datasets.pipelines`.
        prefetch (bool, optional): Whether to prefetch data. Defaults to False.

    Examples:
        >>> dataset = MultiViewDataset(data_source, [2], [pipeline])
        >>> output = dataset[idx]
        The output got 2 views processed by one pipeline.

        >>> dataset = MultiViewDataset(
        >>>     data_source, [2, 6], [pipeline1, pipeline2])
        >>> output = dataset[idx]
        The output got 8 views processed by two pipelines, the first two views
        were processed by pipeline1 and the remaining views by pipeline2.
    """

    def __init__(self,
                 data_source,
                 ex_data_source,
                 num_views,
                 pipelines,
                 ex_num_views,
                 ex_pipelines,
                 prefetch=False):
        """

        :param data_source:           ImageNet data source   (with label,      one view)
        :param ex_data_source:        MillionAID data source (without label, multi view)
        :param num_views:
        :param pipelines:
        :param ex_num_views:
        :param ex_pipelines:
        :param prefetch:
        """
        assert len(num_views) == len(pipelines)
        self.data_source = build_datasource(data_source)
        self.ex_data_source = build_datasource(ex_data_source)

        # ------- pipelines
        self.pipelines = []
        for pipe in pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.pipelines.append(pipeline)

        # ------- extra pipelines
        self.ex_pipelines = []
        for pipe in ex_pipelines:
            pipeline = Compose([build_from_cfg(p, PIPELINES) for p in pipe])
            self.ex_pipelines.append(pipeline)

        self.prefetch = prefetch

        # ------- trans
        trans = []
        assert isinstance(num_views, list)
        for i in range(len(num_views)):
            trans.extend([self.pipelines[i]] * num_views[i])
        self.num_views = num_views
        self.trans = trans

        # ------- extra trans
        ex_trans = []
        assert isinstance(ex_num_views, list)
        for i in range(len(ex_num_views)):
            ex_trans.extend([self.ex_pipelines[i]] * ex_num_views[i])
        self.ex_num_views = ex_num_views
        self.ex_trans = ex_trans


    def __getitem__(self, idx):
        results = dict()
        # ------- ImageNet data source
        if idx > len(self.data_source) - 1:
            img_idx = idx % len(self.data_source)
        else:
            img_idx = idx

        img = self.data_source.get_img(img_idx)
        data_info = self.data_source.data_infos[img_idx]
        gt_label = data_info['gt_label']
        gt_label = [gt_label for i in range(len(self.trans))]
        multi_views = list(map(lambda trans: trans(img), self.trans))
        if self.prefetch:
            multi_views = [
                torch.from_numpy(to_numpy(img)) for img in multi_views
            ]
        results['img'] = multi_views
        results['gt_label'] = gt_label

        # ------- extra (MillionAID) data source
        if idx > len(self.ex_data_source) - 1:
            ex_idx = idx % len(self.ex_data_source)
        else:
            ex_idx = idx

        img = self.ex_data_source.get_img(ex_idx)
        multi_views = list(map(lambda trans: trans(img), self.ex_trans))
        if self.prefetch:
            multi_views = [
                torch.from_numpy(to_numpy(img)) for img in multi_views
            ]
        results['ex_img'] = multi_views

        return results

    def evaluate(self, results, logger=None):
        return NotImplemented

    def __len__(self):
        l_data = len(self.data_source)
        l_ex = len(self.ex_data_source)
        return max(l_data, l_ex)
