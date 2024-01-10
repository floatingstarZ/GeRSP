
## Generic Knowledge Boosted Pre-training For Remote Sensing Images

![diagram](.github/images/GeRSP_diagram.png)

This is the official PyTorch implementation of the [GeRSP](https://arxiv.org/abs/2103.16607):
```
@article{huang2024generic,
  title={Generic Knowledge Boosted Pre-training For Remote Sensing Images},
  author={Ziyue Huang and Mingming Zhang and Yuan Gong and Qingjie Liu and Yunhong Wang},
  journal={arXiv preprint arXiv:2401.04614},
  year={2024}
}
@misc{mmselfsup2021,
    title={{MMSelfSup}: OpenMMLab Self-Supervised Learning Toolbox and Benchmark},
    author={MMSelfSup Contributors},
    howpublished={\url{https://github.com/open-mmlab/mmselfsup}},
    year={2021}
}
```


### Preparation

Install Python dependencies by running:
```shell
#----- Create conda environment
conda create --name gersp python=3.9 -y
conda activate gersp

#----- Install pytorch. We use PyTorch: 1.11.0 or 1.12.1 in our experiments.
#----- You could also use pip to install pytorch following https://pytorch.org/get-started/previous-versions/
conda install pytorch torchvision -c pytorch

#----- Install others. We use mmcv-full: 1.7.1 or 1.6.2 in our experiments.
pip install -U openmim
pip install -v -e .
pip install mmcv-full==1.6.2
pip install mmengine==0.9.0
pip install yapf==0.40.1

#----- MMDetection need these
pip install terminaltables
pip install pycocotools

```


### Datasets

First, download Million-AID dataset by following https://captain-whu.github.io/DiRS/. Afterward, unzip all the files in each folder.

Then, download ImageNet dataset and also unzip all files. 

Next, create a 'data' directory and place the ImageNet and Million-AID datasets into the 'data' directory according to the following directory format.

```shell
data
|---million_aid
|   |---test
|   |   |---P0000000.jpg
|   |   |---P0000001.jpg
|   |   |...
|   |...
|---ImageNet
|   |---train
|   |   |---n01440764
|   |   |---n01443537
|   |   |---n01484850
|   |   |...
|   |...
```


### Pre-training

Use the following command for pre-training ResNet-50:
```shell
bash ./tools/dist_train_GeRSP.sh
```
GeRSP can be pre-trained using eight 2080Ti (10GB). You can reduce the number of GPUs by increasing the batch size and using GPUs with larger memory, but aligning performance is not guaranteed.

### Model Convert

Use the following commands to convert the pre-trained model into usable weights:
```shell
python ./M_Convert_Checkpoints/convert_gersp.py
```
We also provide model conversion code for [CMID](https://github.com/NJU-LHRS/official-CMID) and [GeoAware](https://github.com/sustainlab-group/geography-aware-ssl), allowing to download their pre-trained weights and perform the conversion. Other methods we compared were also converted in a similar manner.


### Downstream tasks

To train a Faster R-CNN on [DIOR](https://gcheng-nwpu.github.io/#Datasets) from a pre-trained GeRSP model, run:
```shell
python ./train_mmdet.py ./configs_mmdet/faster_rcnn_GeRSP.py  --work-dir ./results/faster_rcnn_GeRSP
```
Similar approaches can be applied to train RetinaNet and Dynamic R-CNN, the results can be found in our paper. For semantic segmentation and classification, we conducted experiments using [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [MMClassification](https://github.com/open-mmlab/mmpretrain), with experimental configurations detailed in the paper.



### Pre-trained Models

Our pre-trained GeRSP models can be downloaded as following:

| Name      | architecture | epochs | google drive                                                                           | Baidu Cloud                                                             |
|-----------| ------------ |--------|----------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| GeRSP     | ResNet-50    | 100    | [download](https://drive.google.com/file/d/1IQ6wHL5MiPZt9_cecRd8gkFNsf-0snIH/view?usp=sharing) | [download](https://pan.baidu.com/s/1E7CQUmt0bhWFKT76wrbQ-Q?pwd=wtyd) (wtyd) |
| GeRSP200  | ResNet-50    | 200    | [download](https://drive.google.com/file/d/1qGkPG0j4jwItcGztX-_rLR4sS2W_Pxun/view?usp=sharing)   |  [download](https://pan.baidu.com/s/1keKUzeyIcoFhUdNGJEluVw?pwd=ntjq) (ntjq) |