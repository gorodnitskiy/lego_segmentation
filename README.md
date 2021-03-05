This repo contains image segmentation pipeline for Lego blocks (1 class + background). It's related to DMIA DL Fall 2019.  
Using:  
 - [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) framework  
 - [PSPNet](https://arxiv.org/abs/1612.01105) with ResNext101-32x8d (pretrained on the imagenet) as a backbone,
 [implementation by segmentation_model_pytorch](https://smp.readthedocs.io/en/latest/models.html#pspnet)  
 - [Albumentations](https://github.com/albumentations-team/albumentations) library for augmentations  
 - IoU as metric and [soft Dice](https://smp.readthedocs.io/en/latest/losses.html#segmentation_models_pytorch.losses.DiceLoss) as loss  
