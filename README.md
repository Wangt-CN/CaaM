## CaaM

This repo contains the codes of training our [CaaM](https://arxiv.org/abs/2108.08782) on NICO/ImageNet9 dataset. Due to my recent limited bandwidth, this codebase is still messy, which will be further refined and checked recently.



#### 0. Bibtex

If you find our codes helpful, please cite our paper:

```
@inproceedings{wang2021causal,
  title={Causal Attention for Unbiased Visual Recognition},
  author={Wang, Tan and Zhou, Chang and Sun, Qianru and Zhang, Hanwang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2021}
}
```



#### 1. Preparation

1) Installation: Python3.6, Pytorch1.6, tensorboard, timm(0.3.4), scikit-learn, opencv-python, matplotlib, yaml
2) Dataset: 

- NICO: Please download from https://drive.google.com/file/d/1topMf4xqLpbhI1X6fs3hf8_M1ytieLqP/view?usp=sharing, we remove the damaged images in original NICO and rename the images. The construction details of our proposed subset are in our Appendix.
- ImageNet9: Please follow the usual practice to download the ImageNet (ILSVRC2015) dataset.

3) Please remember to change the data path in the config file.



#### 2. Evaluation:

1) For ResNet18 on NICO dataset

```
CUDA_VISIBLE_DEVICES=0 python train.py -cfg conf/ours_resnet18_multilayer2_bf0.02_noenv_pw5e5.yaml -debug -gpu -eval pretrain_model/nico_resnet18_ours_caam-best.pth
```

The results will be: Val Score: 0.4638461470603943  Test Score: 0.4661538600921631

2) For T2T-ViT7 on NICO dataset

```
CUDA_VISIBLE_DEVICES=0,1 python train.py -cfg conf/ours_t2tvit7_bf0.02_s4_noenv_pw5e4.yaml -debug -gpu -multigpu -eval pretrain_model/nico_t2tvit7_ours_caam-best.pth
```

The results will be: Val Score: 0.3799999952316284  Test Score: 0.3761538565158844

3) For ImageNet-9 dataset

Similarly, the pretrained model is in `pretrain_model`. Please note that on ImageNet9, we report the best performance for the 3 metrics in our paper. The pretrained model is for `bias` and `unbias` and we did not save the model for the best `ImageNet-A`. 



#### 3. Train

To perform training, please run the sh file in scripts. For example:

```
sh scripts/run_baseline_resnet18.sh
```



#### **4. An interesting finding**

Recently I found an interesting thing by accident. The `mixup` added on the baseline model would not bring much performance improvements (see Table 1. in the main paper). However, when performing `mixup` based on our CaaM, the performance can be further boosted.

Specifically, you can active the `mixup` by:

```
sh scripts/run_ours_resnet18_mixup.sh
```

This can make our CaaM achieve about **50~51%** Val & Test accuracy on NICO dataset.





#### **Acknowledgement**

Special thanks to the authors of [ReBias](https://github.com/clovaai/rebias) and [IRM](https://github.com/facebookresearch/InvariantRiskMinimization), and the datasets used in this research project.

If you have any question or find any bug, please kindly email [me](TAN317@e.ntu.edu.sg).