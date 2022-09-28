# Updates
- (2022/05/10) Code of [ResTV2](https://arxiv.org/abs/2204.07366) is released! ResTv2 simplifies the EMSA structure in
[ResTv1](https://arxiv.org/abs/2105.13677) (i.e., eliminating the multi-head interaction part) and employs an upsample
operation to reconstruct the lost medium- and high-frequency information caused by the downsampling operation.

# [ResT: An Efficient Transformer for Visual Recognition](https://arxiv.org/abs/2105.13677)

Official PyTorch implementation of **ResTv1** and **ResTv2**, from the following paper:

[ResT: An Efficient Transformer for Visual Recognition](https://arxiv.org/abs/2105.13677). NeurIPS 2021.\
[ResT V2: Simpler, Faster and Stronger](https://arxiv.org/abs/2204.07366). NeurIPS 2022.\
By Qing-Long Zhang and Yu-Bin Yang \
State Key Laboratory for Novel Software Technology at Nanjing University

--- 

<p align="center">
<img src="figures/fig_1.png" width=100% height=100% 
class="center">
</p>

**ResTv1** is initially described in [arxiv](https://arxiv.org/abs/2105.13677), which capably serves as a
general-purpose backbone for computer vision. It can tackle input images with arbitrary size. Besides, 
ResT compressed the memory of standard MSA and model the interaction between multi-heads while keeping 
the diversity ability. 

## Catalog
- [x] ImageNet-1K Training Code
- [x] ImageNet-1K Fine-tuning Code  
- [x] Downstream Transfer (Detection, Segmentation) Code

<!-- ✅ ⬜️  -->

## Results and Pre-trained Models
### ImageNet-1K trained models

|    name     | resolution |acc@1 | #params | FLOPs | Throughput | model |
|:-----------:|:---:|:---:|:-------:|:-----:|:----------:|:---:|
| ResTv1-Lite | 224x224 | 77.2 |   11M   | 1.4G  |    1246    | [baidu](https://pan.baidu.com/s/1VVzrzZi_tD3yTp_lw9tU9A)
|  ResTv1-S   | 224x224 | 79.6 |   14M   | 1.9G  |    1043    | [baidu](https://pan.baidu.com/s/1Y-MIzzzcQnmrbHfGGR0mrw)
|  ResTv1-B   | 224x224 | 81.6 |   30M   | 4.3G  |    673     | [baidu](https://pan.baidu.com/s/1HhR9YxtGIhouZ0GEA4LYlw)
|  ResTv1-L   | 224x224 | 83.6 |   52M   | 7.9G  |    429     | [baidu](https://pan.baidu.com/s/14c4u_oRoBcKOt1aTlsBBpw)
|  ResTv2-T   | 224x224 | 82.3 |   30M   | 4.1G  |    826     | [baidu](https://pan.baidu.com/s/1LHAbsrXnGsjvAE3d5zhaHQ) |
|  ResTv2-T   | 384x384 | 83.7 |   30M   | 12.7G |    319     | [baidu](https://pan.baidu.com/s/1fEMs_OrDa_xF7Cw1DiBU9w) |
|  ResTv2-S   | 224x224 | 83.2 |   41M   | 6.0G  |    687     | [baidu](https://pan.baidu.com/s/1nysV5MTtwsDLChrRa7vmZQ) |
|  ResTv2-S   | 384x384 | 84.5 |   41M   | 18.4G |    256     | [baidu](https://pan.baidu.com/s/1S1GERP-lYEJANYr17xk3dA) |
|  ResTv2-B   | 224x224 | 83.7 |   56M   | 7.9G  |    582     | [baidu](https://pan.baidu.com/s/1GH3N2_rbZx816mN87UzYgQ) |
|  ResTv2-B   | 384x384 | 85.1 |   56M   | 24.3G |    210     | [baidu](https://pan.baidu.com/s/12RBMZmf6IlJIB3lIkeBH9Q) |
|  ResTv2-L   | 224x224 | 84.2 |   87M   | 13.8G |    415     | [baidu](https://pan.baidu.com/s/1A2huwk_Ii4ZzQllg1iHrEw) |
|  ResTv2-L   | 384x384 | 85.4 |   87M   | 42.4G |    141     | [baidu](https://pan.baidu.com/s/1dlxiWexb9mho63WdWS8nXg) |


Note: Access code for `baidu` is `rest`. Pretrained models of ResTv1 is now available in [google drive](https://drive.google.com/drive/folders/1H6QUZsKYbU6LECtxzGHKqEeGbx1E8uQ9).

## Installation
Please check [INSTALL.md](INSTALL.md) for installation instructions. 

## Evaluation
We give an example evaluation command for a ImageNet-1K pre-trained, then ImageNet-1K fine-tuned ResTv2-T:

Single-GPU
```
python main.py --model restv2_tiny --eval true \
--resume restv2_tiny_384.pth \
--input_size 384 --drop_path 0.1 \
--data_path /path/to/imagenet-1k
```

This should give 
```
* Acc@1 83.708 Acc@5 96.524 loss 0.777
```

- For evaluating other model variants, change `--model`, `--resume`, `--input_size` accordingly. You can get the url to pre-trained models from the tables above. 
- Setting model-specific `--drop_path` is not strictly required in evaluation, as the `DropPath` module in timm behaves the same during evaluation; but it is required in training. See [TRAINING.md](TRAINING.md) or our paper for the values used for different models.

## Training
See [TRAINING.md](TRAINING.md) for training and fine-tuning instructions.

## Acknowledgement
This repository is built using the [timm](https://github.com/rwightman/pytorch-image-models) library.

## License
This project is released under the Apache License 2.0. Please see the [LICENSE](LICENSE) file for more information.

## Citation
If you find this repository helpful, please consider citing:

**ResTv1**
```
@inproceedings{zhang2021rest,
  title={ResT: An Efficient Transformer for Visual Recognition},
  author={Qinglong Zhang and Yu-bin Yang},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021},
  url={https://openreview.net/forum?id=6Ab68Ip4Mu}
}
```

**ResTv2**
```
@article{zhang2022rest,
  title={ResT V2: Simpler, Faster and Stronger},
  author={Zhang, Qing-Long and Yang, Yu-Bin},
  journal={arXiv preprint arXiv:2204.07366},
  year={2022}
```

## Third-party Implementation
[2022/05/26] ResT and ResT v2 have been integrated into [PaddleViT](https://github.com/BR-IDL/PaddleViT), checkout [here](https://github.com/BR-IDL/PaddleViT/tree/develop/image_classification/ResT) for the 3rd party implementation on Paddle framework!
