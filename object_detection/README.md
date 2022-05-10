# Object detection
ResTv1 and ResTv2 for Object Detection by detectron2

This repo contains the supported code and configuration files to reproduce object detection results of ResTv1 and ResTv2. It is based on [detectron2](https://github.com/facebookresearch/detectron2).


## Results and Models

### RetinaNet

|   Backbone   | Pretrain | Lr Schd | box mAP | mask mAP | #params | FPS |                           config                    |                          model                           |
|:------------:| :---: |:-------:|:-------:|:--------:|:-------:| :---: |:-----------------------------------------------------------:|:--------------------------------------------------------:|
| ResTv1-S-FPN | ImageNet-1K |   1x    |  40.3   |    -     |  23.4   | - | [config](configs/ResTv1/retinanet_rest_small_FPN_1x.yaml) | [baidu](https://pan.baidu.com/s/13YXVRQeNcF_3Txns8eJzZw) |
| ResTv1-B-FPN | ImageNet-1K |   1x    |  42.0   |    -     |  40.5   | - | [config](configs/ResTv1/retinanet_rest_base_FPN_1x.yaml)  | [baidu](https://pan.baidu.com/s/1hMRM5YEIGsfWfvqbuC7JWA) |

### Mask R-CNN

|   Backbone   | Pretrain | Lr Schd | box mAP | mask mAP | #params | FPS  |                           config                    |                          model                           |
|:------------:| :---: |:-------:|:-------:|:--------:|:-------:|:----:|:-----------------------------------------------------------:|:--------------------------------------------------------:|
| ResTv1-S-FPN | ImageNet-1K |   1x    |  39.6   |   37.2   |  31.2   |  -   | [config](configs/ResTv1/mask_rcnn_rest_small_FPN_1x.yaml) | [baidu](https://pan.baidu.com/s/1UfDsRGwgZcydXtj56ZFoDg) |
| ResTv1-B-FPN | ImageNet-1K |   1x    |  41.6   |   38.7   |  49.8   |  -   | [config](configs/ResTv1/mask_rcnn_rest_base_FPN_1x.yaml)  | [baidu](https://pan.baidu.com/s/1oSdMGTSBK_JDcLEq3XjY8w) |
| ResTv2-T-FPN | ImageNet-1K |   3x    |  47.6   |   43.2   |  49.9   | 25.0 | [config](configs/ResTv2/mask_rcnn_restv2_tiny_FPN_3x.yaml)  | [baidu](https://pan.baidu.com/s/16fDcEupHBZ1zHyzFFZvM3g) |
| ResTv2-S-FPN | ImageNet-1K |   3x    |  48.1   |   43.3   |  60.7   | 21.3 | [config](configs/ResTv2/mask_rcnn_restv2_small_FPN_3x.yaml) | [baidu](https://pan.baidu.com/s/1UfDsRGwgZcydXtj56ZFoDg) |
| ResTv2-B-FPN | ImageNet-1K |   3x    |  48.7   |   43.9   |  75.5   | 18.3 | [config](configs/ResTv2/mask_rcnn_restv2_base_FPN_3x.yaml)  | [baidu](https://pan.baidu.com/s/1zHQM0KqgtqQzg0-mdtx-Jg) |


## Usage
Please refer to [get_started.md](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) for installation and dataset preparation.

note: you need convert the original pretrained weights to d2 format by [convert_to_d2.py](convert_to_d2.py)

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