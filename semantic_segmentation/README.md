# ADE20k Semantic segmentation with ResTv2

## Getting started 

We add ResTv2 model and config files to the semantic_segmentation.

## Results and Fine-tuned Models

| name | Pretrained Model | Method | Crop Size | Lr Schd | mIoU (ms+flip) | #params | FLOPs | FPS | Fine-tuned Model |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:| :---:| :---:|
| ResTv2-T | ImageNet-1K | UPerNet | 512x512 | 160K | 47.3 | 62.1M | 977G  | 22.4 | [baidu](https://pan.baidu.com/s/1X-hAafTLFnwJPQSI2BNOKw) |
| ResTv2-S | ImageNet-1K | UPerNet | 512x512 | 160K | 49.2 | 72.9M | 1035G | 20.0 | [baidu](https://pan.baidu.com/s/1WHiL0Rf9JeOB76yh6WOvLQ) |
| ResTv2-B | ImageNet-1K | UPerNet | 512x512 | 160K | 49.6 | 87.6M | 1095G | 19.2 | [baidu](https://pan.baidu.com/s/1dtkg68j3vCU-dxJxl8VFdg) |

### Training

Command format:
```
tools/dist_train.sh <CONFIG_PATH> <NUM_GPUS> --work-dir <SAVE_PATH> --seed 0 --deterministic --options model.pretrained=<PRETRAIN_MODEL>
```

For example, using a `ResTv2-T` backbone with UperNet:
```bash
bash tools/dist_train.sh \
    configs/ResTv2/upernet_restv2_tiny_512_160k_ade20k.py 8 \
    --work-dir /path/to/save --seed 0 --deterministic \
    --options model.pretrained=ResTv2_tiny_224.pth
```

More config files can be found at [`configs/ResTv2`](configs/ResTv2).


## Evaluation

Command format:
```
tools/dist_test.sh <CONFIG_PATH> <CHECKPOINT_PATH> <NUM_GPUS> --eval mIoU --aug-test
```

For example, evaluate a `ResTv2-T` backbone with UperNet:
```bash
bash tools/dist_test.sh configs/ResTv2/upernet_ResTv2_tiny_512_160k_ade20k.py \ 
    upernet_restv2_tiny_512_160k_ade20k.pth 4 --eval mIoU --aug-test
```

## Acknowledgment 

This code is built using the [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) library, [Timm](https://github.com/rwightman/pytorch-image-models) library.