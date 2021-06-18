# ResT for Image Classification

This folder contains the implementation of the ResT for image classification.

## Model Zoo

### Regular ImageNet-1K trained models

| name | resolution |acc@1 | acc@5 | #params | FLOPs | FPS| 1K model |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |:---: |
| ResT-Lite  | 224x224 | 77.2 | 93.7 | 10.5M | 1.4G | 1246 | [baidu](https://pan.baidu.com/s/1lVStrppan4nbAqCEuNvRDg)
| ResT-Small | 224x224 | 79.6 | 94.9 | 13.7M | 1.9G | 1043 | [baidu](https://pan.baidu.com/s/1lVStrppan4nbAqCEuNvRDg)
| ResT-Base  | 224x224 | 81.6 | 95.7 | 30.3M | 4.3G | 673  | [baidu](hhttps://pan.baidu.com/s/1lVStrppan4nbAqCEuNvRDg)
| ResT-Large | 224x224 | 83.6 | 96.3 | 51.6M | 7.9G | 429 | [baidu](https://pan.baidu.com/s/1lVStrppan4nbAqCEuNvRDg)

Note: access code for `baidu` is `rest`.

## Usage

### Install

- Clone this repo:

```bash
git clone https://github.com/wofmanaf/ResT.git
cd ResT
```

- Create a conda virtual environment and activate it:

```bash
conda create -n rest python=3.7 -y
conda activate rest
```

- Install `CUDA==10.1` with `cudnn7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch==1.7.1` and `torchvision==0.8.2` with `CUDA==10.1`:

```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
```

- Install `timm==0.3.2`:

```bash
pip install timm==0.3.2
```

### Data preparation

We use standard ImageNet dataset, you can download it from http://image-net.org/. We provide the following two ways to
load data:

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:
  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
 
  ```
  

### Evaluation

To evaluate a pre-trained ` ResT` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --use_env main.py --model <rest-model> \
--eval --resume <checkpoint> --data-path <imagenet-path> 
```

For example, to evaluate the `ResT-Lite` with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --use_env main.py --model rest_lite \
--eval --resume rest_lite.pth --data-path <imagenet-path>
```

### Training from scratch

To train a `ResT` on ImageNet from scratch, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --use_env main.py --model <rest-model> \ 
--drop-path <drop-path> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output_dir <output-directory>]
```

For example, to train `ResT` with 8 GPU on a single node for 300 epochs, run:

`ResT-Lite`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model rest_lite \
--drop-path 0.1 --data-path <imagenet-path> --batch-size 256 --output_dir --output_dir <output-directory>
```

`ResT-Small`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model rest_small \
--drop-path 0.1 --data-path <imagenet-path> --batch-size 256 --output_dir --output_dir <output-directory>
```

`ResT-Base`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model rest_base \
--drop-path 0.2 --data-path <imagenet-path> --batch-size 256 --output_dir --output_dir <output-directory>
```

`ResT-Large`:

```bash
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --model rest_large \
--drop-path 0.3 --data-path <imagenet-path> --batch-size 240 --output_dir --output_dir <output-directory>
```

For fine-tuning a `ResT` on ImageNet, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --use_env main.py --model <rest-model>  \
--finetune <rest-model.pth> --lr 5e-6 --weight-decay 1e-8 --epochs 30 --warmup-epochs 0 --sched step --input-size 384 \
--data-path <imagenet-path> --batch-size 128 --output_dir --output_dir <output-directory>
```

