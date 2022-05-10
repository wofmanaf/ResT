# Installation

We provide installation instructions for ImageNet classification experiments here.

## Dependency Setup
Create a new conda virtual environment
```
conda create -n rest python=3.9 -y
conda activate rest
```

Install [PyTorch](https://pytorch.org/) >= 1.8.0, [torchvision](https://pytorch.org/vision/stable/index.html) >=0.9.0 following official instructions. For example:
```
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Clone this repo and install required packages:
```
pip install timm==0.5.4 tensorboardX six
```

The results in the paper are generated with `torch==1.8.0+cu111 torchvision==0.9.0+cu111 timm==0.5.4`.

## Dataset Preparation

Download the [ImageNet-1K](http://image-net.org/) classification dataset and structure the data as follows:
```
/path/to/imagenet-1k/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```