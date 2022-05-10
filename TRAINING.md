# Training

We provide ImageNet-1K training, and fine-tuning commands here.
Please check [INSTALL.md](INSTALL.md) for installation instructions first.

## ImageNet-1K Training 
Training on ImageNet-1K on a single machine:
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model restv2_tiny --drop_path 0.1 \
--clip_grad 1.0 --warmup_epochs 50 --epochs 300 \
--batch_size 256 --lr 1.5e-4 --update_freq 1 \
--model_ema true --model_ema_eval true \
--data_path /path/to/imagenet-1k 
--output_dir /path/to/save_results
```

## ImageNet-1K Fine-tuning
### Finetune from ImageNet-1K pre-training 
The training commands given above for ImageNet-1K use the default resolution (224). We also fine-tune these trained models with a larger resolution (384). Please specify the path or url to the checkpoint in `--finetune`.

Fine-tuning on ImageNet-1K (384x384):

Single-machine
```
python -m torch.distributed.launch --nproc_per_node=8 main.py \
--model restv2_tiny --drop_path 0.1 --input_size 384 \
--batch_size 64 --lr 1.5e-5 --update_freq 1 \
--warmup_epochs 0 --epochs 30 --weight_decay 1e-8  \
--cutmix 0 --mixup 0 --clip_grad 1.0 \
--finetune /path/to/checkpoint.pth \
--data_path /path/to/imagenet-1k \
--output_dir /path/to/save_results
```
