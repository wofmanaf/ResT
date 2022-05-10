# --------------------------------------------------------
# ResT V1
# Copyright (c) VCU, Nanjing University
# Licensed under The MIT License [see LICENSE for details]
# Written by Qing-Long Zhang
# --------------------------------------------------------

import torch
import torch.nn as nn

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelP6P7, LastLevelMaxPool
from detectron2.layers import ShapeSpec


__all__ = [
    "ResT",
    "build_rest_backbone",
    "build_rest_fpn_backbone",
    "build_retinanet_rest_fpn_backbone"]


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 apply_transform=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio+1, stride=sr_ratio, padding=sr_ratio // 2, groups=dim)
            self.sr_norm = nn.LayerNorm(dim)

        self.apply_transform = apply_transform and num_heads > 1
        if self.apply_transform:
            self.transform_conv = nn.Conv2d(self.num_heads, self.num_heads, kernel_size=1, stride=1)
            self.transform_norm = nn.InstanceNorm2d(self.num_heads)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.sr_norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.apply_transform:
            attn = self.transform_conv(attn)
            attn = attn.softmax(dim=-1)
            attn = self.transform_norm(attn)
        else:
            attn = attn.softmax(dim=-1)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, apply_transform=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, apply_transform=apply_transform)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""
    def __init__(self, patch_size=16, in_ch=3, out_ch=768, with_pos=False):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=patch_size+1, stride=patch_size, padding=patch_size // 2)
        self.norm = nn.BatchNorm2d(out_ch)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv(x)
        x = self.norm(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


class BasicStem(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, with_pos=False):
        super(BasicStem, self).__init__()
        hidden_ch = out_ch // 2
        self.conv1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(hidden_ch)
        self.conv2 = nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(hidden_ch)
        self.conv3 = nn.Conv2d(hidden_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

        self.act = nn.ReLU(inplace=True)
        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act(x)

        x = self.conv3(x)
        if self.with_pos:
            x = self.pos(x)
        return x


class ResT(Backbone):
    def __init__(self, cfg, in_ch=3, embed_dims=[64, 128, 256, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4],
                 qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., depths=[2, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1], norm_layer=nn.LayerNorm, apply_transform=True, out_features=None):
        super().__init__()
        self.depths = depths
        self.num_layers = len(depths)
        self.apply_transform = apply_transform
        self._out_features = out_features

        self.stem = BasicStem(in_ch=in_ch, out_ch=embed_dims[0], with_pos=True)

        self.patch_embed_2 = PatchEmbed(patch_size=2, in_ch=embed_dims[0], out_ch=embed_dims[1], with_pos=True)
        self.patch_embed_3 = PatchEmbed(patch_size=2, in_ch=embed_dims[1], out_ch=embed_dims[2], with_pos=True)
        self.patch_embed_4 = PatchEmbed(patch_size=2, in_ch=embed_dims[2], out_ch=embed_dims[3], with_pos=True)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], mlp_ratios[0], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[0], apply_transform=apply_transform)
            for i in range(self.depths[0])])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], mlp_ratios[1], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[1], apply_transform=apply_transform)
            for i in range(self.depths[1])])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], mlp_ratios[2], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[2], apply_transform=apply_transform)
            for i in range(self.depths[2])])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], mlp_ratios[3], qkv_bias, qk_scale, drop_rate, attn_drop_rate,
                  drop_path=dpr[cur+i], norm_layer=norm_layer, sr_ratio=sr_ratios[3], apply_transform=apply_transform)
            for i in range(self.depths[3])])

        # add a norm layer for each output
        for i in range(self.num_layers-1):
            stage = f'stage{i+1}'
            if stage in self._out_features:
                layer = norm_layer(embed_dims[i])
                layer_name = f'norm{i+1}'
                self.add_module(layer_name, layer)

        self.norm = norm_layer(embed_dims[3])

        # init weights
        self.apply(self._init_weights)
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_AT)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index >= 0:
                # stage 0 is the stem
                self.stem.eval()
                for p in self.stem.parameters():
                    p.requires_grad = False
            if stage_index == 1:
                for p in self.stage1.parameters():
                    p.requires_grad = False
            else:
                patch = getattr(self, "patch_embed_" + str(stage_index))
                stage = getattr(self, "stage" + str(stage_index))
                for p in patch.parameters():
                    p.requires_grad = False
                for p in stage.parameters():
                    p.requires_grad = False

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        B, _, H, W = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        # stage 1
        for blk in self.stage1:
            x = blk(x, H, W)
        if "stage1" in self._out_features:
            x = self.norm1(x)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            outputs["stage1"] = x
        else:
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 2
        x, (H, W) = self.patch_embed_2(x)
        for blk in self.stage2:
            x = blk(x, H, W)
        if "stage2" in self._out_features:
            x = self.norm2(x)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            outputs["stage2"] = x
        else:
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 3
        x, (H, W) = self.patch_embed_3(x)
        for blk in self.stage3:
            x = blk(x, H, W)
        if "stage3" in self._out_features:
            x = self.norm3(x)
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)
            outputs["stage3"] = x
        else:
            x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # stage 4
        x, (H, W) = self.patch_embed_4(x)
        for blk in self.stage4:
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)
        if "stage4" in self._out_features:
            outputs["stage4"] = x

        return outputs


@BACKBONE_REGISTRY.register()
def build_rest_backbone(cfg, input_shape):
    """
    Create a ResT instance from config.
    Returns:
        ResT: a :class:`ResT` instance.
    """
    name = cfg.MODEL.REST.NAME
    out_features = cfg.MODEL.REST.OUT_FEATURES

    depths = {"rest_lite": [2, 2, 2, 2], "rest_small": [2, 2, 6, 2],
              "rest_base": [2, 2, 6, 2], "rest_large": [2, 2, 18, 2]}[name]

    embed_dims = {"rest_lite": [64, 128, 256, 512], "rest_small": [64, 128, 256, 512],
                  "rest_base": [96, 192, 384, 768], "rest_large": [96, 192, 384, 768]}[name]

    drop_path_rate = {"rest_lite": 0.1, "rest_small": 0.1, "rest_base": 0.2, "rest_large": 0.2}[name]

    feature_names = ['stage1', 'stage2', 'stage3', 'stage4']
    out_feature_channels = dict(zip(feature_names, embed_dims))
    out_feature_strides = {"stage1": 4, "stage2": 8, "stage3": 16, "stage4": 32}

    model = ResT(cfg, in_ch=3, embed_dims=embed_dims, qkv_bias=True, drop_path_rate=drop_path_rate,
                 depths=depths, apply_transform=True, out_features=out_features)
    model._out_feature_channels = out_feature_channels
    model._out_feature_strides = out_feature_strides
    return model


@BACKBONE_REGISTRY.register()
def build_rest_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode
    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_rest_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


@BACKBONE_REGISTRY.register()
def build_retinanet_rest_fpn_backbone(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_rest_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = bottom_up.output_shape()["stage4"].channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels, in_feature="stage4"),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone