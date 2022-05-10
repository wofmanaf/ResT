# ------------------------------------------------------------
# Copyright (c) VCU, Nanjing University.
# Licensed under the Apache License 2.0 [see LICENSE for details]
# Written by Qing-Long Zhang
# ------------------------------------------------------------

from detectron2.config import CfgNode as CN


def add_restv2_config(cfg):
    # restv2 backbone
    cfg.MODEL.RESTV2 = CN()
    cfg.MODEL.RESTV2.NAME = "restv2_tiny"
    cfg.MODEL.RESTV2.OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]
    cfg.MODEL.RESTV2.WEIGHTS = None
    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    # addition
    cfg.MODEL.FPN.TOP_LEVELS = 2
    cfg.SOLVER.OPTIMIZER = "AdamW"


def add_restv1_config(cfg):
    # restv1 backbone
    cfg.MODEL.REST = CN()
    cfg.MODEL.REST.NAME = "rest_base"
    cfg.MODEL.REST.OUT_FEATURES = ["stage1", "stage2", "stage3", "stage4"]
    cfg.MODEL.REST.WEIGHTS = None
    cfg.MODEL.BACKBONE.FREEZE_AT = 2

    # addition
    cfg.MODEL.FPN.TOP_LEVELS = 2
    cfg.SOLVER.OPTIMIZER = "AdamW"
