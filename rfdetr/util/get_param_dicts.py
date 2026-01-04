# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Functions to get params dict"""
import torch.nn as nn

from rfdetr.models.backbone import Joiner


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.
    
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    print("name: {}, lr_decay: {}".format(name, lr_decay_rate ** (num_layers + 1 - layer_id)))
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name, weight_decay_rate=1.0):
    if ('gamma' in name) or ('pos_embed' in name) or ('rel_pos' in name) or ('bias' in name) or ('norm' in name):
        weight_decay_rate = 0.
    print("name: {}, weight_decay rate: {}".format(name, weight_decay_rate))
    return weight_decay_rate


def _unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap common wrappers (e.g., torch.compile) to keep parameter names stable."""
    unwrapped = model
    while True:
        if hasattr(unwrapped, "_orig_mod") and isinstance(unwrapped._orig_mod, nn.Module):
            if unwrapped._orig_mod is unwrapped:
                break
            unwrapped = unwrapped._orig_mod
            continue
        if hasattr(unwrapped, "module") and isinstance(unwrapped.module, nn.Module):
            if unwrapped.module is unwrapped:
                break
            unwrapped = unwrapped.module
            continue
        break
    return unwrapped


def get_param_dict(args, model_without_ddp: nn.Module):


    model = _unwrap_model(model_without_ddp)

    assert isinstance(model.backbone, Joiner)

    backbone = model.backbone[0]
    backbone_named_param_lr_pairs = backbone.get_named_param_lr_pairs(args, prefix="backbone.0")
    backbone_param_lr_pairs = [param_dict for _, param_dict in backbone_named_param_lr_pairs.items()]
    backbone_param_ids = {id(param_dict["params"]) for param_dict in backbone_param_lr_pairs}

    decoder_key = "transformer.decoder"
    segmentation_key = "segmentation_head"

    decoder_param_lr_pairs = []
    segmentation_param_lr_pairs = []
    other_param_dicts = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in backbone_param_ids:
            continue
        if decoder_key in name:
            decoder_param_lr_pairs.append(
                {"params": param, "lr": args.lr * args.lr_component_decay}
            )
        elif segmentation_key in name:
            segmentation_param_lr_pairs.append({"params": param, "lr": args.lr})
        else:
            other_param_dicts.append({"params": param, "lr": args.lr})

    final_param_dicts = (
        other_param_dicts + backbone_param_lr_pairs + decoder_param_lr_pairs + segmentation_param_lr_pairs
    )

    return final_param_dicts
