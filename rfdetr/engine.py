# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable
import torch

import rfdetr.util.misc as utils
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.util.prefetcher import maybe_cuda_prefetcher, CUDAPrefetcher

try:
    from torch.amp import autocast, GradScaler
    DEPRECATED_AMP = False
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    DEPRECATED_AMP = True
from typing import DefaultDict, List, Callable
import numpy as np

def get_autocast_args(args):
    if DEPRECATED_AMP:
        return {'enabled': args.amp, 'dtype': torch.bfloat16}
    else:
        return {'device_type': 'cuda', 'enabled': args.amp, 'dtype': torch.bfloat16}


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    batch_size: int,
    max_norm: float = 0,
    ema_m: torch.nn.Module = None,
    schedules: dict = {},
    num_training_steps_per_epoch=None,
    vit_encoder_num_layers=None,
    args=None,
    callbacks: DefaultDict[str, List[Callable]] = None,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20
    start_steps = epoch * num_training_steps_per_epoch

    prefetch_enabled = getattr(args, "cuda_prefetcher", True)
    data_loader = maybe_cuda_prefetcher(
        data_loader,
        device=device,
        enabled=prefetch_enabled,
        prefetch_batches=None,
        warmup_batches=2,
        max_prefetch_mem_frac=0.6,
    )
    use_prefetcher = isinstance(data_loader, CUDAPrefetcher)

    effective_batch_size = batch_size * args.grad_accum_steps
    print("Grad accum steps: ", args.grad_accum_steps)
    print("Total batch size: ", effective_batch_size * utils.get_world_size())

    # Add gradient scaler for AMP
    if DEPRECATED_AMP:
        scaler = GradScaler(enabled=args.amp)
    else:
        scaler = GradScaler('cuda', enabled=args.amp)

    if args.multi_scale or args.do_random_resize_via_padding:
        raise ValueError("Resize/augmentation disabled; images must be fixed size.")

    optimizer.zero_grad()
    print("LENGTH OF DATA LOADER:", len(data_loader))
    update_step = start_steps
    start_micro_steps = start_steps * args.grad_accum_steps
    loss_dict_accum = None
    accum_steps = 0
    max_micro_steps = (
        num_training_steps_per_epoch * args.grad_accum_steps
        if num_training_steps_per_epoch is not None
        else None
    )

    enable_cuda_graph = getattr(args, "cuda_graph", True) and device.type == "cuda"
    cuda_graph = None
    static_samples = None
    static_targets = None
    static_outputs = None
    graph_signature = None
    graph_primed = False
    cudagraph_mark_step_begin = None
    if args is not None and getattr(args, "inductor_cudagraphs", False):
        cudagraph_mark_step_begin = getattr(
            getattr(torch, "compiler", None),
            "cudagraph_mark_step_begin",
            None,
        )

    def _graph_signature(samples, targets):
        if hasattr(samples, "tensors"):
            sample_shape = tuple(samples.tensors.shape)
            mask_shape = tuple(samples.mask.shape) if samples.mask is not None else None
            sample_dtype = samples.tensors.dtype
        else:
            sample_shape = tuple(samples.shape)
            mask_shape = None
            sample_dtype = samples.dtype
        target_sig = tuple(
            (k, tuple(targets[k].shape), targets[k].dtype)
            for k in sorted(targets.keys())
        )
        return (sample_shape, mask_shape, sample_dtype, target_sig)

    def _clone_samples(samples):
        if hasattr(samples, "tensors"):
            mask = samples.mask
            return utils.NestedTensor(
                samples.tensors.clone(),
                mask.clone() if mask is not None else None,
            )
        return samples.clone()

    def _copy_samples(dst, src):
        if hasattr(dst, "tensors"):
            dst.tensors.copy_(src.tensors)
            if dst.mask is not None and src.mask is not None:
                dst.mask.copy_(src.mask)
            return
        dst.copy_(src)

    def _clone_targets(targets):
        return {k: v.clone() for k, v in targets.items()}

    def _copy_targets(dst, src):
        for k, v in src.items():
            dst[k].copy_(v)

    for data_iter_step, (samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if max_micro_steps is not None and data_iter_step >= max_micro_steps:
            break
        micro_step = start_micro_steps + data_iter_step
        callback_dict = {
            "step": micro_step,
            "model": model,
            "epoch": epoch,
        }
        for callback in callbacks["on_train_batch_start"]:
            callback(callback_dict)
        if "dp" in schedules:
            if args.distributed:
                model.module.update_drop_path(
                    schedules["dp"][update_step], vit_encoder_num_layers
                )
            else:
                model.update_drop_path(schedules["dp"][update_step], vit_encoder_num_layers)
        if "do" in schedules:
            if args.distributed:
                model.module.update_dropout(schedules["do"][update_step])
            else:
                model.update_dropout(schedules["do"][update_step])

        if not use_prefetcher:
            samples = samples.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

        if cudagraph_mark_step_begin is not None:
            cudagraph_mark_step_begin()

        with autocast(**get_autocast_args(args)):
            use_cuda_graph = False
            # CUDA Graph: capture forward on first compatible shape, replay on static buffers.
            if enable_cuda_graph:
                current_signature = _graph_signature(samples, targets)
                if current_signature != graph_signature:
                    graph_signature = current_signature
                    cuda_graph = None
                    static_samples = None
                    static_targets = None
                    static_outputs = None
                    graph_primed = False

                if not graph_primed:
                    # Prime compilation and allocations outside capture.
                    outputs = model(samples, targets)
                    graph_primed = True
                else:
                    if cuda_graph is None:
                        static_samples = _clone_samples(samples)
                        static_targets = _clone_targets(targets)
                        torch.cuda.synchronize()
                        cuda_graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(cuda_graph):
                            static_outputs = model(static_samples, static_targets)
                    else:
                        _copy_samples(static_samples, samples)
                        _copy_targets(static_targets, targets)
                        cuda_graph.replay()
                    outputs = static_outputs
                    use_cuda_graph = True
            else:
                outputs = model(samples, targets)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(
                loss_dict[k] * weight_dict[k]
                for k in loss_dict.keys()
                if k in weight_dict
            ) / args.grad_accum_steps

        scaler.scale(losses).backward(retain_graph=use_cuda_graph)
        if loss_dict_accum is None:
            loss_dict_accum = {k: v.detach() for k, v in loss_dict.items()}
        else:
            for k, v in loss_dict.items():
                loss_dict_accum[k] += v.detach()

        accum_steps += 1
        if accum_steps < args.grad_accum_steps:
            continue

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(
            {k: v / accum_steps for k, v in loss_dict_accum.items()}
        )
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k:  v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print(loss_dict_reduced)
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        if max_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        scaler.step(optimizer)
        scaler.update()
        lr_scheduler.step()
        optimizer.zero_grad()
        if ema_m is not None:
            if epoch >= 0:
                ema_m.update(model)
        metric_logger.update(
            loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        update_step += 1
        accum_steps = 0
        loss_dict_accum = None

    if accum_steps:
        optimizer.zero_grad()
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.close()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def coco_extended_metrics(coco_eval):
    """
    Safe version: ignores the –1 sentinel entries so precision/F1 never explode.
    """

    iou_thrs, rec_thrs = coco_eval.params.iouThrs, coco_eval.params.recThrs
    iou50_idx, area_idx, maxdet_idx = (
        int(np.argwhere(np.isclose(iou_thrs, 0.50))), 0, 2)

    P = coco_eval.eval["precision"]
    S = coco_eval.eval["scores"]

    prec_raw = P[iou50_idx, :, :, area_idx, maxdet_idx]

    prec = prec_raw.copy().astype(float)
    prec[prec < 0] = np.nan

    f1_cls   = 2 * prec * rec_thrs[:, None] / (prec + rec_thrs[:, None])
    f1_macro = np.nanmean(f1_cls, axis=1)

    best_j   = int(f1_macro.argmax())

    macro_precision = float(np.nanmean(prec[best_j]))
    macro_recall    = float(rec_thrs[best_j])
    macro_f1        = float(f1_macro[best_j])

    score_vec = S[iou50_idx, best_j, :, area_idx, maxdet_idx].astype(float)
    score_vec[prec_raw[best_j] < 0] = np.nan
    score_thr = float(np.nanmean(score_vec))

    map_50_95, map_50 = float(coco_eval.stats[0]), float(coco_eval.stats[1])

    per_class = []
    cat_ids = coco_eval.params.catIds
    cat_id_to_name = {c["id"]: c["name"] for c in coco_eval.cocoGt.loadCats(cat_ids)}
    for k, cid in enumerate(cat_ids):
        p_slice = P[:, :, k, area_idx, maxdet_idx]
        valid   = p_slice > -1
        ap_50_95 = float(p_slice[valid].mean()) if valid.any() else float("nan")
        ap_50    = float(p_slice[iou50_idx][p_slice[iou50_idx] > -1].mean()) if (p_slice[iou50_idx] > -1).any() else float("nan")

        pc = float(prec[best_j, k]) if prec_raw[best_j, k] > -1 else float("nan")
        rc = macro_recall

        #Doing to this to filter out dataset class
        if np.isnan(ap_50_95) or np.isnan(ap_50) or np.isnan(pc) or np.isnan(rc):
            continue

        per_class.append({
            "class"      : cat_id_to_name[int(cid)],
            "map@50:95"  : ap_50_95,
            "map@50"     : ap_50,
            "precision"  : pc,
            "recall"     : rc,
        })

    per_class.append({
        "class"     : "all",
        "map@50:95" : map_50_95,
        "map@50"    : map_50,
        "precision" : macro_precision,
        "recall"    : macro_recall,
    })

    return {
        "class_map": per_class,
        "map"      : map_50,
        "precision": macro_precision,
        "recall"   : macro_recall
    }

def evaluate(model, criterion, postprocess, data_loader, base_ds, device, args=None):
    model.eval()
    if args.fp16_eval:
        model.half()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "class_error", utils.SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    header = "Test:"
    print_freq = 20

    iou_types = ("bbox",) if not args.segmentation_head else ("bbox", "segm")
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    prefetch_enabled = True if args is None else getattr(args, "cuda_prefetcher", True)
    data_loader = maybe_cuda_prefetcher(
        data_loader, device=device, enabled=prefetch_enabled, prefetch_batches=2
    )
    use_prefetcher = isinstance(data_loader, CUDAPrefetcher)

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        if not use_prefetcher:
            samples = samples.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

        if args.fp16_eval:
            samples.tensors = samples.tensors.half()

        # Add autocast for evaluation
        with autocast(**get_autocast_args(args)):
            outputs = model(samples)

        if args.fp16_eval:
            for key in outputs.keys():
                if key == "enc_outputs":
                    for sub_key in outputs[key].keys():
                        outputs[key][sub_key] = outputs[key][sub_key].float()
                elif key == "aux_outputs":
                    for idx in range(len(outputs[key])):
                        for sub_key in outputs[key][idx].keys():
                            outputs[key][idx][sub_key] = outputs[key][idx][
                                sub_key
                            ].float()
                else:
                    outputs[key] = outputs[key].float()

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items()
            if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f"{k}_unscaled": v for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **loss_dict_reduced_scaled,
            **loss_dict_reduced_unscaled,
        )
        metric_logger.update(class_error=loss_dict_reduced["class_error"])

        orig_target_sizes = targets["orig_size"]
        results_all = postprocess(outputs, orig_target_sizes)
        image_ids = targets["image_id"]
        if image_ids.ndim > 1:
            image_ids = image_ids.squeeze(-1)
        res = {image_id.item(): output for image_id, output in zip(image_ids, results_all)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    metric_logger.close()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        results_json = coco_extended_metrics(coco_evaluator.coco_eval["bbox"])
        stats["results_json"] = results_json
        if "bbox" in iou_types:
            stats["coco_eval_bbox"] = coco_evaluator.coco_eval["bbox"].stats.tolist()

        if "segm" in iou_types:
            results_json = coco_extended_metrics(coco_evaluator.coco_eval["segm"])
            stats["coco_eval_masks"] = coco_evaluator.coco_eval["segm"].stats.tolist()
    return stats, coco_evaluator
