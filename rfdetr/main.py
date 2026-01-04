# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
cleaned main file
"""
import argparse
import ast
import copy
import datetime
import inspect
import math
import os
import random
import shutil
import time
from copy import deepcopy
from logging import getLogger
from pathlib import Path
from typing import DefaultDict, List, Callable

try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass

import numpy as np
import torch
# Optimize for 3090 Ti
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, DistributedSampler

import rfdetr.util.misc as utils
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.engine import evaluate, train_one_epoch
from rfdetr.models import build_model, build_criterion_and_postprocessors, PostProcess
from rfdetr.util.benchmark import benchmark
from rfdetr.util.drop_scheduler import drop_scheduler
from rfdetr.util.files import download_file
from rfdetr.util.get_param_dicts import get_param_dict
from rfdetr.util.json_utils import dump_json, dumps_json, load_json
from rfdetr.util.utils import ModelEma, BestMetricHolder, clean_state_dict

if str(os.environ.get("USE_FILE_SYSTEM_SHARING", "False")).lower() in ["true", "1"]:
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

logger = getLogger(__name__)

DEFAULT_BATCH_SIZE = 3
DEFAULT_GRAD_ACCUM_STEPS = 42

def _get_default_workers():
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(0, len(os.sched_getaffinity(0)) - 2)
        except NotImplementedError:
            pass
    return max(0, (os.cpu_count() or 0) - 2)


DEFAULT_NUM_WORKERS = _get_default_workers()

HOSTED_MODELS = {
    "rf-detr-base.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-coco.pth",
    "rf-detr-base-o365.pth": "https://storage.googleapis.com/rfdetr/top-secret-1234/lwdetr_dinov2_small_o365_checkpoint.pth",
    # below is a less converged model that may be better for finetuning but worse for inference
    "rf-detr-base-2.pth": "https://storage.googleapis.com/rfdetr/rf-detr-base-2.pth",
    "rf-detr-large.pth": "https://storage.googleapis.com/rfdetr/rf-detr-large.pth",
    "rf-detr-nano.pth": "https://storage.googleapis.com/rfdetr/nano_coco/checkpoint_best_regular.pth",
    "rf-detr-small.pth": "https://storage.googleapis.com/rfdetr/small_coco/checkpoint_best_regular.pth",
    "rf-detr-medium.pth": "https://storage.googleapis.com/rfdetr/medium_coco/checkpoint_best_regular.pth",
    "rf-detr-seg-preview.pt": "https://storage.googleapis.com/rfdetr/rf-detr-seg-preview.pt",
}

def _resolve_inductor_cudagraphs(args, fallback=None):
    use_cudagraphs = getattr(args, "inductor_cudagraphs", None)
    if use_cudagraphs is None:
        use_cudagraphs = fallback
    if use_cudagraphs is None:
        use_cudagraphs = not getattr(args, "cuda_graph", False)
    return bool(use_cudagraphs)


def configure_inductor(args, fallback=None):
    use_cudagraphs = _resolve_inductor_cudagraphs(args, fallback=fallback)
    args.inductor_cudagraphs = use_cudagraphs
    if hasattr(torch, "_inductor") and hasattr(torch._inductor, "config"):
        triton_config = getattr(torch._inductor.config, "triton", None)
        if triton_config is not None and hasattr(triton_config, "cudagraphs"):
            triton_config.cudagraphs = use_cudagraphs
    if getattr(args, "cuda_graph", False) and use_cudagraphs:
        print(
            "Warning: both manual CUDA Graph capture and Inductor cudagraphs are enabled; "
            "disable one if you hit instability."
        )

def download_pretrain_weights(pretrain_weights: str, redownload=False):
    if pretrain_weights in HOSTED_MODELS:
        if redownload or not os.path.exists(pretrain_weights):
            logger.info(
                f"Downloading pretrained weights for {pretrain_weights}"
            )
            download_file(
                HOSTED_MODELS[pretrain_weights],
                pretrain_weights,
            )

class Model:
    def __init__(self, **kwargs):
        args = populate_args(**kwargs)
        self.args = args
        self.resolution = args.resolution
        self.model = build_model(args)
        self.device = torch.device(args.device)
        if args.pretrain_weights is not None and args.patch_size != 14:
            print(
                "warning: patch_size != 14; pretrained weights may not match "
                "the current backbone configuration."
            )
        if args.pretrain_weights is not None:
            print("Loading pretrain weights")
            try:
                checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Failed to load pretrain weights: {e}")
                # re-download weights if they are corrupted
                print("Failed to load pretrain weights, re-downloading")
                download_pretrain_weights(args.pretrain_weights, redownload=True)
                checkpoint = torch.load(args.pretrain_weights, map_location='cpu', weights_only=False)

            # Extract class_names from checkpoint if available
            if 'args' in checkpoint and hasattr(checkpoint['args'], 'class_names'):
                self.args.class_names = checkpoint['args'].class_names
                self.class_names = checkpoint['args'].class_names
                
            checkpoint_num_classes = checkpoint['model']['class_embed.bias'].shape[0]
            if checkpoint_num_classes != args.num_classes + 1:
                self.reinitialize_detection_head(checkpoint_num_classes)
            # add support to exclude_keys
            # e.g., when load object365 pretrain, do not load `class_embed.[weight, bias]`
            if args.pretrain_exclude_keys is not None:
                assert isinstance(args.pretrain_exclude_keys, list)
                for exclude_key in args.pretrain_exclude_keys:
                    checkpoint['model'].pop(exclude_key)
            if args.pretrain_keys_modify_to_load is not None:
                from rfdetr.util.obj365_to_coco_model import get_coco_pretrain_from_obj365
                assert isinstance(args.pretrain_keys_modify_to_load, list)
                for modify_key_to_load in args.pretrain_keys_modify_to_load:
                    try:
                        checkpoint['model'][modify_key_to_load] = get_coco_pretrain_from_obj365(
                            self.model.state_dict()[modify_key_to_load],
                            checkpoint['model'][modify_key_to_load]
                        )
                    except:
                        print(f"Failed to load {modify_key_to_load}, deleting from checkpoint")
                        checkpoint['model'].pop(modify_key_to_load)

            # we may want to resume training with a smaller number of groups for group detr
            num_desired_queries = args.num_queries * args.group_detr
            query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
            for name, state in checkpoint['model'].items():
                if any(name.endswith(x) for x in query_param_names):
                    checkpoint['model'][name] = state[:num_desired_queries]

            self.model.load_state_dict(checkpoint['model'], strict=False)

        if args.backbone_lora:
            print("Applying LORA to backbone")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=16,
                use_dora=True,
                target_modules=[
                    "q_proj", "v_proj", "k_proj",  # covers OWL-ViT
                    "qkv", # covers open_clip ie Siglip2
                    "query", "key", "value", "cls_token", "register_tokens", # covers Dinov2 with windowed attn
                ]
            )
            self.model.backbone[0].encoder = get_peft_model(self.model.backbone[0].encoder, lora_config)
        self.model = self.model.to(self.device, memory_format=torch.channels_last)
        configure_inductor(self.args)
        try:
            print("Compiling model...")
            self.model = torch.compile(self.model)
        except Exception as e:
            print(f"Could not compile model: {e}")
        self.postprocess = PostProcess(
            num_select=args.num_select,
            max_detections_per_class=args.max_detections_per_class,
        )
        self.stop_early = False
    
    def reinitialize_detection_head(self, num_classes):
        self.model.reinitialize_detection_head(num_classes)

    def request_early_stop(self):
        self.stop_early = True
        print("Early stopping requested, will complete current epoch and stop")

    def train(self, callbacks: DefaultDict[str, List[Callable]], **kwargs):
        currently_supported_callbacks = ["on_fit_epoch_end", "on_train_batch_start", "on_train_end"]
        for key in callbacks.keys():
            if key not in currently_supported_callbacks:
                raise ValueError(
                    f"Callback {key} is not currently supported, please file an issue if you need it!\n"
                    f"Currently supported callbacks: {currently_supported_callbacks}"
                )
        args = populate_args(**kwargs)
        if not hasattr(args, "patch_size"):
            args.patch_size = getattr(self.args, "patch_size", 16)
        if not hasattr(args, "num_windows"):
            args.num_windows = getattr(self.args, "num_windows", 4)
        configure_inductor(args, fallback=getattr(self.args, "inductor_cudagraphs", None))
        run_test = getattr(args, "run_test", True)
        if hasattr(args, "test"):
            run_test = args.test
        eval_with_ema = bool(getattr(args, "use_ema", False))
        args.run_test = run_test
        args.eval_ema = eval_with_ema

        if getattr(args, 'class_names') is not None:
            self.args.class_names = args.class_names
            self.args.num_classes = args.num_classes

        utils.init_distributed_mode(args)
        total_workers = _get_default_workers()
        if args.distributed:
            per_rank = max(1, total_workers // utils.get_world_size())
            args.num_workers = per_rank
        else:
            args.num_workers = total_workers
        print(f"Using {args.num_workers} workers")
        print("git:\n  {}\n".format(utils.get_sha()))
        print(args)
        device = torch.device(args.device)
        
        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        criterion, postprocess = build_criterion_and_postprocessors(args)
        model = self.model
        model.to(device)

        model_without_ddp = model
        if args.distributed:
            if args.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
            model_without_ddp = model.module

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        param_dicts = get_param_dict(args, model_without_ddp)

        param_dicts = [p for p in param_dicts if p['params'].requires_grad]

        adamw_kwargs = {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
        use_fused_adamw = False
        if device.type == "cuda":
            try:
                if "fused" in inspect.signature(torch.optim.AdamW).parameters:
                    adamw_kwargs["fused"] = True
                    use_fused_adamw = True
            except (TypeError, ValueError):
                pass
        optimizer = torch.optim.AdamW(param_dicts, **adamw_kwargs)
        if use_fused_adamw and utils.is_main_process():
            print("Using fused AdamW optimizer")
        # Choose the learning rate scheduler based on the new argument

        dataset_train = build_dataset(image_set='train', args=args, resolution=args.resolution)
        dataset_val = build_dataset(image_set='val', args=args, resolution=args.resolution)
        dataset_test = None
        if run_test:
            test_split = "test" if args.dataset_file == "roboflow" else "val"
            dataset_test = build_dataset(image_set=test_split, args=args, resolution=args.resolution)

        if getattr(args, "cache_images", False):
            cache_batch_size = max(1, args.batch_size)

            def _warm_cache(dataset, name):
                if hasattr(dataset, "ensure_cache"):
                    cache_dir = getattr(dataset, "cache_dir", None)
                    location = f" ({cache_dir})" if cache_dir else ""
                    print(f"Caching {name} images{location}...")
                    dataset.ensure_cache(
                        workers=args.num_workers,
                        batch_size=cache_batch_size,
                    )

            if utils.is_main_process():
                _warm_cache(dataset_train, "train")
                _warm_cache(dataset_val, "val")
                if dataset_test is not None and args.dataset_file == "roboflow":
                    _warm_cache(dataset_test, "test")
            if utils.is_dist_avail_and_initialized():
                torch.distributed.barrier()

        if getattr(args, "preload_shared", False):
            preload_workers = max(1, args.num_workers)

            def _preload_shared(dataset, name):
                if hasattr(dataset, "preload_shared"):
                    if utils.is_main_process():
                        print(f"Preloading {name} into shared RAM...")
                    dataset.preload_shared(workers=preload_workers)

            _preload_shared(dataset_train, "train")
            _preload_shared(dataset_val, "val")
            if dataset_test is not None and args.dataset_file == "roboflow":
                _preload_shared(dataset_test, "test")

        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            sampler_test = DistributedSampler(dataset_test, shuffle=False) if dataset_test is not None else None
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test) if dataset_test is not None else None

        effective_batch_size = args.batch_size * args.grad_accum_steps
        prefetch_factor = (
            max(1, 6 // max(1, args.batch_size)) if args.num_workers > 0 else None
        )
        min_batches = kwargs.get('min_batches', 5)
        if len(dataset_train) < effective_batch_size * min_batches:
            logger.info(
                f"Training with uniform sampler because dataset is too small: {len(dataset_train)} < {effective_batch_size * min_batches}"
            )
            sampler = torch.utils.data.RandomSampler(
                dataset_train,
                replacement=True,
                num_samples=args.batch_size * args.grad_accum_steps * min_batches,
            )
            data_loader_train = DataLoader(
                dataset_train,
                batch_size=args.batch_size,
                collate_fn=utils.collate_fn,
                num_workers=args.num_workers,
                sampler=sampler,
                prefetch_factor=prefetch_factor,
                persistent_workers=True if args.num_workers > 0 else False,
                pin_memory=True,
            )
        else:
            batch_sampler_train = torch.utils.data.BatchSampler(
                sampler_train, args.batch_size, drop_last=True)
            data_loader_train = DataLoader(
                dataset_train, 
                batch_sampler=batch_sampler_train,
                collate_fn=utils.collate_fn, 
                num_workers=args.num_workers,
                prefetch_factor=prefetch_factor,
                persistent_workers=True if args.num_workers > 0 else False,
                pin_memory=True,
            )
        
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False, collate_fn=utils.collate_fn, 
                                    num_workers=args.num_workers,
                                    prefetch_factor=prefetch_factor,
                                    persistent_workers=True if args.num_workers > 0 else False,
                                    pin_memory=True)
        data_loader_test = None
        if dataset_test is not None:
            data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                        drop_last=False, collate_fn=utils.collate_fn, 
                                        num_workers=args.num_workers,
                                        prefetch_factor=prefetch_factor,
                                        persistent_workers=True if args.num_workers > 0 else False,
                                        pin_memory=True)

        num_training_steps_per_epoch_lr = max(
            1, len(data_loader_train) // max(1, args.grad_accum_steps)
        )
        total_training_steps_lr = num_training_steps_per_epoch_lr * args.epochs
        warmup_steps_lr = int(num_training_steps_per_epoch_lr * args.warmup_epochs)
        def lr_lambda(current_step: int):
            if current_step < warmup_steps_lr:
                # Linear warmup
                return float(current_step) / float(max(1, warmup_steps_lr))
            else:
                # Cosine annealing from multiplier 1.0 down to lr_min_factor
                if args.lr_scheduler == 'cosine':
                    progress = float(current_step - warmup_steps_lr) / float(max(1, total_training_steps_lr - warmup_steps_lr))
                    return args.lr_min_factor + (1 - args.lr_min_factor) * 0.5 * (1 + math.cos(math.pi * progress))
                elif args.lr_scheduler == 'step':
                    if current_step < args.lr_drop * num_training_steps_per_epoch_lr:
                        return 1.0
                    else:
                        return 0.1
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        base_ds = get_coco_api_from_dataset(dataset_val)
        base_ds_test = get_coco_api_from_dataset(dataset_test) if dataset_test is not None else None
        if args.use_ema:
            self.ema_m = ModelEma(model_without_ddp, decay=args.ema_decay, tau=args.ema_tau)
        else:
            self.ema_m = None


        output_dir = Path(args.output_dir)
        
        if  utils.is_main_process():
            print("Get benchmark")
            if args.do_benchmark:
                benchmark_model = copy.deepcopy(model_without_ddp)
                bm = benchmark(benchmark_model.float(), dataset_val, output_dir)
                print(dumps_json(bm, indent=True))
                del benchmark_model
        
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
            model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
            if args.use_ema:
                if 'ema_model' in checkpoint:
                    self.ema_m.module.load_state_dict(clean_state_dict(checkpoint['ema_model']))
                else:
                    del self.ema_m
                    self.ema_m = ModelEma(model, decay=args.ema_decay, tau=args.ema_tau) 
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:                
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1

        if args.eval:
            eval_model = self.ema_m.module if eval_with_ema else model
            test_stats, coco_evaluator = evaluate(
                eval_model, criterion, postprocess, data_loader_val, base_ds, device, args)
            if args.output_dir:
                if not args.segmentation_head:
                    utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
                else:
                    utils.save_on_master(coco_evaluator.coco_eval["segm"].eval, output_dir / "eval.pth")
            return
        
        # for drop
        total_batch_size = effective_batch_size * utils.get_world_size()
        num_training_steps_per_epoch = max(
            1, len(data_loader_train) // max(1, args.grad_accum_steps)
        )
        schedules = {}
        if args.dropout > 0:
            schedules['do'] = drop_scheduler(
                args.dropout, args.epochs, num_training_steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule)
            print("Min DO = %.7f, Max DO = %.7f" % (min(schedules['do']), max(schedules['do'])))

        if args.drop_path > 0:
            schedules['dp'] = drop_scheduler(
                args.drop_path, args.epochs, num_training_steps_per_epoch,
                args.cutoff_epoch, args.drop_mode, args.drop_schedule)
            print("Min DP = %.7f, Max DP = %.7f" % (min(schedules['dp']), max(schedules['dp'])))
        print("Start training")
        start_time = time.time()
        best_map_holder = BestMetricHolder(use_ema=args.use_ema)
        best_map_5095 = 0
        best_map_50 = 0
        best_map_ema_5095 = 0
        best_map_ema_50 = 0
        for epoch in range(args.start_epoch, args.epochs):
            epoch_start_time = time.time()
            if args.distributed:
                sampler_train.set_epoch(epoch)

            model.train()
            criterion.train()
            train_stats = train_one_epoch(
                model, criterion, lr_scheduler, data_loader_train, optimizer, device, epoch,
                args.batch_size, args.clip_max_norm, ema_m=self.ema_m, schedules=schedules, 
                num_training_steps_per_epoch=num_training_steps_per_epoch,
                vit_encoder_num_layers=args.vit_encoder_num_layers, args=args, callbacks=callbacks)
            train_epoch_time = time.time() - epoch_start_time
            train_epoch_time_str = str(datetime.timedelta(seconds=int(train_epoch_time)))
            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every `checkpoint_interval` epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % args.checkpoint_interval == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    weights = {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }
                    if args.use_ema:
                        weights.update({
                            'ema_model': self.ema_m.module.state_dict(),
                        })
                    if not args.dont_save_weights:
                        # create checkpoint dir
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        utils.save_on_master(weights, checkpoint_path)

            with torch.inference_mode():
                eval_model = self.ema_m.module if eval_with_ema else model
                test_stats, coco_evaluator = evaluate(
                    eval_model, criterion, postprocess, data_loader_val, base_ds, device, args=args
                )
            if not args.segmentation_head:
                map_eval = test_stats["coco_eval_bbox"][0]
            else:
                map_eval = test_stats["coco_eval_masks"][0]
            _isbest = best_map_holder.update(map_eval, epoch, is_ema=eval_with_ema)
            if _isbest:
                if not args.segmentation_head:
                    map50 = test_stats["coco_eval_bbox"][1]
                else:
                    map50 = test_stats["coco_eval_masks"][1]
                if eval_with_ema:
                    best_map_ema_5095 = max(best_map_ema_5095, map_eval)
                    best_map_ema_50 = max(best_map_ema_50, map50)
                    checkpoint_path = output_dir / 'checkpoint_best_ema.pth'
                    if not args.dont_save_weights:
                        utils.save_on_master({
                            'model': self.ema_m.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)
                else:
                    best_map_5095 = max(best_map_5095, map_eval)
                    best_map_50 = max(best_map_50, map50)
                    checkpoint_path = output_dir / 'checkpoint_best_regular.pth'
                    if not args.dont_save_weights:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            'args': args,
                        }, checkpoint_path)
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
            if eval_with_ema:
                log_stats.update({f'ema_test_{k}': v for k,v in test_stats.items()})
            log_stats.update(best_map_holder.summary())
            
            # epoch parameters
            ep_paras = {
                    'epoch': epoch,
                    'n_parameters': n_parameters
                }
            log_stats.update(ep_paras)
            try:
                log_stats.update({'now_time': str(datetime.datetime.now())})
            except:
                pass
            log_stats['train_epoch_time'] = train_epoch_time_str
            epoch_time = time.time() - epoch_start_time
            epoch_time_str = str(datetime.timedelta(seconds=int(epoch_time)))
            log_stats['epoch_time'] = epoch_time_str
            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(dumps_json(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            if not args.segmentation_head:
                                torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                        output_dir / "eval" / name)
                            else:
                                torch.save(coco_evaluator.coco_eval["segm"].eval,
                                    output_dir / "eval" / name)

            
            for callback in callbacks["on_fit_epoch_end"]:
                callback(log_stats)

            if self.stop_early:
                print(f"Early stopping requested, stopping at epoch {epoch}")
                break

        best_is_ema = eval_with_ema
        
        if utils.is_main_process():
            if not args.dont_save_weights:
                best_total_path = output_dir / 'checkpoint_best_total.pth'
                candidates = []
                if best_is_ema:
                    candidates.append(output_dir / 'checkpoint_best_ema.pth')
                else:
                    candidates.append(output_dir / 'checkpoint_best_regular.pth')
                candidates.append(output_dir / 'checkpoint.pth')
                source_path = next((p for p in candidates if p.exists()), None)
                if source_path is not None:
                    shutil.copy2(source_path, best_total_path)
                    utils.strip_checkpoint(best_total_path)
                else:
                    print("Warning: No checkpoint file found to create checkpoint_best_total.pth")
        
            results = test_stats["results_json"]

            class_map = results["class_map"]
            results["class_map"] = {"valid": class_map}
            dump_json(output_dir / "results.json", results)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            print('Results saved to {}'.format(output_dir / "results.json"))
            
        
        if best_is_ema:
            self.model = self.ema_m.module
        self.model.eval()

        if run_test:
            best_state_dict = torch.load(output_dir / 'checkpoint_best_total.pth', map_location='cpu', weights_only=False)['model']
            model.load_state_dict(best_state_dict)
            model.eval()

            test_stats, _ = evaluate(
                model, criterion, postprocess, data_loader_test, base_ds_test, device, args=args
            )
            print(f"Test results: {test_stats}")
            results = load_json(output_dir / "results.json")
            test_metrics = test_stats["results_json"]["class_map"]
            results["class_map"]["test"] = test_metrics
            dump_json(output_dir / "results.json", results)

        for callback in callbacks["on_train_end"]:
            callback()
    
    def export(self, output_dir="output", infer_dir=None, simplify=False,  backbone_only=False, opset_version=17, verbose=True, force=False, shape=None, batch_size=1, **kwargs):
        """Export the trained model to ONNX format"""
        print(f"Exporting model to ONNX format")
        try:
            from rfdetr.deploy.export import export_onnx, onnx_simplify, make_infer_image
        except ImportError:
            print("It seems some dependencies for ONNX export are missing. Please run `pip install rfdetr[onnxexport]` and try again.")
            raise


        device = self.device
        model = deepcopy(self.model.to("cpu"))
        model.to(device)

        os.makedirs(output_dir, exist_ok=True)
        output_dir = Path(output_dir)
        if shape is None:
            shape = (self.resolution, self.resolution)
        else:
            if shape[0] % 14 != 0 or shape[1] % 14 != 0:
                raise ValueError("Shape must be divisible by 14")

        input_tensors = make_infer_image(infer_dir, shape, batch_size, device).to(device)
        input_names = ['input']
        output_names = ['features'] if backbone_only else ['dets', 'labels']
        dynamic_axes = None
        self.model.eval()
        with torch.no_grad():
            if backbone_only:
                features = model(input_tensors)
                print(f"PyTorch inference output shape: {features.shape}")
            elif self.args.segmentation_head:
                outputs = model(input_tensors)
                dets = outputs['pred_boxes']
                labels = outputs['pred_logits']
                masks = outputs['pred_masks']
                print(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}, Masks: {masks.shape}")
            else:
                outputs = model(input_tensors)
                dets = outputs['pred_boxes']
                labels = outputs['pred_logits']
                print(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
        model.cpu()
        input_tensors = input_tensors.cpu()

        # Export to ONNX
        output_file = export_onnx(
            output_dir=output_dir,
            model=model,
            input_names=input_names,
            input_tensors=input_tensors,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            backbone_only=backbone_only,
            verbose=verbose,
            opset_version=opset_version
        )
        
        print(f"Successfully exported ONNX model to: {output_file}")

        if simplify:
            sim_output_file = onnx_simplify(
                onnx_dir=output_file,
                input_names=input_names,
                input_tensors=input_tensors,
                force=force
            )
            print(f"Successfully simplified ONNX model to: {sim_output_file}")
        
        print("ONNX export completed successfully")
        self.model = self.model.to(device)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser('LWDETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    config = vars(args)  # Convert Namespace to dictionary
    
    if args.subcommand == 'distill':
        distill(**config)   
    elif args.subcommand is None:
        main(**config)
    elif args.subcommand == 'export_model':
        filter_keys = [
            "num_classes",
            "grad_accum_steps",
            "lr",
            "lr_encoder",
            "weight_decay",
            "epochs",
            "lr_drop",
            "clip_max_norm",
            "lr_vit_layer_decay",
            "lr_component_decay",
            "dropout",
            "drop_path",
            "drop_mode",
            "drop_schedule",
            "cutoff_epoch",
            "pretrained_encoder",
            "pretrain_weights",
            "pretrain_exclude_keys",
            "pretrain_keys_modify_to_load",
            "freeze_florence",
            "freeze_aimv2",
            "decoder_norm",
            "set_cost_class",
            "set_cost_bbox",
            "set_cost_giou",
            "cls_loss_coef",
            "bbox_loss_coef",
            "giou_loss_coef",
            "focal_alpha",
            "aux_loss",
            "sum_group_losses",
            "use_varifocal_loss",
            "use_position_supervised_loss",
            "ia_bce_loss",
            "dataset_file",
            "coco_path",
            "dataset_dir",
            "square_resize_div_64",
            "output_dir",
            "checkpoint_interval",
            "seed",
            "resume",
            "start_epoch",
            "eval",
            "use_ema",
            "ema_decay",
            "ema_tau",
            "num_workers",
            "device",
            "world_size",
            "dist_url",
            "sync_bn",
            "fp16_eval",
            "cuda_graph",
            "inductor_cudagraphs",
            "infer_dir",
            "verbose",
            "opset_version",
            "dry_run",
            "shape",
            "max_detections_per_class",
        ]
        for key in filter_keys:
            config.pop(key, None)  # Use pop with None to avoid KeyError
            
        from deploy.export import main as export_main
        if args.batch_size != 1:
            config['batch_size'] = 1
            print(f"Only batch_size 1 is supported for onnx export, \
                 but got batchsize = {args.batch_size}. batch_size is forcibly set to 1.")
        export_main(**config)

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--grad_accum_steps', default=DEFAULT_GRAD_ACCUM_STEPS, type=int)
    parser.add_argument('--amp', default=False, type=bool)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_encoder', default=1.5e-4, type=float)
    parser.add_argument('--batch_size', default=DEFAULT_BATCH_SIZE, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--lr_drop', default=11, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--lr_vit_layer_decay', default=0.8, type=float)
    parser.add_argument('--lr_component_decay', default=1.0, type=float)
    parser.add_argument('--do_benchmark', action='store_true', help='benchmark the model')

    # drop args 
    # dropout and stochastic depth drop rate; set at most one to non-zero
    parser.add_argument('--dropout', type=float, default=0,
                        help='Drop path rate (default: 0.0)')
    parser.add_argument('--drop_path', type=float, default=0,
                        help='Drop path rate (default: 0.0)')

    # early / late dropout and stochastic depth settings
    parser.add_argument('--drop_mode', type=str, default='standard',
                        choices=['standard', 'early', 'late'], help='drop mode')
    parser.add_argument('--drop_schedule', type=str, default='constant',
                        choices=['constant', 'linear'],
                        help='drop schedule for early dropout / s.d. only')
    parser.add_argument('--cutoff_epoch', type=int, default=0,
                        help='if drop_mode is early / late, this is the epoch where dropout ends / starts')

    # Model parameters
    parser.add_argument('--pretrained_encoder', type=str, default=None, 
                        help="Path to the pretrained encoder.")
    parser.add_argument('--pretrain_weights', type=str, default=None, 
                        help="Path to the pretrained model.")
    parser.add_argument('--pretrain_exclude_keys', type=str, default=None, nargs='+', 
                        help="Keys you do not want to load.")
    parser.add_argument('--pretrain_keys_modify_to_load', type=str, default=None, nargs='+',
                        help="Keys you want to modify to load. Only used when loading objects365 pre-trained weights.")

    # * Backbone
    parser.add_argument('--encoder', default='vit_tiny', type=str,
                        help="Name of the transformer or convolutional encoder to use")
    parser.add_argument('--vit_encoder_num_layers', default=12, type=int,
                        help="Number of layers used in ViT encoder")
    parser.add_argument('--window_block_indexes', default=None, type=int, nargs='+')
    parser.add_argument('--position_embedding', default='sine', type=str, 
                        choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--out_feature_indexes', default=[-1], type=int, nargs='+', help='only for vit now')
    parser.add_argument("--freeze_encoder", action="store_true", dest="freeze_encoder")
    parser.add_argument("--layer_norm", action="store_true", dest="layer_norm")
    parser.add_argument("--rms_norm", action="store_true", dest="rms_norm")
    parser.add_argument("--backbone_lora", action="store_true", dest="backbone_lora")
    parser.add_argument("--force_no_pretrain", action="store_true", dest="force_no_pretrain")

    # * Transformer
    parser.add_argument('--dec_layers', default=3, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--sa_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's self-attentions")
    parser.add_argument('--ca_nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's cross-attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--group_detr', default=13, type=int,
                        help="Number of groups to speed up detr training")
    parser.add_argument('--two_stage', action='store_true')
    parser.add_argument('--projector_scale', default='P4', type=str, nargs='+', choices=('P3', 'P4', 'P5', 'P6'))
    parser.add_argument('--lite_refpoint_refine', action='store_true', help='lite refpoint refine mode for speed-up')
    parser.add_argument('--num_select', default=100, type=int,
                        help='the number of predictions selected for evaluation')
    parser.add_argument(
        '--max_detections_per_class',
        default=20,
        type=int,
        help='max detections per class, per image (0 disables the cap)',
    )
    parser.add_argument('--dec_n_points', default=4, type=int,
                        help='the number of sampling points')
    parser.add_argument('--decoder_norm', default='LN', type=str)
    parser.add_argument('--bbox_reparam', action='store_true')
    parser.add_argument('--segmentation_head', action='store_true')
    parser.add_argument('--freeze_batch_norm', action='store_true')
    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--mask_ce_loss_coef', default=1.0, type=float)
    parser.add_argument('--mask_dice_loss_coef', default=1.0, type=float)
    parser.add_argument('--mask_point_sample_ratio', default=16, type=int)
    parser.add_argument('--mask_downsample_ratio', default=4, type=int)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--sum_group_losses', action='store_true',
                        help="To sum losses across groups or mean losses.")
    parser.add_argument('--use_varifocal_loss', action='store_true')
    parser.add_argument('--use_position_supervised_loss', action='store_true')
    parser.add_argument('--ia_bce_loss', action='store_true')

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--square_resize_div_64', action='store_true')
    parser.add_argument('--cache_images', dest='cache_images', action='store_true',
                        help='cache images as normalized numpy arrays on disk')
    parser.add_argument('--no-cache_images', dest='cache_images', action='store_false',
                        help='disable image caching')
    parser.set_defaults(cache_images=True)
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='cache directory (default: <dataset_root>/cache)')
    parser.add_argument('--preload_shared', action='store_true',
                        help='preload images/targets into shared RAM for dataloader workers')
    parser.add_argument(
        '--share_labels',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='share label data across ranks via shared memory when distributed',
    )

    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--dont_save_weights', action='store_true')
    parser.add_argument('--checkpoint_interval', default=10, type=int,
                        help='epoch interval to save checkpoint')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--use_ema', action='store_true')
    parser.add_argument('--ema_decay', default=0.9997, type=float)
    parser.add_argument('--ema_tau', default=0, type=float)

    parser.add_argument('--num_workers', default=DEFAULT_NUM_WORKERS, type=int)

    # distributed training parameters
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--sync_bn', default=True, type=bool,
                        help='setup synchronized BatchNorm for distributed training')
    
    # fp16
    parser.add_argument('--fp16_eval', default=False, action='store_true',
                        help='evaluate in fp16 precision.')
    parser.add_argument(
        '--cuda_graph',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable CUDA Graph capture for the model forward pass.',
    )
    parser.add_argument(
        '--inductor_cudagraphs',
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            'Enable TorchInductor CUDA graphs for torch.compile (default: auto; '
            'disabled when --cuda_graph is on).'
        ),
    )

    # custom args
    parser.add_argument('--encoder_only', action='store_true', help='Export and benchmark encoder only')
    parser.add_argument('--backbone_only', action='store_true', help='Export and benchmark backbone only')
    parser.add_argument('--resolution', type=int, default=576, help="input resolution")
    parser.add_argument('--use_cls_token', action='store_true', help='use cls token')
    parser.add_argument('--multi_scale', action='store_true', help='use multi scale')
    parser.add_argument('--expanded_scales', action='store_true', help='use expanded scales')
    parser.add_argument('--do_random_resize_via_padding', action='store_true', help='use random resize via padding')
    parser.add_argument('--warmup_epochs', default=1, type=float, 
        help='Number of warmup epochs for linear warmup before cosine annealing')
    # Add scheduler type argument: 'step' or 'cosine'
    parser.add_argument(
        '--lr_scheduler',
        default='step',
        choices=['step', 'cosine'],
        help="Type of learning rate scheduler to use: 'step' (default) or 'cosine'"
    )
    parser.add_argument('--lr_min_factor', default=0.0, type=float, 
        help='Minimum learning rate factor (as a fraction of initial lr) at the end of cosine annealing')
    # Early stopping parameters
    parser.add_argument('--early_stopping', action='store_true',
                        help='Enable early stopping based on mAP improvement')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Number of epochs with no improvement after which training will be stopped')
    parser.add_argument('--early_stopping_min_delta', default=0.001, type=float,
                        help='Minimum change in mAP to qualify as an improvement')
    parser.add_argument('--early_stopping_use_ema', action='store_true',
                        help='Use EMA model metrics for early stopping')
    # subparsers
    subparsers = parser.add_subparsers(title='sub-commands', dest='subcommand',
        description='valid subcommands', help='additional help')

    # subparser for export model
    parser_export = subparsers.add_parser('export_model', help='LWDETR model export')
    parser_export.add_argument('--infer_dir', type=str, default=None)
    parser_export.add_argument('--verbose', type=ast.literal_eval, default=False, nargs="?", const=True)
    parser_export.add_argument('--opset_version', type=int, default=17)
    parser_export.add_argument('--simplify', action='store_true', help="Simplify onnx model")
    parser_export.add_argument('--tensorrt', '--trtexec', '--trt', action='store_true',
                               help="build tensorrt engine")
    parser_export.add_argument('--dry-run', '--test', '-t', action='store_true', help="just print command")
    parser_export.add_argument('--profile', action='store_true', help='Run nsys profiling during TensorRT export')
    parser_export.add_argument('--shape', type=int, nargs=2, default=(576, 576), help="input shape (width, height)")
    return parser

def populate_args(
    # Basic training parameters
    num_classes=2,
    grad_accum_steps=DEFAULT_GRAD_ACCUM_STEPS,
    amp=False,
    lr=1e-4,
    lr_encoder=1.5e-4,
    batch_size=DEFAULT_BATCH_SIZE,
    weight_decay=1e-4,
    epochs=12,
    lr_drop=11,
    clip_max_norm=0.1,
    lr_vit_layer_decay=0.8,
    lr_component_decay=1.0,
    do_benchmark=False,
    
    # Drop parameters
    dropout=0,
    drop_path=0,
    drop_mode='standard',
    drop_schedule='constant',
    cutoff_epoch=0,
    
    # Model parameters
    pretrained_encoder=None,
    pretrain_weights=None, 
    pretrain_exclude_keys=None,
    pretrain_keys_modify_to_load=None,
    pretrained_distiller=None,
    
    # Backbone parameters
    encoder='vit_tiny',
    vit_encoder_num_layers=12,
    window_block_indexes=None,
    position_embedding='sine',
    out_feature_indexes=[-1],
    freeze_encoder=False,
    layer_norm=False,
    rms_norm=False,
    backbone_lora=False,
    force_no_pretrain=False,
    
    # Transformer parameters
    dec_layers=3,
    dim_feedforward=2048,
    hidden_dim=256,
    sa_nheads=8,
    ca_nheads=8,
    num_queries=300,
    group_detr=13,
    two_stage=False,
    projector_scale='P4',
    lite_refpoint_refine=False,
    num_select=100,
    max_detections_per_class=20,
    dec_n_points=4,
    decoder_norm='LN',
    bbox_reparam=False,
    segmentation_head=False,
    freeze_batch_norm=False,
    
    # Matcher parameters
    set_cost_class=2,
    set_cost_bbox=5,
    set_cost_giou=2,
    
    # Loss coefficients
    cls_loss_coef=2,
    bbox_loss_coef=5,
    giou_loss_coef=2,
    mask_ce_loss_coef=1.0,
    mask_dice_loss_coef=1.0,
    mask_point_sample_ratio=16,
    mask_downsample_ratio=4,
    focal_alpha=0.25,
    aux_loss=True,
    sum_group_losses=False,
    use_varifocal_loss=False,
    use_position_supervised_loss=False,
    ia_bce_loss=False,
    
    # Dataset parameters
    dataset_file='coco',
    coco_path=None,
    dataset_dir=None,
    square_resize_div_64=False,
    cache_images=True,
    cache_dir=None,
    preload_shared=False,
    share_labels=True,
    
    # Output parameters
    output_dir='output',
    dont_save_weights=False,
    checkpoint_interval=10,
    seed=42,
    resume='',
    start_epoch=0,
    eval=False,
    use_ema=False,
    ema_decay=0.9997,
    ema_tau=0,
    num_workers=DEFAULT_NUM_WORKERS,
    
    # Distributed training parameters
    device='cuda',
    world_size=1,
    dist_url='env://',
    sync_bn=True,
    
    # FP16
    fp16_eval=False,
    cuda_graph=True,
    inductor_cudagraphs=None,
    
    # Custom args
    encoder_only=False,
    backbone_only=False,
    resolution=576,
    use_cls_token=False,
    multi_scale=False,
    expanded_scales=False,
    do_random_resize_via_padding=False,
    warmup_epochs=1,
    lr_scheduler='step',
    lr_min_factor=0.0,
    # Early stopping parameters
    early_stopping=True,
    early_stopping_patience=10,
    early_stopping_min_delta=0.001,
    early_stopping_use_ema=False,
    gradient_checkpointing=False,
    # Additional
    subcommand=None,
    **extra_kwargs  # To handle any unexpected arguments
):
    args = argparse.Namespace(
        num_classes=num_classes,
        grad_accum_steps=grad_accum_steps,
        amp=amp,
        lr=lr,
        lr_encoder=lr_encoder,
        batch_size=batch_size,
        weight_decay=weight_decay,
        epochs=epochs,
        lr_drop=lr_drop,
        clip_max_norm=clip_max_norm,
        lr_vit_layer_decay=lr_vit_layer_decay,
        lr_component_decay=lr_component_decay,
        do_benchmark=do_benchmark,
        dropout=dropout,
        drop_path=drop_path,
        drop_mode=drop_mode,
        drop_schedule=drop_schedule,
        cutoff_epoch=cutoff_epoch,
        pretrained_encoder=pretrained_encoder,
        pretrain_weights=pretrain_weights,
        pretrain_exclude_keys=pretrain_exclude_keys,
        pretrain_keys_modify_to_load=pretrain_keys_modify_to_load,
        pretrained_distiller=pretrained_distiller,
        encoder=encoder,
        vit_encoder_num_layers=vit_encoder_num_layers,
        window_block_indexes=window_block_indexes,
        position_embedding=position_embedding,
        out_feature_indexes=out_feature_indexes,
        freeze_encoder=freeze_encoder,
        layer_norm=layer_norm,
        rms_norm=rms_norm,
        backbone_lora=backbone_lora,
        force_no_pretrain=force_no_pretrain,
        dec_layers=dec_layers,
        dim_feedforward=dim_feedforward,
        hidden_dim=hidden_dim,
        sa_nheads=sa_nheads,
        ca_nheads=ca_nheads,
        num_queries=num_queries,
        group_detr=group_detr,
        two_stage=two_stage,
        projector_scale=projector_scale,
        lite_refpoint_refine=lite_refpoint_refine,
        num_select=num_select,
        max_detections_per_class=max_detections_per_class,
        dec_n_points=dec_n_points,
        decoder_norm=decoder_norm,
        bbox_reparam=bbox_reparam,
        segmentation_head=segmentation_head,
        freeze_batch_norm=freeze_batch_norm,
        set_cost_class=set_cost_class,
        set_cost_bbox=set_cost_bbox,
        set_cost_giou=set_cost_giou,
        cls_loss_coef=cls_loss_coef,
        bbox_loss_coef=bbox_loss_coef,
        giou_loss_coef=giou_loss_coef,
        mask_ce_loss_coef=mask_ce_loss_coef,
        mask_dice_loss_coef=mask_dice_loss_coef,
        mask_point_sample_ratio=mask_point_sample_ratio,
        mask_downsample_ratio=mask_downsample_ratio,
        focal_alpha=focal_alpha,
        aux_loss=aux_loss,
        sum_group_losses=sum_group_losses,
        use_varifocal_loss=use_varifocal_loss,
        use_position_supervised_loss=use_position_supervised_loss,
        ia_bce_loss=ia_bce_loss,
        dataset_file=dataset_file,
        coco_path=coco_path,
        dataset_dir=dataset_dir,
        square_resize_div_64=square_resize_div_64,
        cache_images=cache_images,
        cache_dir=cache_dir,
        preload_shared=preload_shared,
        share_labels=share_labels,
        output_dir=output_dir,
        dont_save_weights=dont_save_weights,
        checkpoint_interval=checkpoint_interval,
        seed=seed,
        resume=resume,
        start_epoch=start_epoch,
        eval=eval,
        use_ema=use_ema,
        ema_decay=ema_decay,
        ema_tau=ema_tau,
        num_workers=num_workers,
        device=device,
        world_size=world_size,
        dist_url=dist_url,
        sync_bn=sync_bn,
        fp16_eval=fp16_eval,
        cuda_graph=cuda_graph,
        inductor_cudagraphs=inductor_cudagraphs,
        encoder_only=encoder_only,
        backbone_only=backbone_only,
        resolution=resolution,
        use_cls_token=use_cls_token,
        multi_scale=multi_scale,
        expanded_scales=expanded_scales,
        do_random_resize_via_padding=do_random_resize_via_padding,
        warmup_epochs=warmup_epochs,
        lr_scheduler=lr_scheduler,
        lr_min_factor=lr_min_factor,
        early_stopping=early_stopping,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_use_ema=early_stopping_use_ema,
        gradient_checkpointing=gradient_checkpointing,
        **extra_kwargs
    )
    return args
