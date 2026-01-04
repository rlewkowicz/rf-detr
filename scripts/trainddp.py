#!/usr/bin/env python3
import os
from collections import defaultdict

import torch

from rfdetr import RFDETRMedium


CLASS_NAMES = [
    "person",
    "friendly",
    "enemy",
    "ads_sight_center",
    "iron_sight",
    "scope",
    "sniper",
    "center_dot",
]


def _get_default_workers() -> int:
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(0, len(os.sched_getaffinity(0)) - 2)
        except NotImplementedError:
            pass
    return max(0, (os.cpu_count() or 0) - 2)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def main() -> None:
    num_workers = _get_default_workers()
    share_labels = _env_flag("SHARE_LABELS", True)

    model = RFDETRMedium()
    # Initialize with N+1 for background class
    model.model.reinitialize_detection_head(len(CLASS_NAMES) + 1)

    resume_path = os.path.join("output", "train_synth", "checkpoint.pth")
    resume_arg = resume_path if os.path.exists(resume_path) else ""
    if resume_arg and int(os.environ.get("RANK", "0")) == 0:
        print(f"Resuming from {resume_arg}")

    callbacks = defaultdict(list)

    model.model.train(
        cuda_graph=False,
        inductor_cudagraphs=False,
        callbacks=callbacks,
        dataset_file="coco",
        coco_path="synth",
        num_classes=len(CLASS_NAMES),
        class_names=CLASS_NAMES,
        output_dir="output/train_synth",
        batch_size=3,
        grad_accum_steps=1,
        device="cuda",
        amp=False,
        num_workers=num_workers,
        share_labels=share_labels,
        preload_shared=False,
        segmentation_head=False,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        run_test=False,
        use_ema=True,
        epochs=200,
        resume=resume_arg,
    )


if __name__ == "__main__":
    main()
