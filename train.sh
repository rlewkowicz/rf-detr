#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import os
import torch
from rfdetr import RFDETRMedium
from collections import defaultdict

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

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
num_workers = max(0, (os.cpu_count() or 0) - 2)

model = RFDETRMedium()
# Initialize with N+1 for background class
model.model.reinitialize_detection_head(len(CLASS_NAMES) + 1)

resume_path = os.path.join("output", "train_synth", "checkpoint.pth")
resume_arg = resume_path if os.path.exists(resume_path) else ""
if resume_arg:
    print(f"Resuming from {resume_arg}")

# Create required callbacks dictionary
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
    segmentation_head=False,
    multi_scale=False,
    expanded_scales=False,
    do_random_resize_via_padding=False,
    run_test=False,
    use_ema=True,
    epochs=200,
    resume=resume_arg,
)
PY
