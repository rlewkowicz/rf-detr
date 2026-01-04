#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

python - <<'PY'
import os
from pathlib import Path
from types import SimpleNamespace

import orjson
import torch
from torch.utils.data import DataLoader, Subset
import supervision as sv
from PIL import Image

import rfdetr.util.misc as utils
from rfdetr import RFDETRMedium
from rfdetr.datasets import build_dataset, get_coco_api_from_dataset
from rfdetr.datasets.coco_eval import CocoEvaluator
from rfdetr.engine import coco_extended_metrics
from rfdetr.models import PostProcess

NUM_IMAGES = 100
BATCH_SIZE = 4
SCORE_THRESHOLD = 0.5
OUTPUT_DIR = Path("output/smallval")
SAMPLES_DIR = OUTPUT_DIR / "samples"

ckpt_candidates = [
    Path("output/train_synth/checkpoint_best_total.pth"),
    Path("output/train_synth/checkpoint.pth"),
    Path("output/train_synth/checkpoint_best_regular.pth"),
    Path("output/train_synth/checkpoint_best_ema.pth"),
]
ckpt_path = next((p for p in ckpt_candidates if p.exists()), None)
if ckpt_path is None:
    raise FileNotFoundError(
        "No checkpoint found in output/train_synth. "
        "Expected checkpoint_best_total.pth or checkpoint.pth."
    )

checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
args = checkpoint.get("args") if isinstance(checkpoint, dict) else None
if args is None:
    args = SimpleNamespace()

def _set_default(name, value):
    if not hasattr(args, name):
        setattr(args, name, value)

def _write_json(path, obj, indent=False):
    option = orjson.OPT_NON_STR_KEYS | orjson.OPT_SERIALIZE_NUMPY
    if indent:
        option |= orjson.OPT_INDENT_2
    path.write_bytes(orjson.dumps(obj, option=option))

_set_default("dataset_file", "coco")
_set_default("coco_path", "synth")
_set_default("dataset_dir", "synth")
_set_default("resolution", 576)
_set_default("segmentation_head", False)
_set_default("multi_scale", False)
_set_default("expanded_scales", False)
_set_default("do_random_resize_via_padding", False)
_set_default("patch_size", 16)
_set_default("num_windows", 2)
_set_default("square_resize_div_64", False)
_set_default("cache_images", False)
_set_default("cache_dir", None)
_set_default("num_workers", max(0, (os.cpu_count() or 0) - 2))
_set_default("num_select", 300)
_set_default("device", "cuda" if torch.cuda.is_available() else "cpu")

class_names = None
if hasattr(args, "class_names") and args.class_names:
    class_names = list(args.class_names)
num_classes = getattr(args, "num_classes", None)
if num_classes is None:
    num_classes = len(class_names) if class_names else 8

model = RFDETRMedium(
    num_classes=num_classes,
    pretrain_weights=None,
    device=args.device,
    resolution=args.resolution,
    patch_size=args.patch_size,
    num_windows=args.num_windows,
    segmentation_head=args.segmentation_head,
)
missing, unexpected = model.model.model.load_state_dict(state_dict, strict=False)
if missing:
    print(f"Warning: missing keys in checkpoint: {len(missing)}")
if unexpected:
    print(f"Warning: unexpected keys in checkpoint: {len(unexpected)}")

device = torch.device(args.device)
model.model.model.to(device)
model.model.model.eval()
postprocess = PostProcess(num_select=args.num_select)

dataset_val = build_dataset(image_set="val", args=args, resolution=args.resolution)
subset_size = min(NUM_IMAGES, len(dataset_val))
subset = Subset(dataset_val, list(range(subset_size)))
data_loader = DataLoader(
    subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=args.num_workers,
    collate_fn=utils.collate_fn,
    pin_memory=True,
)

coco_gt = get_coco_api_from_dataset(subset)
coco_evaluator = CocoEvaluator(coco_gt, ("bbox",))

label_names = {}
if coco_gt is not None and getattr(coco_gt, "cats", None):
    label_names = {
        int(cat_id): cat_info.get("name", str(cat_id))
        for cat_id, cat_info in coco_gt.cats.items()
    }
elif class_names:
    label_names = {i + 1: name for i, name in enumerate(class_names)}
    label_names[0] = "__background__"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

means = torch.tensor([0.485, 0.456, 0.406])
stds = torch.tensor([0.229, 0.224, 0.225])

default_color = sv.Color.from_hex("808080")
class_colors = {
    "__background__": sv.Color.from_hex("808080"),
    "person": sv.Color.from_hex("FFD700"),
    "friendly": sv.Color.from_hex("0000FF"),
    "enemy": sv.Color.from_hex("FF0000"),
    "ads_sight_center": sv.Color.from_hex("800080"),
    "scope": sv.Color.from_hex("FFA500"),
    "sniper": sv.Color.from_hex("FFFFFF"),
    "center_dot": sv.Color.from_hex("00FF00"),
    "centerdot": sv.Color.from_hex("00FF00"),
}
max_label_id = max(label_names) if label_names else 0
palette_colors = [default_color for _ in range(max_label_id + 1)]
for class_id, name in label_names.items():
    if name in class_colors and class_id < len(palette_colors):
        palette_colors[class_id] = class_colors[name]

def text_color_for(bg_color):
    luminance = 0.299 * bg_color.r + 0.587 * bg_color.g + 0.114 * bg_color.b
    return sv.Color.from_hex("000000") if luminance > 160 else sv.Color.from_hex("FFFFFF")

color_palette = sv.ColorPalette(colors=palette_colors)
text_palette = sv.ColorPalette(colors=[text_color_for(c) for c in palette_colors])

box_annotator = sv.BoxAnnotator(color=color_palette, thickness=1)
label_annotator = sv.LabelAnnotator(
    color=color_palette,
    text_color=text_palette,
    text_scale=0.4,
    text_thickness=1,
    text_padding=4,
)

predictions = []
sample_index = 0

def xyxy_to_xywh(box):
    x0, y0, x1, y1 = box
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]

with torch.inference_mode():
    for samples, targets in data_loader:
        samples_cpu = samples
        targets_cpu = targets
        samples = samples.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        outputs = model.model.model(samples)
        if isinstance(outputs, tuple):
            outputs = {
                "pred_logits": outputs[1],
                "pred_boxes": outputs[0],
            }
            if len(outputs) == 3:
                outputs["pred_masks"] = outputs[2]

        orig_target_sizes = targets["orig_size"]
        results_all = postprocess(outputs, orig_target_sizes)

        image_ids = targets["image_id"]
        if image_ids.ndim > 1:
            image_ids = image_ids.squeeze(-1)
        res = {
            image_id.item(): output
            for image_id, output in zip(image_ids, results_all)
        }
        coco_evaluator.update(res)

        image_ids_cpu = targets_cpu["image_id"]
        if image_ids_cpu.ndim > 1:
            image_ids_cpu = image_ids_cpu.squeeze(-1)
        orig_sizes_cpu = targets_cpu["orig_size"]
        for idx, output in enumerate(results_all):
            image_id = image_ids_cpu[idx].item()
            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()

            for box, score, label in zip(boxes, scores, labels):
                predictions.append({
                    "image_id": int(image_id),
                    "category_id": int(label),
                    "bbox": xyxy_to_xywh(box.tolist()),
                    "score": float(score),
                })

            keep = scores >= SCORE_THRESHOLD
            if keep.any():
                keep_boxes = boxes[keep].numpy()
                keep_scores = scores[keep].numpy()
                keep_labels = labels[keep].numpy()
            else:
                keep_boxes = boxes[:0].numpy()
                keep_scores = scores[:0].numpy()
                keep_labels = labels[:0].numpy()

            detections = sv.Detections(
                xyxy=keep_boxes,
                confidence=keep_scores,
                class_id=keep_labels,
            )
            label_strings = [str(int(class_id)) for class_id in keep_labels]

            orig_size = orig_sizes_cpu[idx].tolist()
            h, w = int(orig_size[0]), int(orig_size[1])
            img = samples_cpu.tensors[idx][:, :h, :w].detach().cpu()
            img = (img * stds[:, None, None]) + means[:, None, None]
            img = img.clamp(0, 1)
            img = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")

            annotated = box_annotator.annotate(img.copy(), detections)
            annotated = label_annotator.annotate(annotated, detections, label_strings)

            sample_index += 1
            out_path = SAMPLES_DIR / f"{sample_index:04d}_image_{image_id}.jpg"
            Image.fromarray(annotated).save(out_path)

coco_evaluator.synchronize_between_processes()
coco_evaluator.accumulate()
coco_evaluator.summarize()

metrics = {
    "checkpoint": str(ckpt_path),
    "num_images": subset_size,
    "coco_eval_bbox": coco_evaluator.coco_eval["bbox"].stats.tolist(),
    "results_json": coco_extended_metrics(coco_evaluator.coco_eval["bbox"]),
}

_write_json(OUTPUT_DIR / "metrics.json", metrics, indent=True)
_write_json(OUTPUT_DIR / "predictions.json", predictions)

print(f"Saved metrics to {OUTPUT_DIR / 'metrics.json'}")
print(f"Saved predictions to {OUTPUT_DIR / 'predictions.json'}")
print(f"Saved samples to {SAMPLES_DIR}")
PY
