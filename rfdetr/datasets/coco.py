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
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path
import atexit
import hashlib
import mmap
import os
import tempfile
import time
from typing import Optional
import multiprocessing as mp_std
from multiprocessing import shared_memory

import numpy as np
import torch
import torch.utils.data
import torchvision
import torch.multiprocessing as torch_mp
import torch.distributed as dist
from torchvision.datasets import VisionDataset
import pycocotools.mask as coco_mask
from pycocotools.coco import COCO
from PIL import Image

import rfdetr.datasets.transforms as T
import rfdetr.util.misc as utils

try:
    import orjson
except ImportError:
    orjson = None


def _use_orjson():
    if orjson is None:
        raise RuntimeError("orjson is required for COCO loading.")
    return True


def _orjson_load_path(path):
    with open(path, "rb") as f:
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError):
            return orjson.loads(f.read())
        try:
            return orjson.loads(memoryview(mm))
        finally:
            mm.close()


def _cache_args(root: Path, args):
    cache_images = bool(getattr(args, "cache_images", False))
    cache_dir = getattr(args, "cache_dir", None)
    if cache_images:
        if cache_dir is None:
            cache_dir = root / "cache"
        else:
            cache_dir = Path(cache_dir)
            if not cache_dir.is_absolute():
                cache_dir = root / cache_dir
    else:
        cache_dir = None
    return cache_images, cache_dir


_SHARED_PRELOAD_DATASET = None


def _init_shared_preload(dataset):
    global _SHARED_PRELOAD_DATASET
    _SHARED_PRELOAD_DATASET = dataset


def _share_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"shared preload expects tensor values, got {type(tensor)}")
    if tensor.device.type != "cpu":
        raise ValueError("shared preload only supports CPU tensors")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    tensor.share_memory_()
    return tensor


def _share_target(target: dict) -> dict:
    shared = {}
    for key, value in target.items():
        if torch.is_tensor(value):
            shared[key] = _share_tensor(value)
        else:
            shared[key] = value
    return shared


def _preload_chunk(indices):
    dataset = _SHARED_PRELOAD_DATASET
    results = []
    for idx in indices:
        img, target = dataset._getitem_no_shared(idx)
        img = _share_tensor(img)
        target = _share_target(target)
        results.append((idx, img, target))
    return results


class OrjsonCOCO(COCO):
    def __init__(self, annotation_file=None):
        super().__init__(annotation_file=None)
        if annotation_file is None:
            return
        print("loading annotations into memory...")
        tic = time.time()
        dataset = _orjson_load_path(annotation_file)
        assert type(dataset) == dict, (
            "annotation file format {} not supported".format(type(dataset))
        )
        print("Done (t={:0.2f}s)".format(time.time() - tic))
        self.dataset = dataset
        self.createIndex()


_LABEL_BUILD_CONTEXT = {}


def _label_build_workers():
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)) - 2)
        except NotImplementedError:
            pass
    return max(1, (os.cpu_count() or 1) - 2)


def _init_label_builder(img_info_map, ann_by_img, include_masks):
    _LABEL_BUILD_CONTEXT["img_info_map"] = img_info_map
    _LABEL_BUILD_CONTEXT["ann_by_img"] = ann_by_img
    _LABEL_BUILD_CONTEXT["include_masks"] = include_masks


def _build_label_chunk(entries):
    img_info_map = _LABEL_BUILD_CONTEXT["img_info_map"]
    ann_by_img = _LABEL_BUILD_CONTEXT["ann_by_img"]
    include_masks = _LABEL_BUILD_CONTEXT["include_masks"]
    results = []
    for idx, img_id in entries:
        info = img_info_map.get(img_id, {})
        file_name = info.get("file_name") if isinstance(info, dict) else None
        file_name_bytes = file_name.encode("utf-8") if isinstance(file_name, str) else b""
        width = int(info.get("width", 0)) if isinstance(info, dict) else 0
        height = int(info.get("height", 0)) if isinstance(info, dict) else 0
        ann_list = ann_by_img.get(img_id, [])

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        ann_ids = []
        segms = [] if include_masks else None

        for ann_idx, ann in enumerate(ann_list):
            if not isinstance(ann, dict):
                continue
            bbox = ann.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            try:
                x, y, w, h = bbox
            except ValueError:
                continue
            x1 = float(x)
            y1 = float(y)
            x2 = float(x + w)
            y2 = float(y + h)
            if width > 0:
                x1 = max(0.0, min(x1, float(width)))
                x2 = max(0.0, min(x2, float(width)))
            if height > 0:
                y1 = max(0.0, min(y1, float(height)))
                y2 = max(0.0, min(y2, float(height)))
            if x2 <= x1 or y2 <= y1:
                continue
            ann_id = ann.get("id")
            if ann_id is None:
                ann_id = (int(img_id) << 32) | ann_idx
            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann.get("category_id", 0)))
            area_val = ann.get("area")
            if area_val is None:
                area_val = (x2 - x1) * (y2 - y1)
            areas.append(float(area_val))
            iscrowd.append(int(ann.get("iscrowd", 0)))
            ann_ids.append(int(ann_id))
            if include_masks:
                segms.append(ann.get("segmentation", []))

        boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels_arr = np.asarray(labels, dtype=np.int64)
        areas_arr = np.asarray(areas, dtype=np.float32)
        iscrowd_arr = np.asarray(iscrowd, dtype=np.int64)
        ann_ids_arr = np.asarray(ann_ids, dtype=np.int64)
        segm_bytes = None
        if include_masks:
            segm_bytes = orjson.dumps(segms or [])
        results.append(
            (
                idx,
                file_name_bytes,
                width,
                height,
                boxes_arr,
                labels_arr,
                areas_arr,
                iscrowd_arr,
                ann_ids_arr,
                segm_bytes,
            )
        )
    return results


def _chunk_entries(entries, chunk_size):
    return [entries[i:i + chunk_size] for i in range(0, len(entries), chunk_size)]


def _create_shared_array(array: np.ndarray):
    size = max(1, array.nbytes)
    shm = shared_memory.SharedMemory(create=True, size=size)
    shared = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shared[...] = array
    return shared, shm


class SharedCocoLabelStore:
    def __init__(self, meta, arrays, shms, owner=False):
        self._meta = meta
        self._arrays = arrays
        self._shms = shms
        self._owner = owner
        self._id_to_index = None
        if owner:
            atexit.register(self._cleanup_owner)

    def __del__(self):
        if self._owner:
            # relying on atexit for owner cleanup to ensure persistence during training
            return
        try:
            self.close()
        except Exception:
            pass

    @classmethod
    def build(cls, ann_file: Path, include_masks: bool, owner: bool = True):
        _use_orjson()
        dataset = _orjson_load_path(str(ann_file))
        if not isinstance(dataset, dict):
            raise ValueError("annotation file did not produce a dict")
        images = dataset.get("images", [])
        annotations = dataset.get("annotations", [])

        img_info_map = {}
        for info in images:
            if not isinstance(info, dict):
                continue
            img_id = info.get("id")
            if img_id is None:
                continue
            img_info_map[int(img_id)] = info

        ann_by_img = {}
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            img_id = ann.get("image_id")
            if img_id is None:
                continue
            img_id = int(img_id)
            ann_by_img.setdefault(img_id, []).append(ann)

        ids = sorted(img_info_map.keys())
        entries = [(idx, img_id) for idx, img_id in enumerate(ids)]
        num_images = len(entries)

        file_name_bytes = [b""] * num_images
        widths = [0] * num_images
        heights = [0] * num_images
        boxes_list = [np.zeros((0, 4), dtype=np.float32)] * num_images
        labels_list = [np.zeros((0,), dtype=np.int64)] * num_images
        areas_list = [np.zeros((0,), dtype=np.float32)] * num_images
        iscrowd_list = [np.zeros((0,), dtype=np.int64)] * num_images
        ann_ids_list = [np.zeros((0,), dtype=np.int64)] * num_images
        segm_list = [b""] * num_images if include_masks else None

        workers = _label_build_workers()
        if num_images == 0:
            workers = 1
        chunk_size = max(1, num_images // max(1, workers * 4))
        chunks = _chunk_entries(entries, chunk_size)

        if workers == 1:
            _init_label_builder(img_info_map, ann_by_img, include_masks)
            results = _build_label_chunk(entries)
            for (
                idx,
                name_bytes,
                width,
                height,
                boxes_arr,
                labels_arr,
                areas_arr,
                iscrowd_arr,
                ann_ids_arr,
                segm_bytes,
            ) in results:
                file_name_bytes[idx] = name_bytes
                widths[idx] = width
                heights[idx] = height
                boxes_list[idx] = boxes_arr
                labels_list[idx] = labels_arr
                areas_list[idx] = areas_arr
                iscrowd_list[idx] = iscrowd_arr
                ann_ids_list[idx] = ann_ids_arr
                if include_masks:
                    segm_list[idx] = segm_bytes or b"[]"
        else:
            try:
                ctx = mp_std.get_context("fork")
            except ValueError:
                ctx = mp_std.get_context()
            with ctx.Pool(
                processes=workers,
                initializer=_init_label_builder,
                initargs=(img_info_map, ann_by_img, include_masks),
            ) as pool:
                for results in pool.imap_unordered(_build_label_chunk, chunks, chunksize=1):
                    for (
                        idx,
                        name_bytes,
                        width,
                        height,
                        boxes_arr,
                        labels_arr,
                        areas_arr,
                        iscrowd_arr,
                        ann_ids_arr,
                        segm_bytes,
                    ) in results:
                        file_name_bytes[idx] = name_bytes
                        widths[idx] = width
                        heights[idx] = height
                        boxes_list[idx] = boxes_arr
                        labels_list[idx] = labels_arr
                        areas_list[idx] = areas_arr
                        iscrowd_list[idx] = iscrowd_arr
                        ann_ids_list[idx] = ann_ids_arr
                        if include_masks:
                            segm_list[idx] = segm_bytes or b"[]"

        ids_arr = np.asarray(ids, dtype=np.int64)
        widths_arr = np.asarray(widths, dtype=np.int32)
        heights_arr = np.asarray(heights, dtype=np.int32)

        box_offsets = np.zeros((num_images + 1,), dtype=np.int64)
        total_boxes = 0
        for i, boxes in enumerate(boxes_list):
            total_boxes += int(boxes.shape[0])
            box_offsets[i + 1] = total_boxes

        boxes_arr = np.zeros((total_boxes, 4), dtype=np.float32)
        labels_arr = np.zeros((total_boxes,), dtype=np.int64)
        areas_arr = np.zeros((total_boxes,), dtype=np.float32)
        iscrowd_arr = np.zeros((total_boxes,), dtype=np.int64)
        ann_ids_arr = np.zeros((total_boxes,), dtype=np.int64)
        cursor = 0
        for boxes, labels, areas, iscrowd, ann_ids in zip(
            boxes_list,
            labels_list,
            areas_list,
            iscrowd_list,
            ann_ids_list,
        ):
            count = int(boxes.shape[0])
            if count == 0:
                continue
            boxes_arr[cursor:cursor + count] = boxes
            labels_arr[cursor:cursor + count] = labels
            areas_arr[cursor:cursor + count] = areas
            iscrowd_arr[cursor:cursor + count] = iscrowd
            ann_ids_arr[cursor:cursor + count] = ann_ids
            cursor += count

        file_name_offsets = np.zeros((num_images + 1,), dtype=np.int64)
        total_name_bytes = 0
        for i, name in enumerate(file_name_bytes):
            total_name_bytes += len(name)
            file_name_offsets[i + 1] = total_name_bytes
        file_name_data = np.zeros((total_name_bytes,), dtype=np.uint8)
        cursor = 0
        for name in file_name_bytes:
            if not name:
                continue
            end = cursor + len(name)
            file_name_data[cursor:end] = np.frombuffer(name, dtype=np.uint8)
            cursor = end

        segm_offsets = None
        segm_data = None
        if include_masks:
            segm_offsets = np.zeros((num_images + 1,), dtype=np.int64)
            total_segm_bytes = 0
            for i, segm in enumerate(segm_list):
                total_segm_bytes += len(segm)
                segm_offsets[i + 1] = total_segm_bytes
            segm_data = np.zeros((total_segm_bytes,), dtype=np.uint8)
            cursor = 0
            for segm in segm_list:
                if not segm:
                    continue
                end = cursor + len(segm)
                segm_data[cursor:end] = np.frombuffer(segm, dtype=np.uint8)
                cursor = end

        arrays = {}
        shms = {}

        arrays["ids"], shms["ids"] = _create_shared_array(ids_arr)
        arrays["widths"], shms["widths"] = _create_shared_array(widths_arr)
        arrays["heights"], shms["heights"] = _create_shared_array(heights_arr)
        arrays["file_name_offsets"], shms["file_name_offsets"] = _create_shared_array(file_name_offsets)
        arrays["file_name_data"], shms["file_name_data"] = _create_shared_array(file_name_data)
        arrays["box_offsets"], shms["box_offsets"] = _create_shared_array(box_offsets)
        arrays["boxes"], shms["boxes"] = _create_shared_array(boxes_arr)
        arrays["labels"], shms["labels"] = _create_shared_array(labels_arr)
        arrays["areas"], shms["areas"] = _create_shared_array(areas_arr)
        arrays["iscrowd"], shms["iscrowd"] = _create_shared_array(iscrowd_arr)
        arrays["ann_ids"], shms["ann_ids"] = _create_shared_array(ann_ids_arr)
        if include_masks:
            arrays["segm_offsets"], shms["segm_offsets"] = _create_shared_array(segm_offsets)
            arrays["segm_data"], shms["segm_data"] = _create_shared_array(segm_data)

        meta = {}
        for key, arr in arrays.items():
            meta[key] = {
                "name": shms[key].name,
                "shape": arr.shape,
                "dtype": str(arr.dtype),
            }
        meta["include_masks"] = bool(include_masks)
        meta["categories"] = dataset.get("categories", [])
        meta["dataset_info"] = dataset.get("info", {})
        meta["licenses"] = dataset.get("licenses", [])

        return cls(meta=meta, arrays=arrays, shms=shms, owner=owner)

    @classmethod
    def attach(cls, meta, owner: bool = False):
        arrays = {}
        shms = {}
        for key, info in meta.items():
            if key in {"include_masks", "categories", "dataset_info", "licenses"}:
                continue
            shm = shared_memory.SharedMemory(name=info["name"])
            arr = np.ndarray(info["shape"], dtype=np.dtype(info["dtype"]), buffer=shm.buf)
            arrays[key] = arr
            shms[key] = shm
        return cls(meta=meta, arrays=arrays, shms=shms, owner=owner)

    def metadata(self):
        return self._meta

    def __len__(self):
        return int(self._arrays["ids"].shape[0])

    @property
    def ids(self):
        return self._arrays["ids"]

    @property
    def ann_ids(self):
        return self._arrays["ann_ids"]

    def image_id(self, idx: int) -> int:
        return int(self._arrays["ids"][idx])

    def id_to_index(self, image_id: int) -> Optional[int]:
        if self._id_to_index is None:
            self._id_to_index = {int(img_id): idx for idx, img_id in enumerate(self._arrays["ids"])}
        return self._id_to_index.get(int(image_id))

    def image_info(self, idx: int) -> dict:
        return {
            "id": self.image_id(idx),
            "file_name": self.file_name(idx),
            "width": int(self._arrays["widths"][idx]),
            "height": int(self._arrays["heights"][idx]),
        }

    def file_name(self, idx: int) -> str:
        offsets = self._arrays["file_name_offsets"]
        data = self._arrays["file_name_data"]
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        if end <= start:
            return ""
        return bytes(memoryview(data)[start:end]).decode("utf-8")

    def segmentations(self, idx: int):
        if not self._meta.get("include_masks"):
            return None
        offsets = self._arrays["segm_offsets"]
        data = self._arrays["segm_data"]
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        if end <= start:
            return []
        return orjson.loads(memoryview(data)[start:end])

    def target(self, idx: int):
        box_offsets = self._arrays["box_offsets"]
        boxes = self._arrays["boxes"]
        labels = self._arrays["labels"]
        areas = self._arrays["areas"]
        iscrowd = self._arrays["iscrowd"]
        start = int(box_offsets[idx])
        end = int(box_offsets[idx + 1])
        target = {
            "image_id": torch.tensor([self.image_id(idx)]),
            "boxes": torch.from_numpy(boxes[start:end]),
            "labels": torch.from_numpy(labels[start:end]),
            "area": torch.from_numpy(areas[start:end]),
            "iscrowd": torch.from_numpy(iscrowd[start:end]),
        }
        return target

    def close(self):
        for shm in self._shms.values():
            try:
                shm.close()
            except FileNotFoundError:
                pass

    def _cleanup_owner(self):
        for shm in self._shms.values():
            try:
                shm.close()
            except FileNotFoundError:
                pass
        for shm in self._shms.values():
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def __getstate__(self):
        return {"meta": self._meta}

    def __setstate__(self, state):
        attached = self.attach(state["meta"])
        self.__dict__.update(attached.__dict__)


def _is_array_like(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


def _to_list(value):
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    if _is_array_like(value):
        return list(value)
    return [value]


class SharedCOCO:
    def __init__(self, store: SharedCocoLabelStore):
        self.store = store
        meta = store.metadata()
        self.dataset = {
            "info": meta.get("dataset_info", {}),
            "licenses": meta.get("licenses", []),
            "categories": meta.get("categories", []),
            "images": self._build_images(),
            "annotations": [],
        }
        self.cats = {
            int(cat["id"]): cat
            for cat in self.dataset["categories"]
            if isinstance(cat, dict) and "id" in cat
        }
        self._ann_id_map = None
        self._segm_cache = {}
        self._imgs = None

    def __getstate__(self):
        return {"store": self.store}

    def __setstate__(self, state):
        self.__init__(state["store"])

    def _build_images(self):
        images = []
        for idx in range(len(self.store)):
            info = self.store.image_info(idx)
            images.append(info)
        return images

    @property
    def imgs(self):
        if self._imgs is None:
            self._imgs = {img["id"]: img for img in self.dataset["images"]}
        return self._imgs

    def getImgIds(self, imgIds=None, catIds=None):
        img_ids = _to_list(imgIds)
        cat_ids = _to_list(catIds)
        if len(img_ids) == 0 and len(cat_ids) == 0:
            return [int(img_id) for img_id in self.store.ids.tolist()]

        ids = set(int(img_id) for img_id in img_ids) if img_ids else None
        if cat_ids:
            cat_set = {int(cat_id) for cat_id in cat_ids}
            matched = set()
            for idx in range(len(self.store)):
                start = int(self.store._arrays["box_offsets"][idx])
                end = int(self.store._arrays["box_offsets"][idx + 1])
                if start == end:
                    continue
                labels = self.store._arrays["labels"][start:end]
                if cat_set.issubset(set(int(val) for val in labels)):
                    matched.add(self.store.image_id(idx))
            ids = matched if ids is None else ids & matched
        return list(ids or [])

    def getCatIds(self, catNms=None, supNms=None, catIds=None):
        cat_nms = _to_list(catNms)
        sup_nms = _to_list(supNms)
        cat_ids = _to_list(catIds)
        cats = [cat for cat in self.dataset.get("categories", []) if isinstance(cat, dict)]
        if len(cat_nms) == len(sup_nms) == len(cat_ids) == 0:
            return [int(cat["id"]) for cat in cats if isinstance(cat, dict) and "id" in cat]
        if cat_nms:
            cats = [cat for cat in cats if cat.get("name") in cat_nms]
        if sup_nms:
            cats = [cat for cat in cats if cat.get("supercategory") in sup_nms]
        if cat_ids:
            cat_id_set = {int(cid) for cid in cat_ids}
            cats = [cat for cat in cats if int(cat.get("id", -1)) in cat_id_set]
        return [int(cat["id"]) for cat in cats if isinstance(cat, dict) and "id" in cat]

    def loadCats(self, ids=None):
        cat_ids = _to_list(ids)
        if not cat_ids:
            return []
        return [self.cats[int(cat_id)] for cat_id in cat_ids if int(cat_id) in self.cats]

    def loadImgs(self, ids=None):
        img_ids = _to_list(ids)
        images = []
        for img_id in img_ids:
            idx = self.store.id_to_index(img_id)
            if idx is None:
                continue
            images.append(self.store.image_info(idx))
        return images

    def _ensure_ann_id_map(self):
        if self._ann_id_map is not None:
            return
        ann_id_map = {}
        ann_ids = self.store.ann_ids
        offsets = self.store._arrays["box_offsets"]
        for img_idx in range(len(self.store)):
            start = int(offsets[img_idx])
            end = int(offsets[img_idx + 1])
            if start == end:
                continue
            for local_idx, ann_id in enumerate(ann_ids[start:end]):
                ann_id_map[int(ann_id)] = (img_idx, local_idx)
        self._ann_id_map = ann_id_map

    def _segms_for_image(self, img_idx: int):
        if not self.store.metadata().get("include_masks"):
            return None
        cached = self._segm_cache.get(img_idx)
        if cached is None:
            cached = self.store.segmentations(img_idx) or []
            self._segm_cache[img_idx] = cached
        return cached

    def getAnnIds(self, imgIds=None, catIds=None, areaRng=None, iscrowd=None):
        img_ids = _to_list(imgIds)
        cat_ids = _to_list(catIds)
        area_rng = _to_list(areaRng)
        if len(img_ids) == len(cat_ids) == len(area_rng) == 0:
            if iscrowd is None:
                return [int(val) for val in self.store.ann_ids.tolist()]
            iscrowd_val = int(iscrowd)
            ann_ids = self.store.ann_ids
            crowds = self.store._arrays["iscrowd"]
            return [int(ann_id) for ann_id, crowd in zip(ann_ids, crowds) if int(crowd) == iscrowd_val]

        if not img_ids:
            img_ids = [self.store.image_id(idx) for idx in range(len(self.store))]

        cat_set = {int(cat_id) for cat_id in cat_ids} if cat_ids else None
        use_area = len(area_rng) >= 2
        min_area = float(area_rng[0]) if use_area else None
        max_area = float(area_rng[1]) if use_area else None
        iscrowd_val = int(iscrowd) if iscrowd is not None else None

        ann_ids_out = []
        offsets = self.store._arrays["box_offsets"]
        labels_arr = self.store._arrays["labels"]
        areas_arr = self.store._arrays["areas"]
        iscrowd_arr = self.store._arrays["iscrowd"]
        ann_ids_arr = self.store.ann_ids

        for img_id in img_ids:
            idx = self.store.id_to_index(img_id)
            if idx is None:
                continue
            start = int(offsets[idx])
            end = int(offsets[idx + 1])
            if start == end:
                continue
            labels = labels_arr[start:end]
            areas = areas_arr[start:end]
            crowds = iscrowd_arr[start:end]
            ann_ids = ann_ids_arr[start:end]
            mask = np.ones((end - start,), dtype=bool)
            if cat_set is not None:
                mask &= np.isin(labels, list(cat_set))
            if use_area:
                mask &= (areas > min_area) & (areas < max_area)
            if iscrowd_val is not None:
                mask &= (crowds == iscrowd_val)
            if mask.any():
                ann_ids_out.extend(int(val) for val in ann_ids[mask])
        return ann_ids_out

    def loadAnns(self, ids=None):
        ann_ids = _to_list(ids)
        if not ann_ids:
            return []
        self._ensure_ann_id_map()
        anns = []
        offsets = self.store._arrays["box_offsets"]
        boxes = self.store._arrays["boxes"]
        labels = self.store._arrays["labels"]
        areas = self.store._arrays["areas"]
        crowds = self.store._arrays["iscrowd"]

        for ann_id in ann_ids:
            mapped = self._ann_id_map.get(int(ann_id))
            if mapped is None:
                continue
            img_idx, local_idx = mapped
            start = int(offsets[img_idx])
            global_idx = start + local_idx
            box = boxes[global_idx]
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            ann = {
                "id": int(ann_id),
                "image_id": self.store.image_id(img_idx),
                "category_id": int(labels[global_idx]),
                "bbox": [x1, y1, float(x2 - x1), float(y2 - y1)],
                "area": float(areas[global_idx]),
                "iscrowd": int(crowds[global_idx]),
            }
            if self.store.metadata().get("include_masks"):
                segms = self._segms_for_image(img_idx) or []
                if local_idx < len(segms):
                    ann["segmentation"] = segms[local_idx]
                else:
                    ann["segmentation"] = []
            anns.append(ann)
        return anns

    def annToRLE(self, ann):
        img_idx = self.store.id_to_index(ann["image_id"])
        if img_idx is None:
            raise KeyError(f"image_id {ann['image_id']} not found")
        height = int(self.store._arrays["heights"][img_idx])
        width = int(self.store._arrays["widths"][img_idx])
        segm = ann["segmentation"]
        if isinstance(segm, list):
            rles = coco_mask.frPyObjects(segm, height, width)
            rle = coco_mask.merge(rles)
        elif isinstance(segm, dict) and isinstance(segm.get("counts"), list):
            rle = coco_mask.frPyObjects(segm, height, width)
        else:
            rle = ann["segmentation"]
        return rle

    def annToMask(self, ann):
        rle = self.annToRLE(ann)
        return coco_mask.decode(rle)


_SHARED_LABEL_STORE_CACHE = {}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _shared_label_meta_path(ann_file: Path, include_masks: bool) -> Path:
    master_port = os.environ.get("MASTER_PORT", "0")
    uid = str(os.getuid()) if hasattr(os, "getuid") else "0"
    run_id = os.environ.get("TORCHELASTIC_RUN_ID", "")
    token = f"{ann_file.resolve()}|{include_masks}|{master_port}|{uid}|{run_id}"
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return Path(tempfile.gettempdir()) / f"rfdetr_label_store_{digest}.json"


def _read_label_store_meta(meta_path: Path, ann_file: Path):
    _use_orjson()
    try:
        data = meta_path.read_bytes()
    except FileNotFoundError:
        return None
    try:
        payload = orjson.loads(data)
    except orjson.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return None
    if payload.get("ann_file") != str(ann_file):
        return None
    stored_mtime = payload.get("ann_mtime")
    if stored_mtime is not None:
        try:
            current_mtime = ann_file.stat().st_mtime
        except OSError:
            return None
        if abs(float(stored_mtime) - current_mtime) > 1e-6:
            return None
    return meta


def _write_label_store_meta(meta_path: Path, ann_file: Path, meta: dict) -> None:
    _use_orjson()
    try:
        ann_mtime = ann_file.stat().st_mtime
    except OSError:
        ann_mtime = None
    payload = {
        "ann_file": str(ann_file),
        "ann_mtime": ann_mtime,
        "meta": meta,
    }
    tmp_path = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_path.write_bytes(orjson.dumps(payload))
    os.replace(tmp_path, meta_path)


def _label_store_helper_main(ann_file: str, include_masks: bool, meta_path: str) -> None:
    ann_path = Path(ann_file)
    store = SharedCocoLabelStore.build(ann_path, include_masks=include_masks, owner=False)
    meta = store.metadata()
    _write_label_store_meta(Path(meta_path), ann_path, meta)


def _build_label_store_with_helper(ann_file: Path, include_masks: bool, meta_path: Path) -> None:
    ctx = mp_std.get_context("spawn")
    proc = ctx.Process(
        target=_label_store_helper_main,
        args=(str(ann_file), include_masks, str(meta_path)),
    )
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"label store helper failed with exit code {proc.exitcode}")


def _wait_for_label_meta(meta_path: Path, ann_file: Path, timeout: float = 300.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        meta = _read_label_store_meta(meta_path, ann_file)
        if meta is not None:
            return meta
        time.sleep(0.1)
    raise TimeoutError(f"timed out waiting for label store metadata at {meta_path}")


def _should_use_helper() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_initialized()


def maybe_shared_labels(ann_file: Path, include_masks: bool, args):
    if not bool(getattr(args, "share_labels", False)):
        return None
    try:
        ann_file = ann_file.resolve()
    except OSError:
        ann_file = ann_file
    key = (str(ann_file), bool(include_masks))
    if key in _SHARED_LABEL_STORE_CACHE:
        return _SHARED_LABEL_STORE_CACHE[key]

    rank = _env_int("RANK", 0)
    meta_path = _shared_label_meta_path(ann_file, include_masks)
    use_helper = _should_use_helper()

    store = None
    meta = _read_label_store_meta(meta_path, ann_file)
    if meta is not None and rank == 0:
        try:
            store = SharedCocoLabelStore.attach(meta, owner=True)
        except FileNotFoundError:
            meta = None

    if meta is None and rank == 0:
        if use_helper:
            _build_label_store_with_helper(ann_file, include_masks, meta_path)
            meta = _wait_for_label_meta(meta_path, ann_file)
            store = SharedCocoLabelStore.attach(meta, owner=True)
        else:
            store = SharedCocoLabelStore.build(ann_file, include_masks=include_masks, owner=True)
            meta = store.metadata()
            _write_label_store_meta(meta_path, ann_file, meta)

    if store is None:
        if meta is None:
            meta = _wait_for_label_meta(meta_path, ann_file)
        while True:
            try:
                store = SharedCocoLabelStore.attach(meta, owner=False)
                break
            except FileNotFoundError:
                meta = _wait_for_label_meta(meta_path, ann_file)
    if utils.is_dist_avail_and_initialized():
        dist.barrier()
    _SHARED_LABEL_STORE_CACHE[key] = store
    return store


def compute_multi_scale_scales(resolution, expanded_scales=False, patch_size=16, num_windows=4):
    # round to the nearest multiple of 4*patch_size to enable both patching and windowing
    base_num_patches_per_window = resolution // (patch_size * num_windows)
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4] if not expanded_scales else [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
    scales = [base_num_patches_per_window + offset for offset in offsets]
    proposed_scales = [scale * patch_size * num_windows for scale in scales]
    proposed_scales = [scale for scale in proposed_scales if scale >= patch_size * num_windows * 2]  # ensure minimum image size
    return proposed_scales


def convert_coco_poly_to_mask(segmentations, height, width):
    """Convert polygon segmentation to a binary mask tensor of shape [N, H, W].
    Requires pycocotools.
    """
    masks = []
    for polygons in segmentations:
        if polygons is None or len(polygons) == 0:
            # empty segmentation for this instance
            masks.append(torch.zeros((height, width), dtype=torch.uint8))
            continue
        try:
            rles = coco_mask.frPyObjects(polygons, height, width)
        except:
            rles = polygons
        mask = coco_mask.decode(rles)
        if mask.ndim < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if len(masks) == 0:
        return torch.zeros((0, height, width), dtype=torch.uint8)
    return torch.stack(masks, dim=0)


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        img_folder,
        ann_file,
        transforms,
        include_masks=False,
        stream_labels=True,
        cache_images=False,
        cache_dir=None,
        cache_transforms=None,
        shared_labels=None,
    ):
        VisionDataset.__init__(
            self, img_folder, transforms=None, transform=None, target_transform=None
        )
        self._shared_labels = shared_labels
        if self._shared_labels is not None:
            self.coco = SharedCOCO(self._shared_labels)
            self.ids = self._shared_labels.ids
            self._img_infos = None
            self._annos_bytes = None
        else:
            ann_path = str(ann_file)
            if _use_orjson():
                self.coco = OrjsonCOCO(ann_path)
            else:
                self.coco = COCO(ann_path)
            self.ids = list(sorted(self.coco.imgs.keys()))
            if stream_labels:
                self._img_infos = [self.coco.imgs[img_id] for img_id in self.ids]
                self._annos_bytes = [
                    orjson.dumps(self.coco.imgToAnns.get(img_id, []))
                    for img_id in self.ids
                ]
            else:
                self._img_infos = None
                self._annos_bytes = None
        self._transforms = transforms
        self._cache_transforms = cache_transforms
        self.include_masks = include_masks
        self.prepare = ConvertCoco(include_masks=include_masks)
        self.cache_images = bool(cache_images)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache_ready = False
        self._shared_ready = False
        self._shared_images = None
        self._shared_targets = None
        self._split_name = Path(self.root).name
        self._root_override = self._infer_root_override()
        if self.cache_images:
            if self.cache_dir is None:
                self.cache_dir = Path(self.root).parent / "cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            if self._cache_transforms is None:
                self._cache_transforms = T.Compose([T.NormalizeTarget()])

    def _infer_root_override(self) -> Optional[Path]:
        if len(self.ids) == 0:
            return None
        root = Path(self.root)
        parent = root.parent
        for idx in range(min(50, len(self.ids))):
            info = self._image_info_for_index(idx)
            if not isinstance(info, dict):
                continue
            file_name = info.get("file_name")
            if not isinstance(file_name, str):
                continue
            rel = Path(file_name)
            if rel.is_absolute() or rel.parent == Path("."):
                continue
            candidate = root / rel
            if candidate.exists():
                return None
            parent_candidate = parent / rel
            if parent_candidate.exists():
                return parent
        return None

    def _resolve_image_path(self, file_name: str) -> Path:
        if os.path.isabs(file_name):
            return Path(file_name)
        rel = Path(file_name)
        if self._root_override is not None and rel.parent != Path("."):
            return self._root_override / rel
        return Path(self.root) / rel

    def _cache_relpath(self, file_name: str) -> Path:
        rel = Path(file_name)
        if rel.is_absolute():
            rel = Path(rel.name)
        if rel.parent == Path("."):
            rel = Path(self._split_name) / rel
        return rel.with_suffix(".npy")

    def _cache_path_for(self, file_name: str) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / self._cache_relpath(file_name)

    def _cache_is_stale(self, cache_path: Path, image_path: Optional[Path]) -> bool:
        if image_path is None:
            return False
        try:
            if not image_path.exists():
                return False
            return cache_path.stat().st_mtime < image_path.stat().st_mtime
        except OSError:
            return True

    def _load_cache(self, cache_path: Path) -> torch.Tensor:
        array = np.load(cache_path, allow_pickle=False)
        if array.ndim != 3 or array.shape[0] != 3:
            raise ValueError("cache has unexpected shape")
        if array.dtype != np.float32:
            array = array.astype(np.float32, copy=False)
        return torch.from_numpy(array)

    def _write_cache(self, cache_path: Path, tensor: torch.Tensor) -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        array = tensor.detach().cpu().contiguous().numpy()
        tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with tmp_path.open("wb") as f:
            np.save(f, array, allow_pickle=False)
        os.replace(tmp_path, cache_path)

    def _image_info_for_index(self, idx: int) -> Optional[dict]:
        if self._shared_labels is not None:
            info = self._shared_labels.image_info(idx)
            file_name = info.get("file_name") if isinstance(info, dict) else None
            return info if isinstance(file_name, str) and file_name else None
        if self._img_infos is not None:
            return self._img_infos[idx]
        image_id = self.ids[idx]
        return self.coco.imgs.get(image_id)

    def _cache_is_complete(self) -> bool:
        if not self.cache_images or self.cache_dir is None:
            return True
        for idx in range(len(self.ids)):
            img_info = self._image_info_for_index(idx)
            if not isinstance(img_info, dict):
                return False
            file_name = img_info.get("file_name")
            if not isinstance(file_name, str):
                return False
            cache_path = self._cache_path_for(file_name)
            if cache_path is None or not cache_path.is_file():
                return False
            image_path = self._resolve_image_path(file_name)
            if self._cache_is_stale(cache_path, image_path):
                return False
        return True

    def ensure_cache(self, workers: int = 0, batch_size: int = 64) -> None:
        if not self.cache_images or self._cache_ready:
            return
        if self._cache_is_complete():
            self._cache_ready = True
            return
        loader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=lambda batch: batch,
            persistent_workers=workers > 0,
        )
        try:
            from tqdm import tqdm
        except ImportError:
            tqdm = None
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(
                loader,
                total=len(loader),
                desc=f"cache {self._split_name}",
                unit="batch",
                dynamic_ncols=True,
                mininterval=0.2,
                leave=False,
            )
        for _ in iterator:
            pass
        self._cache_ready = True

    def __getstate__(self):
        state = self.__dict__.copy()
        if state.get("_shared_ready"):
            state["coco"] = None
            state["_annos_bytes"] = None
        if state.get("_shared_labels") is not None:
            state["coco"] = None
            state["_img_infos"] = None
            state["_annos_bytes"] = None
        return state

    def preload_shared(self, workers: Optional[int] = None, chunk_size: Optional[int] = None) -> None:
        if self._shared_ready:
            return
        total = len(self.ids)
        if total == 0:
            self._shared_images = []
            self._shared_targets = []
            self._shared_ready = True
            return
        if workers is None:
            if hasattr(os, "sched_getaffinity"):
                try:
                    workers = max(1, len(os.sched_getaffinity(0)) - 2)
                except NotImplementedError:
                    workers = max(1, (os.cpu_count() or 1) - 2)
            else:
                workers = max(1, (os.cpu_count() or 1) - 2)
        workers = max(1, int(workers))
        if chunk_size is None:
            chunk_size = max(1, total // max(1, workers * 4))
        indices = list(range(total))
        chunks = [indices[i:i + chunk_size] for i in range(0, total, chunk_size)]
        shared_images = [None] * total
        shared_targets = [None] * total

        if workers == 1:
            for idx in indices:
                img, target = self._getitem_no_shared(idx)
                img = _share_tensor(img)
                target = _share_target(target)
                shared_images[idx] = img
                shared_targets[idx] = target
        else:
            try:
                ctx = torch_mp.get_context("fork")
            except ValueError:
                ctx = torch_mp.get_context()
            with ctx.Pool(processes=workers, initializer=_init_shared_preload, initargs=(self,)) as pool:
                for results in pool.imap_unordered(_preload_chunk, chunks, chunksize=1):
                    for idx, img, target in results:
                        shared_images[idx] = img
                        shared_targets[idx] = target

        if any(item is None for item in shared_images) or any(item is None for item in shared_targets):
            raise RuntimeError("shared preload did not populate all entries")
        self._shared_images = shared_images
        self._shared_targets = shared_targets
        self._shared_ready = True

    def _build_shared_target(self, idx, img):
        if isinstance(img, Image.Image):
            w, h = img.size
        elif torch.is_tensor(img):
            h, w = img.shape[-2:]
        else:
            h, w = img.shape[:2]

        target = self._shared_labels.target(idx)
        boxes = target["boxes"]
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        if "iscrowd" in target:
            keep &= target["iscrowd"] == 0
        target["boxes"] = boxes[keep]
        target["labels"] = target["labels"][keep]
        target["area"] = target["area"][keep]
        if "iscrowd" in target:
            target["iscrowd"] = target["iscrowd"][keep]
        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        if self.include_masks:
            segms = self._shared_labels.segmentations(idx) or []
            if segms:
                keep_list = keep.tolist()
                segms = [segm for segm, keep_val in zip(segms, keep_list) if keep_val]
            if segms:
                masks = convert_coco_poly_to_mask(segms, h, w)
                target["masks"] = masks.bool()
            else:
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8).bool()
        return target

    def _getitem_no_shared(self, idx):
        if self._shared_labels is not None:
            image_id = self._shared_labels.image_id(idx)
        else:
            image_id = self.ids[idx]
        annotations = None
        file_name = None
        if self._shared_labels is None:
            img_info = self._image_info_for_index(idx)
            if self._img_infos is None:
                annotations = self.coco.imgToAnns.get(image_id, [])
            else:
                annotations = orjson.loads(self._annos_bytes[idx])
            if isinstance(img_info, dict):
                file_name = img_info.get("file_name")
            if not isinstance(file_name, str) or not file_name:
                file_name = None
        else:
            file_name = self._shared_labels.file_name(idx)
            if not isinstance(file_name, str) or not file_name:
                file_name = None

        image_path = None
        cache_path = None
        if isinstance(file_name, str):
            image_path = self._resolve_image_path(file_name)
            if self.cache_images:
                cache_path = self._cache_path_for(file_name)

        if cache_path is not None and cache_path.is_file():
            if not self._cache_is_stale(cache_path, image_path):
                try:
                    img = self._load_cache(cache_path)
                except Exception:
                    img = None
                if img is not None:
                    if self._shared_labels is not None:
                        target = self._build_shared_target(idx, img)
                    else:
                        target = {"image_id": image_id, "annotations": annotations}
                        img, target = self.prepare(img, target)
                    if self._cache_transforms is not None:
                        img, target = self._cache_transforms(img, target)
                    return img, target

        if image_path is None:
            img, target = super(CocoDetection, self).__getitem__(idx)
            target = {"image_id": image_id, "annotations": target}
        else:
            img = Image.open(image_path).convert("RGB")
            target = {"image_id": image_id, "annotations": annotations}
        if self._shared_labels is not None:
            target = self._build_shared_target(idx, img)
        else:
            img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if cache_path is not None and torch.is_tensor(img):
            self._write_cache(cache_path, img)
        return img, target

    def __getitem__(self, idx):
        if self._shared_ready:
            img = self._shared_images[idx]
            target = self._shared_targets[idx]
            if img is None or target is None:
                raise RuntimeError("shared preload is missing data")
            return img, target
        return self._getitem_no_shared(idx)


class ConvertCoco(object):

    def __init__(self, include_masks=False):
        self.include_masks = include_masks

    def __call__(self, image, target):
        if image is None:
            width = target.get("width")
            height = target.get("height")
            if not isinstance(width, int) or not isinstance(height, int):
                raise ValueError("image size missing for COCO target")
            w, h = width, height
        elif torch.is_tensor(image):
            h, w = image.shape[-2:]
        elif isinstance(image, np.ndarray):
            h, w = image.shape[:2]
        else:
            w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # add segmentation masks if requested, otherwise ensure consistent key when include_masks=True
        if self.include_masks:
            if len(anno) > 0 and 'segmentation' in anno[0]:
                segmentations = [obj.get("segmentation", []) for obj in anno]
                masks = convert_coco_poly_to_mask(segmentations, h, w)
                if masks.numel() > 0:
                    target["masks"] = masks[keep]
                else:
                    target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)
            else:
                target["masks"] = torch.zeros((0, h, w), dtype=torch.uint8)

            target["masks"] = target["masks"].bool()

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return normalize


def make_coco_transforms_square_div_64(image_set, resolution, multi_scale=False, expanded_scales=False, skip_random_resize=False, patch_size=16, num_windows=4):
    """
    """

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return normalize

def build(image_set, args, resolution):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    cache_images, cache_dir = _cache_args(root, args)
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root /  "val2017", root / "annotations" / f'{mode}_val2017.json'),
        "test": (root / "test2017", root / "annotations" / f'image_info_test-dev2017.json'),
    }
    
    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    
    try:
        square_resize = args.square_resize
    except:
        square_resize = False
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False

    
    include_masks = bool(getattr(args, "segmentation_head", False))
    shared_labels = maybe_shared_labels(ann_file, include_masks, args)

    if square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                skip_random_resize=not args.do_random_resize_via_padding,
                patch_size=args.patch_size,
                num_windows=args.num_windows,
            ),
            include_masks=include_masks,
            cache_images=cache_images,
            cache_dir=cache_dir,
            shared_labels=shared_labels,
        )
    else:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                skip_random_resize=not args.do_random_resize_via_padding,
                patch_size=args.patch_size,
                num_windows=args.num_windows,
            ),
            include_masks=include_masks,
            cache_images=cache_images,
            cache_dir=cache_dir,
            shared_labels=shared_labels,
        )
    return dataset

def build_roboflow(image_set, args, resolution):
    root = Path(args.dataset_dir)
    assert root.exists(), f'provided Roboflow path {root} does not exist'
    cache_images, cache_dir = _cache_args(root, args)
    mode = 'instances'
    PATHS = {
        "train": (root / "train", root / "train" / "_annotations.coco.json"),
        "val": (root /  "valid", root / "valid" / "_annotations.coco.json"),
        "test": (root / "test", root / "test" / "_annotations.coco.json"),
    }
    
    img_folder, ann_file = PATHS[image_set.split("_")[0]]
    
    try:
        square_resize = args.square_resize
    except:
        square_resize = False
    
    try:
        square_resize_div_64 = args.square_resize_div_64
    except:
        square_resize_div_64 = False
    
    try:
        include_masks = args.segmentation_head
    except:
        include_masks = False

    
    shared_labels = maybe_shared_labels(ann_file, include_masks, args)

    if square_resize_div_64:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms_square_div_64(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                skip_random_resize=not args.do_random_resize_via_padding,
                patch_size=args.patch_size,
                num_windows=args.num_windows,
            ),
            include_masks=include_masks,
            cache_images=cache_images,
            cache_dir=cache_dir,
            shared_labels=shared_labels,
        )
    else:
        dataset = CocoDetection(
            img_folder,
            ann_file,
            transforms=make_coco_transforms(
                image_set,
                resolution,
                multi_scale=args.multi_scale,
                expanded_scales=args.expanded_scales,
                skip_random_resize=not args.do_random_resize_via_padding,
                patch_size=args.patch_size,
                num_windows=args.num_windows,
            ),
            include_masks=include_masks,
            cache_images=cache_images,
            cache_dir=cache_dir,
            shared_labels=shared_labels,
        )
    return dataset
