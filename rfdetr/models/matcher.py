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
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from scipy.optimize import linear_sum_assignment
from torch import nn
import torch.nn.functional as F

from rfdetr.util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou, batch_sigmoid_ce_loss, batch_dice_loss
from rfdetr.models.segmentation_head import point_sample

_MATCHER_POOL = None
_MATCHER_POOL_WORKERS = None


def _default_matcher_workers():
    cpu_count = os.cpu_count() or 1
    return max(1, cpu_count - 2)


def _get_matcher_pool():
    global _MATCHER_POOL, _MATCHER_POOL_WORKERS
    workers = _default_matcher_workers()
    if _MATCHER_POOL is None or _MATCHER_POOL_WORKERS != workers:
        if _MATCHER_POOL is not None:
            _MATCHER_POOL.shutdown(wait=False)
        _MATCHER_POOL = ThreadPoolExecutor(max_workers=workers)
        _MATCHER_POOL_WORKERS = workers
    return _MATCHER_POOL


def _run_linear_sum_assignment(cost_matrix):
    return linear_sum_assignment(cost_matrix)


def _parallel_linear_sum_assignment(cost_matrix, sizes, group_detr):
    bs, num_queries, _ = cost_matrix.shape
    g_num_queries = num_queries // group_detr
    offsets = [0]
    for size in sizes:
        offsets.append(offsets[-1] + size)
    if offsets[-1] != cost_matrix.shape[2]:
        raise RuntimeError("Sum of target sizes must match cost matrix width.")

    tasks = []
    for g_i in range(group_detr):
        row_start = g_i * g_num_queries
        row_end = row_start + g_num_queries
        for b_i in range(bs):
            col_start = offsets[b_i]
            col_end = offsets[b_i + 1]
            tasks.append((g_i, b_i, cost_matrix[b_i, row_start:row_end, col_start:col_end]))

    if not tasks:
        return []
    if len(tasks) == 1:
        g_i, b_i, mat = tasks[0]
        return [(g_i, b_i, linear_sum_assignment(mat))]

    pool = _get_matcher_pool()
    matrices = [task[2] for task in tasks]
    results = list(pool.map(_run_linear_sum_assignment, matrices))
    return [(tasks[i][0], tasks[i][1], results[i]) for i in range(len(tasks))]


@torch.compile(mode="max-autotune")
def compute_cost_matrix(
    pred_logits,
    pred_boxes,
    tgt_ids,
    tgt_bbox,
    cost_bbox,
    cost_class,
    cost_giou,
    pred_masks=None,
    tgt_masks=None,
    cost_mask_ce=0.0,
    cost_mask_dice=0.0,
    mask_point_sample_ratio=1,
):
    flat_pred_logits = pred_logits.flatten(0, 1)
    out_prob = flat_pred_logits.sigmoid()
    out_bbox = pred_boxes.flatten(0, 1)

    giou = generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
    cost_giou_matrix = -giou

    alpha = 0.25
    gamma = 2.0
    neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-F.logsigmoid(-flat_pred_logits))
    pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-F.logsigmoid(flat_pred_logits))
    cost_class_matrix = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

    cost_bbox_matrix = torch.cdist(out_bbox, tgt_bbox, p=1)

    C = cost_bbox * cost_bbox_matrix + cost_class * cost_class_matrix + cost_giou * cost_giou_matrix

    if pred_masks is not None and tgt_masks is not None:
        out_masks = pred_masks.flatten(0, 1)
        num_points = out_masks.shape[-2] * out_masks.shape[-1] // mask_point_sample_ratio

        tgt_masks = tgt_masks.to(out_masks.dtype)

        point_coords = torch.rand(1, num_points, 2, device=out_masks.device)
        pred_masks_logits = point_sample(
            out_masks.unsqueeze(1),
            point_coords.repeat(out_masks.shape[0], 1, 1),
            align_corners=False,
        ).squeeze(1)
        tgt_masks_flat = point_sample(
            tgt_masks.unsqueeze(1),
            point_coords.repeat(tgt_masks.shape[0], 1, 1),
            align_corners=False,
            mode="nearest",
        ).squeeze(1)

        cost_mask_ce_matrix = batch_sigmoid_ce_loss(pred_masks_logits, tgt_masks_flat)
        cost_mask_dice_matrix = batch_dice_loss(pred_masks_logits, tgt_masks_flat)
        C = C + cost_mask_ce * cost_mask_ce_matrix + cost_mask_dice * cost_mask_dice_matrix

    return C


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha: float = 0.25, use_pos_only: bool = False,
                 use_position_modulated_cost: bool = False, mask_point_sample_ratio: int = 16, cost_mask_ce: float = 1, cost_mask_dice: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"
        self.focal_alpha = focal_alpha
        self.mask_point_sample_ratio = mask_point_sample_ratio
        self.cost_mask_ce = cost_mask_ce
        self.cost_mask_dice = cost_mask_dice

    @torch.no_grad()
    def forward(self, outputs, targets, group_detr=1):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: Dict of padded tensors with keys:
                 "labels": Tensor of dim [batch_size, max_targets]
                 "boxes": Tensor of dim [batch_size, max_targets, 4]
                 "lengths": Tensor of dim [batch_size] with per-image target counts
                 "masks": Optional tensor of dim [batch_size, max_targets, H, W]
            group_detr: Number of groups used for matching.
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        max_targets = targets["labels"].shape[1]
        valid_mask = torch.arange(max_targets, device=targets["labels"].device)[None, :] < targets["lengths"][:, None]
        tgt_ids = targets["labels"][valid_mask]
        tgt_bbox = targets["boxes"][valid_mask]

        masks_present = "masks" in targets
        tgt_masks = None
        pred_masks = None
        if masks_present:
            tgt_masks = targets["masks"][valid_mask]
            pred_masks = outputs["pred_masks"]

        C = compute_cost_matrix(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            tgt_ids,
            tgt_bbox,
            self.cost_bbox,
            self.cost_class,
            self.cost_giou,
            pred_masks=pred_masks,
            tgt_masks=tgt_masks,
            cost_mask_ce=self.cost_mask_ce,
            cost_mask_dice=self.cost_mask_dice,
            mask_point_sample_ratio=self.mask_point_sample_ratio,
        )
        C = C.view(bs, num_queries, -1).float().cpu().contiguous()  # convert to float because bfloat16 doesn't play nicely with CPU

        # we assume any good match will not cause NaN or Inf, so we replace them with a large value
        max_cost = C.max() if C.numel() > 0 else 0
        C[C.isinf() | C.isnan()] = max_cost * 2

        sizes = targets["lengths"].to("cpu").tolist()
        assignments = _parallel_linear_sum_assignment(C.numpy(), sizes, group_detr)
        indices_by_group = [[None] * bs for _ in range(group_detr)]
        for g_i, b_i, match in assignments:
            indices_by_group[g_i][b_i] = match
        empty_match = (np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64))
        for g_i in range(group_detr):
            for b_i in range(bs):
                if indices_by_group[g_i][b_i] is None:
                    indices_by_group[g_i][b_i] = empty_match

        indices = []
        g_num_queries = num_queries // group_detr
        for g_i in range(group_detr):
            indices_g = indices_by_group[g_i]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher(args):
    if args.segmentation_head:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
            cost_mask_ce=args.mask_ce_loss_coef,
            cost_mask_dice=args.mask_dice_loss_coef,
            mask_point_sample_ratio=args.mask_point_sample_ratio,)
    else:
        return HungarianMatcher(
            cost_class=args.set_cost_class,
            cost_bbox=args.set_cost_bbox,
            cost_giou=args.set_cost_giou,
            focal_alpha=args.focal_alpha,
        )
