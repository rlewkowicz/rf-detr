# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

from collections import deque

import torch

from rfdetr.util.misc import NestedTensor


def _move_to_device(value, device):
    if isinstance(value, NestedTensor):
        return value.to(device, non_blocking=True)
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


def _record_stream(value, stream):
    if isinstance(value, NestedTensor):
        value.tensors.record_stream(stream)
        if value.mask is not None:
            value.mask.record_stream(stream)
        return
    if torch.is_tensor(value):
        value.record_stream(stream)
        return
    if isinstance(value, dict):
        for v in value.values():
            _record_stream(v, stream)
        return
    if isinstance(value, (list, tuple)):
        for v in value:
            _record_stream(v, stream)
        return


def _estimate_bytes(value):
    if isinstance(value, NestedTensor):
        total = _estimate_bytes(value.tensors)
        if value.mask is not None:
            total += _estimate_bytes(value.mask)
        return total
    if torch.is_tensor(value):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_estimate_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_estimate_bytes(v) for v in value)
    return 0


class CUDAPrefetcher:
    def __init__(
        self,
        data_loader,
        device,
        prefetch_batches=2,
        warmup_batches=0,
        max_prefetch_mem_frac=0.6,
    ):
        self.data_loader = data_loader
        self.device = device
        self.prefetch_batches = (
            None if prefetch_batches is None else max(1, int(prefetch_batches))
        )
        self.warmup_batches = max(0, int(warmup_batches))
        self.max_prefetch_mem_frac = float(max_prefetch_mem_frac)
        self.stream = torch.cuda.Stream()
        self._queue = deque()
        self._iter = None
        self._length = len(data_loader) if hasattr(data_loader, "__len__") else None
        self._warmup_remaining = 0
        self._prefetch_target = self.prefetch_batches
        self._batch_bytes = None

    def __len__(self):
        if self._length is None:
            raise TypeError("CUDAPrefetcher has no length")
        return self._length

    def __iter__(self):
        self._iter = iter(self.data_loader)
        self._queue.clear()
        self._warmup_remaining = self.warmup_batches
        self._prefetch_target = self.prefetch_batches
        self._batch_bytes = None
        if self._warmup_remaining == 0 and self._prefetch_target is not None:
            self._prefill()
        return self

    def __next__(self):
        if self._warmup_remaining > 0:
            batch = self._next_on_demand()
            self._warmup_remaining -= 1
            batch_bytes = _estimate_bytes(batch)
            if batch_bytes:
                self._batch_bytes = (
                    batch_bytes
                    if self._batch_bytes is None
                    else max(self._batch_bytes, batch_bytes)
                )
            if self._warmup_remaining == 0 and self._prefetch_target is None:
                self._init_dynamic_prefetch()
                self._prefill()
            return batch
        if not self._queue:
            if self._prefetch_target is None:
                batch = self._next_on_demand()
                batch_bytes = _estimate_bytes(batch)
                if batch_bytes:
                    self._batch_bytes = (
                        batch_bytes
                        if self._batch_bytes is None
                        else max(self._batch_bytes, batch_bytes)
                    )
                self._init_dynamic_prefetch()
                self._prefill()
                return batch
            raise StopIteration
        batch = self._queue.popleft()
        if batch is None:
            self._queue.clear()
            raise StopIteration
        torch.cuda.current_stream().wait_stream(self.stream)
        samples, targets = batch
        _record_stream(samples, torch.cuda.current_stream())
        _record_stream(targets, torch.cuda.current_stream())
        if self._prefetch_target is not None:
            self._preload()
        return samples, targets

    def _next_on_demand(self):
        try:
            batch = next(self._iter)
        except StopIteration:
            raise StopIteration
        samples, targets = batch
        samples = _move_to_device(samples, self.device)
        targets = _move_to_device(targets, self.device)
        return samples, targets

    def _prefill(self):
        if self._prefetch_target is None:
            return
        while len(self._queue) < self._prefetch_target:
            self._preload()
            if self._queue and self._queue[-1] is None:
                break

    def _preload(self):
        if self._iter is None:
            return
        try:
            batch = next(self._iter)
        except StopIteration:
            self._queue.append(None)
            return
        with torch.cuda.stream(self.stream):
            samples, targets = batch
            samples = _move_to_device(samples, self.device)
            targets = _move_to_device(targets, self.device)
        self._queue.append((samples, targets))

    def _init_dynamic_prefetch(self):
        if self.prefetch_batches is not None:
            self._prefetch_target = self.prefetch_batches
            return
        if self._batch_bytes in (None, 0):
            self._prefetch_target = 2
            return
        device_index = (
            self.device.index
            if isinstance(self.device, torch.device) and self.device.index is not None
            else torch.cuda.current_device()
        )
        free_bytes, _ = torch.cuda.mem_get_info(device_index)
        target_bytes = int(free_bytes * self.max_prefetch_mem_frac)
        self._prefetch_target = max(1, target_bytes // max(1, self._batch_bytes))


def maybe_cuda_prefetcher(
    data_loader,
    device,
    enabled=True,
    prefetch_batches=2,
    warmup_batches=0,
    max_prefetch_mem_frac=0.6,
):
    if not enabled or device.type != "cuda" or not torch.cuda.is_available():
        return data_loader
    if isinstance(data_loader, CUDAPrefetcher):
        return data_loader
    return CUDAPrefetcher(
        data_loader,
        device=device,
        prefetch_batches=prefetch_batches,
        warmup_batches=warmup_batches,
        max_prefetch_mem_frac=max_prefetch_mem_frac,
    )
