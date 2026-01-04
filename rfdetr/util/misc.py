# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Conditional DETR
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import datetime
import os
import pickle
import queue
import subprocess
import threading
import time
from collections import defaultdict, deque
from typing import Optional, List

import torch
import torch.distributed as dist
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from torch import Tensor

if float(torchvision.__version__.split(".")[1]) < 7.0:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        if not self.deque:
            return 0.0
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        if not self.deque:
            return 0.0
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0
        return self.total / self.count

    @property
    def max(self):
        if not self.deque:
            return 0.0
        return max(self.deque)

    @property
    def value(self):
        if not self.deque:
            return 0.0
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


class _AsyncLogWorker:
    def __init__(self, enabled: bool, wandb=None):
        self._enabled = bool(enabled)
        self._wandb = wandb
        self._queue = None
        self._stop = None
        self._thread = None
        if not self._enabled:
            return
        self._queue = queue.SimpleQueue()
        self._stop = object()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is self._stop:
                break
            message, log_dict = item
            if log_dict and self._wandb:
                try:
                    self._wandb.log(log_dict)
                except Exception:
                    pass
            if message:
                print(message)

    def log(self, message=None, log_dict=None):
        if not self._enabled:
            if log_dict and self._wandb:
                try:
                    self._wandb.log(log_dict)
                except Exception:
                    pass
            if message:
                print(message)
            return
        self._queue.put((message, log_dict))

    def close(self):
        if not self._enabled or self._thread is None:
            return
        self._queue.put(self._stop)
        self._thread.join(timeout=5)


class MetricLogger(object):
    def __init__(self, delimiter="\t", wandb_logging=False, async_logging=True):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        if wandb_logging and is_main_process():
            import wandb
            self.wandb = wandb
        else:
            self.wandb = None
        # Async logging is always enabled for the main process.
        self._async_logger = _AsyncLogWorker(
            enabled=is_main_process(),
            wandb=self.wandb,
        )

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def close(self):
        if self._async_logger is not None:
            self._async_logger.close()

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        iterable_len = len(iterable)
        space_fmt = ':' + str(len(str(iterable_len))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == iterable_len - 1:
                eta_seconds = iter_time.global_avg * (iterable_len - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if self.wandb:
                    log_dict = {k: v.value for k, v in self.meters.items()}
                else:
                    log_dict = None
                if torch.cuda.is_available():
                    log_message = log_msg.format(
                        i, iterable_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB)
                else:
                    log_message = log_msg.format(
                        i, iterable_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time))
                if self._async_logger is not None:
                    self._async_logger.log(log_message, log_dict)
                else:
                    if log_dict and self.wandb:
                        self.wandb.log(log_dict)
                    print(log_message)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        total_message = '{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / iterable_len)
        if self._async_logger is not None:
            self._async_logger.log(total_message, None)
        else:
            print(total_message)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    targets = batch[1]
    if not targets:
        batch[1] = {}
        return tuple(batch)
    if isinstance(targets, dict):
        return tuple(batch)

    per_image_keys = {"image_id", "orig_size", "size"}
    lengths = []
    for t in targets:
        if "labels" in t:
            lengths.append(int(t["labels"].shape[0]))
        elif "boxes" in t:
            lengths.append(int(t["boxes"].shape[0]))
        elif "masks" in t:
            lengths.append(int(t["masks"].shape[0]))
        else:
            lengths.append(0)

    bs = len(targets)
    max_len = max(max(lengths), 1)
    lengths_tensor = torch.as_tensor(lengths, dtype=torch.int64)

    padded = {"lengths": lengths_tensor}

    if "boxes" in targets[0]:
        boxes = [t["boxes"] for t in targets]
        box_shape = boxes[0].shape[1:]
        boxes_padded = torch.zeros(
            (bs, max_len, *box_shape),
            dtype=boxes[0].dtype,
            device=boxes[0].device,
        )
        for i, b in enumerate(boxes):
            n = b.shape[0]
            if n > 0:
                boxes_padded[i, :n] = b
        padded["boxes"] = boxes_padded

    if "labels" in targets[0]:
        labels = [t["labels"] for t in targets]
        label_shape = labels[0].shape[1:]
        labels_padded = torch.full(
            (bs, max_len, *label_shape),
            -1,
            dtype=labels[0].dtype,
            device=labels[0].device,
        )
        for i, l in enumerate(labels):
            n = l.shape[0]
            if n > 0:
                labels_padded[i, :n] = l
        padded["labels"] = labels_padded

    if "area" in targets[0]:
        areas = [t["area"] for t in targets]
        area_shape = areas[0].shape[1:]
        areas_padded = torch.zeros(
            (bs, max_len, *area_shape),
            dtype=areas[0].dtype,
            device=areas[0].device,
        )
        for i, a in enumerate(areas):
            n = a.shape[0]
            if n > 0:
                areas_padded[i, :n] = a
        padded["area"] = areas_padded

    if "iscrowd" in targets[0]:
        crowds = [t["iscrowd"] for t in targets]
        crowd_shape = crowds[0].shape[1:]
        crowds_padded = torch.zeros(
            (bs, max_len, *crowd_shape),
            dtype=crowds[0].dtype,
            device=crowds[0].device,
        )
        for i, c in enumerate(crowds):
            n = c.shape[0]
            if n > 0:
                crowds_padded[i, :n] = c
        padded["iscrowd"] = crowds_padded

    if "masks" in targets[0]:
        masks = [t["masks"] for t in targets]
        max_h = max(m.shape[-2] for m in masks)
        max_w = max(m.shape[-1] for m in masks)
        masks_padded = torch.zeros(
            (bs, max_len, max_h, max_w),
            dtype=masks[0].dtype,
            device=masks[0].device,
        )
        for i, m in enumerate(masks):
            n = m.shape[0]
            if n > 0:
                h, w = m.shape[-2:]
                masks_padded[i, :n, :h, :w] = m
        padded["masks"] = masks_padded

    for key in per_image_keys:
        if key in targets[0]:
            padded[key] = torch.stack([t[key] for t in targets], dim=0)

    batch[1] = padded
    return tuple(batch)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        # type: (Device, bool) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def pin_memory(self):
        tensors = self.tensors.pin_memory()
        mask = self.mask.pin_memory() if self.mask is not None else None
        return NestedTensor(tensors, mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        sizes = [list(img.shape) for img in tensor_list]
        max_size = _max_by_axis(sizes)
        if all(size == max_size for size in sizes):
            return NestedTensor(torch.stack(tensor_list, dim=0), None)
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError('not supported')
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(obj, f, *args, **kwargs):
    """
    Safely save objects, removing any callbacks that can't be pickled
    """
    if is_main_process():
        torch.save(obj, f, *args, **kwargs)

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__.split(".")[1]) < 7.0:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def strip_checkpoint(checkpoint):
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=False)
    new_state_dict = {
        'model': state_dict['model'],
        'args': state_dict['args'],
    }
    torch.save(new_state_dict, checkpoint)
