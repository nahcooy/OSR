# vit_training/utils.py
import json
import os
from pathlib import Path
import importlib
import torch
import pandas as pd
from datetime import datetime

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: No GPU available, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: Requested {n_gpu_use} GPUs, but only {n_gpu} are available.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        if self.writer is not None:
            self.writer.add_scalar(key, value)
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def result(self):
        return dict(self._data.average)

class TensorboardWriter:
    def __init__(self, log_dir, enabled):
        self.writer = None
        self.selected_module = ""
        if enabled:
            log_dir = str(log_dir)
            succeeded = False
            for module in ["tensorboardX"]:
                try:
                    self.writer = importlib.import_module(module).SummaryWriter(log_dir)
                    succeeded = True
                    break
                except ImportError:
                    succeeded = False
                self.selected_module = module
            if not succeeded:
                print("Warning: Tensorboard not installed. Install with 'pip install tensorboardx' or disable in config.")
        self.step = 0
        self.mode = ''
        self.tb_writer_ftns = {'add_scalar', 'add_scalars', 'add_image', 'add_images', 'add_audio', 'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding'}
        self.tag_mode_exceptions = {'add_histogram', 'add_embedding'}
        self.timer = datetime.now()

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar('steps_per_sec', 1 / duration.total_seconds())
            self.timer = datetime.now()

    def __getattr__(self, name):
        if name in self.tb_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    if name not in self.tag_mode_exceptions:
                        tag = f'{tag}/{self.mode}'
                    if name == 'add_embedding':
                        add_data(tag=tag, mat=data, global_step=self.step, *args, **kwargs)
                    else:
                        add_data(tag, data, self.step, *args, **kwargs)
            return wrapper
        else:
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError(f"type object '{self.selected_module}' has no attribute '{name}'")
            return attr

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k / batch_size * 100.0)
    return res