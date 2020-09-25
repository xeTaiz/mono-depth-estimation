# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 19:53
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com


import torch
import math
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.metrics.metric import TensorMetric

class MetricLogger(object):
    def __init__(self, metrics):
        self.computer = MetricComputation(metrics)
    
    def log_train(self, pred, target, loss):
        values = self.computer.compute(pred, target)
        result = pl.TrainResult(loss)
        for name, value in zip(self.computer.names, values):
            result.log(name, value, logger=True, on_epoch=True)
            result.log(name+"(AVG)", self.computer.avg(name), logger=False, prog_bar=True)
        return result

    def log_val(self, pred, target, checkpoint_on=None):
        values = self.computer.compute(pred, target)
        if checkpoint_on:
            result = pl.EvalResult(checkpoint_on=self.computer.avg(checkpoint_on))
        else:
            result = pl.EvalResult()
        for name, value in zip(self.computer.names, values):
            result.log(name, value, logger=True, on_epoch=True)
            result.log(name+"(AVG)", self.computer.avg(name), logger=False, prog_bar=True)
        return result

    def reset(self):
        self.computer.reset()

class MetricComputation(object):
    def __init__(self, metrics):
        self.names = metrics
        self.metrics = [METRICS[m] for m in metrics]
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = [0.0 for _ in self.metrics]

    def compute(self, pred, target):
        current_values = [metric(pred.data, target.data) for metric in self.metrics]
        self.count += 1
        for i, value in enumerate(current_values):
            self.sum[i] += value
        return current_values

    def avg(self, metric):
        if isinstance(metric, int): return self.sum[metric] / self.count
        if isinstance(metric, str): return self.sum[self.names.index(metric)] / self.count
        assert False, "metric must be int or str"

class Delta(TensorMetric):
    def __init__(self, exp=1, *args, **kwargs):
        super(Delta, self).__init__(*args, **kwargs)
        self.exp = exp

    def forward(self, pred, target):
        maxRatio = torch.max(pred / target, target / pred)
        return (maxRatio < 1.25 ** self.exp).float().mean()

class Log10(TensorMetric):
    def log10(self, x):
        return torch.log(x) / torch.log(torch.tensor(10.0))
    def forward(self, pred, target):
        return (self.log10(pred) - self.log10(target)).abs().mean()

METRICS = pl.metrics.functional.__dict__
METRICS['delta1'] = Delta(exp=1, name="delta1")
METRICS['delta2'] = Delta(exp=2, name="delta2")
METRICS['delta3'] = Delta(exp=3, name="delta3")
METRICS['log10'] = Log10(name="log10")