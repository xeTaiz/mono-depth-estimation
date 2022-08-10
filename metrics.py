# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 19:53
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com


import torch
import torchmetrics


class MetricLogger(object):
    def __init__(self, metrics, module):
        self.context = module
        self.computer = MetricComputation(metrics)

    def log_train(self, pred, target, loss):
        values = self.computer.compute(pred, target)
        result = {"loss": loss}
        self.context.log("loss", loss)
        for name, value in zip(self.computer.names, values):
            self.context.log("train_{}".format(name), value, logger=True, on_epoch=True)
            result[name] = value
        return result

    def log_val(self, pred, target):
        values = self.computer.compute(pred, target)
        result = {}
        for name, value in zip(self.computer.names, values):
            self.context.log("val_{}".format(name), value, logger=True, on_epoch=True)
            result[name] = value
        return result

    def log_test(self, pred, target):
        values = self.computer.compute(pred, target)
        result = {}
        for name, value in zip(self.computer.names, values):
            self.context.log("{}".format(name), value, on_step=True, on_epoch=True)
            result[name] = value
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
        pred = torch.clamp_min(pred, 1e-07)
        valid_mask = target > 0
        assert torch.sum(valid_mask) > 0, "invalid target!"
        current_values = [metric(pred[valid_mask].data, target[valid_mask].data) for metric in self.metrics]
        self.count += 1
        for i, value in enumerate(current_values):
            self.sum[i] += value
        return current_values

    def avg(self, metric):
        if isinstance(metric, int): return self.sum[metric] / self.count
        if isinstance(metric, str): return self.sum[self.names.index(metric)] / self.count
        assert False, "metric must be int or str"


def Delta1_multi_gpu(pred, target):
    maxRatio = torch.max(pred / target, target / pred)
    return (maxRatio < 1.25 ** 1).float().mean()


def Delta2_multi_gpu(pred, target):
    maxRatio = torch.max(pred / target, target / pred)
    return (maxRatio < 1.25 ** 2).float().mean()


def Delta3_multi_gpu(pred, target):
    maxRatio = torch.max(pred / target, target / pred)
    return (maxRatio < 1.25 ** 3).float().mean()


def Log10_multi_gpu(pred, target):
    return (torch.log10(pred) - torch.log10(target)).abs().mean()


def AbsoluteRelativeError(pred, target):
    if (target == 0).any():
        raise NotComputableError("The ground truth has 0.")
    return (torch.abs(pred - target) / target).mean()


def RelativeSquareError(pred, target):
    if (target == 0).any():
        raise NotComputableError("The ground truth has 0.")
    return ((pred - target)**2 / target).mean()


def RelativeMeanSquareError(pred, target):
    if (target == 0).any():
        raise NotComputableError("The ground truth has 0.")
    return torch.sqrt((pred - target)**2 / target).mean()


METRICS = {}
METRICS['mse'] = torchmetrics.functional.regression.mean_squared_error
METRICS['msle'] = torchmetrics.functional.regression.mean_squared_log_error
METRICS['mae'] = torchmetrics.functional.regression.mean_absolute_error
METRICS['delta1'] = Delta1_multi_gpu  # Delta(exp=1, name="delta1")
METRICS['delta2'] = Delta2_multi_gpu  # Delta(exp=2, name="delta2")
METRICS['delta3'] = Delta3_multi_gpu  # Delta(exp=3, name="delta3")
METRICS['log10'] = Log10_multi_gpu    # Log10(name="log10")
METRICS['absrel'] = AbsoluteRelativeError
METRICS['sqrel'] = RelativeSquareError
METRICS['rmse'] = RelativeMeanSquareError
