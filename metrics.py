from pytorch_lightning.metrics.metric import TensorMetric
import torch
import math

class Delta(TensorMetric):
    def __init__(self, exponent=1.0, *args, **kwargs):
        super(Delta, self).__init__(*args, **kwargs)
        self.exponent = exponent
    def forward(self, pred, target):
        maxRatio = torch.max(pred / target, target / pred)
        return float((maxRatio < 1.25 ** self.exponent).float().mean())

class Log10(TensorMetric):
    def log10(self, x):
        """Convert a new tensor with the base-10 logarithm of the elements of x. """
        return torch.log(x) / math.log(10)
    def forward(self, pred, target):
        return float((self.log10(pred) - self.log10(target)).abs().mean())

delta1 = Delta(name="delta1", exponent=1.0)
delta2 = Delta(name="delta2", exponent=2.0)
delta3 = Delta(name="delta3", exponent=3.0)
log10 = Log10(name="log10")