from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np

from torchvtk.datasets import TorchDataset
from torchvtk.utils import make_3d

from datasets.dataset import BaseDataset

def get_stdepth_dataset(args, split, output_size, resize):
    return SemiTransparentDepthDataset(args.path, split=split, output_size=output_size, resize=resize, depth_method=args.depth_method)


class SemiTransparentDepthDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    def __init__(self, path, resize, output_size, depth_method='first_hit', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize = resize
        self.output_size = output_size
        self.path = path
        self.torch_ds = TorchDataset(path)
        self.depth_method = depth_method

    def training_preprocess(self, rgb, depth):
        #depth = transforms.ToPILImage()(depth)
        s = np.random.uniform(1, 1.5)
        # color jitter
        rgb = transforms.ColorJitter(0.4, 0.4, 0.4)(rgb)
        # Resize
        resize = transforms.Resize(self.resize)
        rgb = resize(rgb)
        depth = resize(depth)
        # Random Rotation
        angle = np.random.uniform(-5,5)
        rgb = TF.rotate(rgb, angle)
        depth = TF.rotate(depth, angle)
        # Resize
        resize = transforms.Resize(int(self.resize * s))
        rgb = resize(rgb)
        depth = resize(depth)
        # Center crop
        crop = transforms.CenterCrop(self.output_size)
        rgb = crop(rgb)
        depth = crop(depth)
        # Random horizontal flipping
        if np.random.uniform(0,1) > 0.5:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)
        # Transform to tensor
        rgb = TF.to_tensor(rgb)
        depth = TF.to_tensor(depth / s)
        return rgb, depth

    def validation_preprocess(self, rgb, depth):
        return rgb, depth

    def test_preprocess(self, rgb, depth):
        return rgb, depth

    def get_raw(self, index):
        item = self.torch_ds[index]
        return torch.clamp(item['rgba'][:3].float() * 255.0, 0.0, 255.0), item[self.depth_method].float() * 10.0

    def __len__(self):
        return len(self.torch_ds)
    
    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = parent_parser.add_parser('stdepth')
        BaseDataset.add_dataset_specific_args(parser)
        parser.add_argument('--depth-method', type=str, default='first_hit', help='Depth method. first_hit, max_opacity, max_gradient, wysiwyg')
