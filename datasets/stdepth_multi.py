from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np

from torchvtk.datasets import TorchDataset
from torchvtk.utils import make_3d

from datasets.dataset import BaseDataset
from stdepth_utils import depth_sort, composite_layers

def get_stdepthmulti_dataset(args, split, output_size, resize):
    if split == 'train':
        filter_fn = lambda fn: int(fn.name.split('_')[0].split('-')[-1]) < 400
    elif split == 'val':
        filter_fn = lambda fn: 400 <= int(fn.name.split('_')[0].split('-')[-1]) < 450
    elif split == 'test':
        filter_fn = lambda fn: 450 <= int(fn.name.split('_')[0].split('-')[-1])
    else:
        raise Exception(f'Invalid split: {split}. Either train, val or test')
    return SemiTransparentMultiDepthDataset(args.path, split=split, output_size=output_size, filter_fn=filter_fn, resize=resize, single_layer=args.single_layer, set_bg_depth=args.background_depth_max)


class SemiTransparentMultiDepthDataset(BaseDataset):
    def __init__(self):
        super().__init__()

    def __init__(self, path, resize, output_size, filter_fn=lambda _: True, single_layer=False, set_bg_depth=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.resize = resize
        self.output_size = output_size
        self.path = path
        self.torch_ds = TorchDataset(path, filter_fn=filter_fn)
        self.set_bg_depth = set_bg_depth
        self.single_layer = single_layer

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
        return TF.to_tensor(rgb), TF.to_tensor(depth)

    def test_preprocess(self, rgb, depth):
        return TF.to_tensor(rgb), TF.to_tensor(depth)
    def get_raw(self, index):
        item = self.torch_ds[index]
        rgba = torch.clamp(item['rgba'].float(), 0.0, 1.0)
        l1, l2, l3 = item['layer1'].float(), item['layer2'].float(), item['layer3'].float()
        if self.set_bg_depth:
            l1[4][l1[4] == 0.0] = 1.0
            l2[4][l2[4] == 0.0] = 1.0
            l3[4][l3[4] == 0.0] = 1.0
        if self.single_layer:
            front = l1[:4]
        else:
            sorted_layers = depth_sort(torch.stack([l1, l2, l3]))
            front = composite_layers(sorted_layers)
        back_a   = torch.clamp((rgba[[3]] - front[[3]]) / (1.0 - front[[3]]), 0.0, 1.0)
        back_rgb = torch.clamp((rgba[:3]  - front[[3]]) / (1.0 - front[[3]]), 0.0, 1.0)
        if self.single_layer:
            gt = [ l1[:4], torch.cat([back_rgb, back_a], dim=0), l1[[4]], rgba[[3]] ] # 10 Channels
        else:
            gt = [
                l1[:4], l2[:4], l3[:4],                            # RGBAs     (3x4 = 12 channels)
                torch.cat([back_rgb, back_a], dim=0),              # Background Layer (4 channels)
                torch.stack([l1[4], l2[4], l3[4], rgba[3]], dim=0),# Depths & Full a  (4 channels)
            ]                                                      # Total Channels: 20

        rgb = torch.clamp(rgba[:3] * 255.0, 0.0, 255.0).byte()
        return rgb, gt

    def __len__(self):
        return len(self.torch_ds)

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = parent_parser.add_parser('stdepthmulti')
        BaseDataset.add_dataset_specific_args(parser)
        parser.add_argument('--depth-method', type=str, default='multi', help='Depth method to use')
        parser.add_argument('--single-layer', action='store_true', help='Do the multi thing but just with 1 layer')
        parser.add_argument('--background-depth-max', action='store_true', help='Whether to replace depth for background(0.0) with max depth (1.0)')
