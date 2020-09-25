import numpy as np
import dataloaders.transforms as transforms
import torch.utils.data as data
import os
from PIL import Image

iheight, iwidth = 720, 1280  # raw image size

def PILLoader(file):
    assert os.path.exists(file), "file not found: {}".format(file)
    return np.asarray(Image.open(file).convert('RGB'), dtype=np.uint8)

def DepthLoader(file):
    # loads depth map D from png file
    assert os.path.exists(file), "file not found: {}".format(file)
    depth_png = np.array(Image.open(file), dtype=np.uint16)
    depth = depth_png.astype(np.float32) / ((2**16) - 1)
    depth *= 10.0 # rescale to range[0..10]
    return depth

to_tensor = transforms.ToTensor()


class Floorplan3DDataset(data.Dataset):
    color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4)
    def __init__(self, root, dataset_type, split):
        self.output_size = (228, 405)
        self.root = root
        file_list = "{}/{}_{}.list".format(root, dataset_type, split)
        with open(file_list, "r") as f:
            self.imgs = f.readlines()

        self.depth_loader = DepthLoader
        self.color_loader = PILLoader

        if split == 'train':
            self.transform = self.train_transform
        elif split == 'val':
            self.transform = self.val_transform
        

    def train_transform(self, rgb, depth):       
        s = np.random.uniform(1.0, 1.5)  # random scaling
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight),  # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)  # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        
        depth_np = transform(depth_np)
    
        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        depth_np = depth
        transform = transforms.Compose([
            transforms.Resize(240.0 / iheight),
            transforms.CenterCrop(self.output_size),
        ])
        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        (path, target) = self.imgs[index].strip().split("  ")
        path = os.path.join(self.root, path)
        target = os.path.join(self.root, target)
        rgb = self.color_loader(path)
        depth = self.depth_loader(target)
        return rgb, depth

    def __getitem__(self, index):
        rgb, depth = self.__getraw__(index)
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise (RuntimeError("transform not defined"))

        input_tensor = to_tensor(rgb_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor

    def __len__(self):
        return len(self.imgs)
