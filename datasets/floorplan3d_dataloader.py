from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from datasets.dataset import BaseDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
from enum import Enum

class DatasetType(Enum):
    NO_REFLECTION = '0'
    ISOTROPIC_MATERIAL = '1'
    ISOTROPIC_PLANAR_SURFACES = '2'

class Floorplan3DDataset(BaseDataset):
    def __init__(self, path, datast_type=DatasetType.NO_REFLECTION, output_size=(360, 640), resize=400, *args, **kwargs):
        super(Floorplan3DDataset, self).__init__(*args, **kwargs)
        self.path = Path(path)
        self.output_size = output_size
        self.resize = resize
        self.dataset_type = datast_type.value
        self.load_scene_names()
        self.load_images()
        print("Found {} scenes containing {} images for {}".format(len(self.scene_names),self.__len__(), self.split))

    def load_scene_names(self):
        scene_names = [scene for scene in self.path.glob('*/*')]
        if self.split == 'train':
            self.scene_names = scene_names[0:500]
        else:
            self.scene_names = scene_names[500:]

    def load_images(self):
        self.images = []
        for scene_name in self.scene_names:
            self.images += [f for f in scene_name.glob('**/*') if all([s in f.name for s in ['color', '.jpg']]) and f.parent.name == self.dataset_type]

    def training_preprocess(self, rgb, depth):
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
        rgb = TF.to_tensor(np.array(rgb))
        depth = np.array(depth, dtype=np.float32)
        depth /= 1000 
        depth = np.clip(depth, 0, 10)
        depth = depth / s
        depth = TF.to_tensor(depth)
        return rgb, depth

    def validation_preprocess(self, rgb, depth):
        # Resize
        resize = transforms.Resize(self.resize)
        rgb = resize(rgb)
        depth = resize(depth)
        # Center crop
        crop = transforms.CenterCrop(self.output_size)
        rgb = crop(rgb)
        depth = crop(depth)
        # Transform to tensor
        rgb = TF.to_tensor(np.array(rgb))
        depth = np.array(depth, dtype=np.float32)
        depth /= 1000
        depth = np.clip(depth, 0, 10)
        depth = TF.to_tensor(depth)
        return rgb, depth

    def get_raw(self, index):
        path = self.images[index]
        rgb = Image.open(path).convert('RGB')
        depth_path = path.parent/path.name.replace('color', 'depth').replace('jpg', 'png')
        depth = Image.open(depth_path)
        return rgb, depth

if __name__ == "__main__":
    import visualize
    import cv2
    dataset = Floorplan3DDataset(path="G:/data/floorplan3d", split="train", datast_type=DatasetType.ISOTROPIC_PLANAR_SURFACES)
    img, depth = dataset.__getitem__(100)
    print(torch.min(depth))
    print(torch.max(depth))
    viz = visualize.merge_into_row(img, depth, depth)
    cv2.imshow("viz", viz.astype('uint8'))
    cv2.waitKey(0)

    