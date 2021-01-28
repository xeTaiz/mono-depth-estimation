from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from datasets.dataset import BaseDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
from enum import Enum
import json

class DatasetType(Enum):
    NO_REFLECTION = '0'
    ISOTROPIC_MATERIAL = '1'
    ISOTROPIC_PLANAR_SURFACES = '2'

class Floorplan3DDataset(BaseDataset):
    def __init__(self, path, datast_type, output_size, resize, n_images=-1, *args, **kwargs):
        super(Floorplan3DDataset, self).__init__(*args, **kwargs)
        self.path = Path(path)
        self.output_size = output_size
        self.resize = resize
        self.dataset_type = datast_type.value
        self.n_images = n_images
        self.load_scene_names()
        self.load_images()
        print("Found {} scenes containing {} images for {}".format(len(self.scene_names),self.__len__(), self.split))

    def load_cubicasa_split(self):
        split_file = Path(self.path, "{}.txt".format(self.split))
        assert split_file.exists(), "Missing cubicasa split file: {}".format(split_file.as_posix())
        with open(split_file.as_posix(), "r") as txt_file:
            lines = txt_file.readlines()
            return [line.split("/")[2] for line in lines]

    def load_scene_names(self):
        scene_names = self.load_cubicasa_split()
        self.scene_names = [scene for scene in self.path.glob('*/*') if scene.name in scene_names]
        

    def load_images(self):
        self.images = []
        self.depth = []
        for scene_name in self.scene_names:
            images = [f for f in scene_name.glob('**/*') if all([s in f.name for s in ['color', '.jpg']]) and self.dataset_type in f.parent.name]
            for img_path in images:
                depth_path = img_path.parent/img_path.name.replace('color', 'depth').replace('jpg', 'png')
                if img_path.exists() and depth_path.exists():
                    self.images.append(img_path)
                    self.depth.append(depth_path)
        if self.n_images > 0: self.images = self.images[0:self.n_images]
                
    def safe_txt(self, focal):
        mapping = ["noreflection", "reflection", "mirror"]
        txt_file_path = self.path/"{}.{}.txt".format(mapping[int(self.dataset_type)], self.split)
        with open(txt_file_path.as_posix(), "w") as txt_file:
            for image, depth in zip(self.images, self.depth):
                txt_file.write("{} {} {}\n".format(image.relative_to(self.path).as_posix(), depth.relative_to(self.path).as_posix(), focal))

    def safe_json(self):
        mapping = ["noreflection", "reflection", "mirror"]
        json_file_path = self.path/"{}.{}.json".format(mapping[int(self.dataset_type)], self.split)
        with open(json_file_path.as_posix(), "w") as json_file:
            data = []
            for image, depth in zip(self.images, self.depth):
                data.append({
                    'rgb_path': image.relative_to(self.path).as_posix(),
                    'depth_path': depth.relative_to(self.path).as_posix(),
                })
            json.dump(data, json_file)

    def training_preprocess(self, rgb, depth):
        depth = transforms.ToPILImage()(depth)
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
        depth = depth / s
        depth = TF.to_tensor(depth)
        return rgb, depth

    def validation_preprocess(self, rgb, depth):
        depth = transforms.ToPILImage()(depth)
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
        depth = TF.to_tensor(depth)
        return rgb, depth

    def test_preprocess(self, rgb, depth):
        depth = transforms.ToPILImage()(depth)
   
        # Resize
        resize = transforms.Resize(500)
        rgb = resize(rgb)
        depth = resize(depth)
        # Center crop
        crop = transforms.CenterCrop((480, 640))
        rgb = crop(rgb)
        depth = crop(depth)
        # Resize
        resize = transforms.Resize(self.output_size)
        rgb = resize(rgb)
        depth = resize(depth)
        # Transform to tensor
        rgb = TF.to_tensor(np.array(rgb))
        depth = TF.to_tensor(np.array(depth))
        return rgb, depth

    def get_raw(self, index):
        img_path = self.images[index]
        depth_path = self.depth[index]
        rgb = Image.open(img_path).convert('RGB')
        depth = Image.open(depth_path)
        depth = np.array(depth, dtype=np.float32)
        depth /= 1000 
        depth = np.clip(depth, 0, 10)
        return rgb, depth

if __name__ == "__main__":
    Floorplan3DDataset(path="/mnt/hdd/shared_datasets/Floorplan3D", split="train", datast_type=DatasetType.NO_REFLECTION, output_size=(512,512), resize=512).safe_json()
    #Floorplan3DDataset(path="/mnt/hdd/shared_datasets/Floorplan3D", split="val", datast_type=DatasetType.NO_REFLECTION, output_size=(512,512), resize=512).safe_json()
    #Floorplan3DDataset(path="/mnt/hdd/shared_datasets/Floorplan3D", split="test", datast_type=DatasetType.NO_REFLECTION, output_size=(512,512), resize=512).safe_json()

    #Floorplan3DDataset(path="/mnt/hdd/shared_datasets/Floorplan3D", split="train", datast_type=DatasetType.ISOTROPIC_MATERIAL, output_size=(512,512), resize=512).safe_json()
    #Floorplan3DDataset(path="/mnt/hdd/shared_datasets/Floorplan3D", split="val", datast_type=DatasetType.ISOTROPIC_MATERIAL, output_size=(512,512), resize=512).safe_json()
    #Floorplan3DDataset(path="/mnt/hdd/shared_datasets/Floorplan3D", split="test", datast_type=DatasetType.ISOTROPIC_MATERIAL, output_size=(512,512), resize=512).safe_json()

    #Floorplan3DDataset(path="/mnt/hdd/shared_datasets/Floorplan3D", split="train", datast_type=DatasetType.ISOTROPIC_PLANAR_SURFACES, output_size=(512,512), resize=512).safe_json()
    #Floorplan3DDataset(path="/mnt/hdd/shared_datasets/Floorplan3D", split="val", datast_type=DatasetType.ISOTROPIC_PLANAR_SURFACES, output_size=(512,512), resize=512).safe_json()
    #Floorplan3DDataset(path="/mnt/hdd/shared_datasets/Floorplan3D", split="test", datast_type=DatasetType.ISOTROPIC_PLANAR_SURFACES, output_size=(512,512), resize=512).safe_json()

    