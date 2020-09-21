from datasets.dataset import BaseDataset
from pathlib import Path
from tqdm import tqdm
import torch
import json
from PIL import Image
import numpy as np
from torchvision import transforms

def Structured3DLoader(path, split, batch_size, img_size, n_workers, cache):
    return torch.utils.data.DataLoader(
        Structured3DDataset(path=path, split=split, img_size=img_size, cache=cache), 
        batch_size=batch_size, 
        shuffle=split=='train', 
        num_workers=n_workers,
        pin_memory=True        
        )


class Structured3DDataset(BaseDataset):
    def __init__(self, path, img_size, cache=False, *args, **kwargs):
        super(Structured3DDataset, self).__init__(*args, **kwargs)
        self.path = path
        self.img_size = img_size
        self.load_scene_names()
        self.load_images()
        self.cache = []
        if cache:
            self.cache_images()

    def load_image(self, index):
        rgb_path = self.images[index]
        depth_path = rgb_path.replace("rgb_rawlight", "depth")
        rgb_raw = Image.open(rgb_path).convert("RGB")
        depth_raw = Image.open(depth_path)
        depth_np = np.array(depth_raw, dtype=np.uint16)
        depth_np = depth_np / 1000.0 # mm -> meter
        depth_np = np.clip(depth_np, 0.0, 10.0) # clamp to range [0,10] meters
        depth_np /= 10.0 # normalize to values between 0 and 1
        depth_np *= 255.0 # 0 .. 255
        depth_np = depth_np.astype(np.uint8)
        depth_raw = Image.fromarray(depth_np)
        return rgb_raw, depth_raw

    def get_raw(self, index):
        if self.cache:
            return self.cache[index]
        else:
            return self.load_image(index)

    def load_scene_names(self):
        if self.split == 'train':
            self.scene_names = [d.stem for d in Path(self.path).glob("*") if d.is_dir()][0:3000]
        else:
            self.scene_names = [d.stem for d in Path(self.path).glob("*") if d.is_dir()][3000:]

    def cache_images(self):
        self.cache = []
        for i, _ in tqdm(enumerate(self.images), desc="Caching images"):
            self.cache.append(self.load_image(i))

    def load_images(self):
        self.images = []
        for scene_name in tqdm(self.scene_names, desc="Loading image paths"):
            scene_directory = Path(self.path)/scene_name
            self.images += [img.as_posix() for img in scene_directory.glob("**/*") if "rgb_rawlight" in img.name and img.parents[2].name == "perspective"]
        print("Found {} images.".format(self.__len__()))

    def preprocess_training(self, raw_data):
        rgb_pil, depth_pil = raw_data

        transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
            transforms.RandomVerticalFlip(0.5),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(degrees=5),
            transforms.ToTensor()
        ])

        rgb_t = transform(transforms.ColorJitter(0.4, 0.4, 0.4)(rgb_pil)).to(torch.float16)
        depth_t= transform(depth_pil).to(torch.float16)

        return rgb_t, depth_t

    def preprocess_validation(self, raw_data):
        rgb_pil, depth_pil = raw_data

        transform = transforms.Compose([
            transforms.Resize(self.img_size, interpolation=Image.BILINEAR),
            transforms.ToTensor()
        ])

        rgb_t = transform(rgb_pil).to(torch.float16)
        depth_t= transform(depth_pil).to(torch.float16)

        return rgb_t, depth_t

if __name__ == "__main__":
    def make_relative(m):
        return (m - np.min(m)) / (np.max(m) - np.min(m))
    def show(name, mat):
        cv2.imshow(name, make_relative(mat))
    import cv2
    depth = cv2.imread("D:/Documents/data/Structured3D/Structured3D/scene_00000/2D_rendering/490854/perspective/full/2/depth.png", cv2.IMREAD_ANYDEPTH)
    depth_log = np.log(depth) + 1
    min_ = np.min(depth_log)
    max_ = np.max(depth_log)
    depth_log = (depth_log - min_) / (max_ - min_)
    depth_grad = np.abs(np.gradient(depth))
    #depth_grad = make_relative(depth_grad)
    show("depth_grad0", depth_grad[0])
    show("depth_grad1", depth_grad[1])
    show("depth_grad_sum", depth_grad[0] + depth_grad[1])
    cv2.waitKey(0)
    
    