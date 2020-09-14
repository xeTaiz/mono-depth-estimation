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
        rgb_raw = Image.open(rgb_path).convert("RGB").resize(self.img_size, Image.BILINEAR)
        depth_raw = Image.open(depth_path).resize(self.img_size, Image.BILINEAR)
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
            transforms.ToTensor()
        ])

        rgb_t = transform(rgb_pil).to(torch.float16)
        depth_t= transform(depth_pil).to(torch.float16)

        return rgb_t, depth_t

if __name__ == "__main__":
    import cv2
    structured3d = Structured3DDataset(path="D:/Documents/data/Structured3D/Structured3D", img_size=(256, 256), split="valid", cache=False)
    rgb, depth = structured3d.__getitem__(0)
    rgb_np = rgb.numpy()
    rgb_np *= 255
    print(rgb_np.shape)
    rgb_np = np.transpose(rgb_np, (1,2,0))
    print(rgb_np.shape)
    rgb_np = cv2.cvtColor(rgb_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("img", rgb_np)
    cv2.waitKey(0)