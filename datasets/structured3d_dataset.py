import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from datasets.dataset import BaseDataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def PILLoader(file):
    assert Path(file).exists(), "file not found: {}".format(file)
    return Image.open(file).convert('RGB')

def DepthLoader(file):
    # loads depth map D from png file
    assert Path(file).exists(), "file not found: {}".format(file)
    depth_png = np.array(Image.open(file), dtype=np.float32) # uint16 --> [0,65535] in millimeters
    depth = depth_png/1000 #conversion to meters [0, 65.535]
    np.clip(depth, 0, 10, depth) # clip to range[0..10] in meters
    depth /= 10.0 #normalize to range [0,1]
    return depth


class Structured3DDataset(BaseDataset):
    def __init__(self, path, dataset_type='perspective', *args, **kwargs):
        super(Structured3DDataset, self).__init__(*args, **kwargs)
        assert dataset_type in ['perspective', 'panorama','panorama_empty', 'panorama_simple', 'panorama_full']
        self.dataset_type = dataset_type
        self.output_size = (228, 405)
        self.path = path
        self.rgb_loader = PILLoader
        self.depth_loader = DepthLoader
        self.load_scene_names()
        self.load_images()

    def load_scene_names(self):
        if self.split == 'train':
            self.scene_names = [d.stem for d in Path(self.path).glob("*") if d.is_dir()][0:3000]
        else:
            self.scene_names = [d.stem for d in Path(self.path).glob("*") if d.is_dir()][3000:]

    def load_images(self):
        self.images = []
        for scene_name in tqdm(self.scene_names, desc="Loading image paths"):
            scene_directory = Path(self.path)/scene_name
            self.images += [img.as_posix() for img in scene_directory.glob("**/*") if "rgb_rawlight" in img.name and self.dataset_type.split('_')[-1] in img.as_posix()]
        print("Found {} images.".format(self.__len__()))
        
    def training_preprocess(self, rgb, depth):
        s = np.random.uniform(1, 1.5)
        depth = depth / s

        depth = transforms.ToPILImage()(depth)
        # color jitter
        rgb = transforms.ColorJitter(0.4, 0.4, 0.4)(rgb)
        # Resize
        resize = transforms.Resize(250)
        rgb = resize(rgb)
        depth = resize(depth)
        # Random Rotation
        angle = np.random.uniform(-5,5)
        rgb = TF.rotate(rgb, angle)
        depth = TF.rotate(depth, angle)
        # Resize
        resize = transforms.Resize(int(250 * s))
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
        depth = TF.to_tensor(np.array(depth))
        return rgb, depth

    def validation_preprocess(self, rgb, depth):
        rgb = transforms.ToPILImage()(rgb)
        depth = transforms.ToPILImage()(depth)
        # Resize
        resize = transforms.Resize(250)
        rgb = resize(rgb)
        depth = resize(depth)
        # Center crop
        crop = transforms.CenterCrop(self.output_size)
        rgb = crop(rgb)
        depth = crop(depth)
        # Transform to tensor
        rgb = TF.to_tensor(np.array(rgb))
        depth = TF.to_tensor(np.array(depth))
        return rgb, depth

    def get_raw(self, index):
        rgb_path = self.images[index]
        depth_path = rgb_path.replace("rgb_rawlight", "depth")
        return self.rgb_loader(rgb_path), self.depth_loader(depth_path)
    