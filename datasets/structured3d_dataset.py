import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from datasets.dataset import BaseDataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm

def get_structured3d_dataset(args, split, output_size, resize):
    return Structured3DDataset(args.path, split=split, output_size=output_size, resize=resize, dataset_type=args.type)

class Structured3DDataset(BaseDataset):
    def __init__(self, path, dataset_type='perspective', output_size=(360, 640), resize=400, *args, **kwargs):
        super(Structured3DDataset, self).__init__(*args, **kwargs)
        assert dataset_type in ['perspective', 'panorama','panorama_empty', 'panorama_simple', 'panorama_full']
        self.dataset_type = dataset_type
        self.output_size = output_size
        self.resize = resize
        self.path = path
        self.load_scene_names()
        self.load_images()

    def load_scene_names(self):
        if self.split == 'train':
            self.scene_names = [d.stem for d in Path(self.path).glob("*") if d.is_dir()][0:3400]
        elif self.split == 'val':
            self.scene_names = [d.stem for d in Path(self.path).glob("*") if d.is_dir()][3400:3450]
        else:
            self.scene_names = [d.stem for d in Path(self.path).glob("*") if d.is_dir()][3450:]

    def load_images(self):
        self.images = []
        for scene_name in tqdm(self.scene_names, desc="Loading image paths"):
            scene_directory = Path(self.path)/scene_name
            self.images += [img.as_posix() for img in scene_directory.glob("**/*") if "rgb_rawlight" in img.name and self.dataset_type.split('_')[-1] in img.as_posix()]
        print("Found {} images.".format(self.__len__()))

    def get_raw(self, index):
        rgb_path = self.images[index]
        depth_path = rgb_path.replace("rgb_rawlight", "depth")
        rgb = Image.open(rgb_path).convert('RGB')
        depth = Image.open(depth_path)
        depth = np.array(depth, dtype=np.float32)
        depth /= 1000 
        depth = np.clip(depth, 0, 10)
        return rgb, depth

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = parent_parser.add_parser('structured3d')
        parser.add_argument('--type', required=True, type=str, help="Structured3D type [perspective, panorama]")
        BaseDataset.add_dataset_specific_args(parser)
    