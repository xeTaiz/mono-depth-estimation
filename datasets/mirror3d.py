from pathlib import Path
import numpy as np
from PIL import Image
from datasets.dataset import BaseDataset
from enum import Enum
import cv2
import csv

def get_mirror3d_dataset(args, split, output_size, resize):
    return Mirror3DDataset(args.path, split=split, output_size=output_size, resize=resize, dataset_type=args.type)

class DatasetType(Enum):
    NYU = 'nyu'
    MP3D = 'mp3d'
    SCANNET = 'scannet'
    ALL = 'all'

class Mirror3DDataset(BaseDataset):
    def __init__(self, path, dataset_type, output_size, resize, *args, **kwargs):
        super(Mirror3DDataset, self).__init__(*args, **kwargs)
        self.path = Path(path)
        self.output_size = output_size
        self.resize = resize
        self.dataset_type = DatasetType(dataset_type)
        self.load_images()
        print("Found {} images for {}".format(self.__len__(), self.split))        

    def load_images(self):
        datasets = ['nyu', 'mp3d', 'scannet'] if self.dataset_type == DatasetType.ALL else [self.dataset_type.value]
        self.images = []
        for dataset in datasets:
            with open(self.path/dataset/"{}.csv".format(dataset), "r" , newline='') as csvfile:
                spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                sample_ids = [row[1].replace("_i", "_d") for row in spamreader if row[3] == self.split]
            self.images  += [f for f in Path(self.path, dataset, "refined_sensorD_precise").glob("**/*") if f.suffix == ".png" if f.stem in sample_ids]

    def get_rgb_path(self, index):
        depth_path = self.images[index]
        rgb_path = depth_path.as_posix().replace("refined_sensorD_precise", "mirror_color_images").replace(".png", ".jpg")

        if "mp3d" in rgb_path:
            rgb_path = rgb_path.replace("_d", "_i")

        return rgb_path
        
    def get_raw(self, index):
        depth_path = self.images[index]
        img_path = self.get_rgb_path(index)
        assert depth_path.exists(), depth_path.as_posix()
        assert Path(img_path).exists(), img_path
        rgb = Image.open(img_path).convert('RGB')
        depth = Image.open(depth_path)
        depth = np.array(depth, dtype=np.float32)
        depth /= 1000 
        depth = np.clip(depth, 0, 10)
        return rgb, depth

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = parent_parser.add_parser('mirror3d')
        parser.add_argument('--type', required=True, type=str, help="Mirror3D type [nyu, mp3d, scannet, all]")
        BaseDataset.add_dataset_specific_args(parser)

if __name__ == "__main__":
    import cv2
    dataset = Mirror3DDataset(path="H:/data/mirror3d", split="test", dataset_type=DatasetType.ALL, output_size=(512,512), resize=512)
    
    #Mirror3DDataset(path="H:/data/mirror3d", split="valid", dataset_type=DatasetType.MP3D, output_size=(512,512), resize=512)
    #zipfiles = [f for f in Path("H:/data/matterport3d/v1/scans").glob('**/*') if f.suffix == ".zip" and 'undistorted_color_images' in f.name]
    #for zf in tqdm(zipfiles):
    #    with zipfile.ZipFile(zf, "r") as zip_ref:
    #        zip_ref.extractall("H:/data/matterport3d/v1/scans/undistorted_color_images")

    #zipfiles = [f for f in Path("H:/data/matterport3d/v1/scans").glob('**/*') if f.suffix == ".zip" and 'undistorted_depth_images' in f.name]
    #for zf in tqdm(zipfiles):
    #    with zipfile.ZipFile(zf, "r") as zip_ref:
    #        zip_ref.extractall("H:/data/matterport3d/undistorted_depth_images")
