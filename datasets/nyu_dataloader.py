from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from datasets.dataset import BaseDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
import urllib.request
from scipy.io import loadmat
import visualize

NYU_V2_SPLIT_MAT_URL = 'http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat'
NYU_V2_MAPPING_40_URL = 'https://github.com/ankurhanda/nyuv2-meta-data/raw/master/classMapping40.mat'
NYU_V2_LABELED_MAT_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'

MIRROR_IDX = [25, 26, 76, 77, 86, 102, 131, 161, 162, 171, 172, 194, 195, 196, 199, 259, 266, 267, 268, 269, 271, 272, 273, 276, 277, 282, 283, 285, 286, 287, 290, 292, 294, 299, 302, 303, 305, 306, 308, 310, 313, 314, 323, 391, 401, 423, 427, 435, 440, 445, 457, 458, 487, 496, 505, 579, 583, 585, 586, 606, 609, 612, 613, 619]

def my_hook(t):
    last_b = [0]
    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return update_to

def download(filename, url):
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading: {}".format(filename.name)) as t:
        urllib.request.urlretrieve(url, filename = filename, reporthook = my_hook(t), data = None)

class NYUDataset(BaseDataset):
    def __init__(self, path, output_size=(228, 304), resize=250, use_mat=False, n_images=-1, exclude_mirrors=False, mirrors_only=False, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)
        self.output_size = output_size
        self.resize = resize
        self.nyu_depth_v2_labeled_file = None
        self.use_mat = exclude_mirrors or mirrors_only or use_mat
        if not use_mat:
            self.path = Path(path)/('train' if 'train' in self.split else 'val')
            self.images = [path.as_posix() for path in self.path.glob("**/*") if path.name.endswith('.h5')]            
        else:
            self.path = Path(path)
            self.images = self.load_images()
            self.mapping40 = np.insert(loadmat(self.mapping40_file)['mapClass'][0], 0, 0)
        assert len(self.images) > 0, "Found 0 images in subfolders of: " + path + "\n"
        if exclude_mirrors: self.images = self.images[[idx for idx in np.arange(0, len(self.images)) if not idx in MIRROR_IDX]]
        if mirrors_only: self.images = self.images[[idx for idx in np.arange(0, len(self.images)) if idx in MIRROR_IDX]]
        if n_images > 0: self.images = self.images[0:n_images]
        print("Found {} images in {} folder.".format(len(self.images), self.split))

    def training_preprocess(self, rgb, depth):
        s = np.random.uniform(1, 1.5)
        depth = depth / s

        rgb = transforms.ToPILImage()(rgb)
        depth = transforms.ToPILImage()(depth)
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
        depth = TF.to_tensor(np.array(depth))
        return rgb, depth

    def validation_preprocess(self, rgb, depth):
        rgb = transforms.ToPILImage()(rgb)
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
        depth = TF.to_tensor(np.array(depth))
        return rgb, depth

    def test_preprocess(self, rgb, depth):
        rgb = transforms.ToPILImage()(rgb)
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
        path = self.images[index]
        if self.use_mat:
            return self.mat_loader(path)
        else:
            return self.h5_loader(path)

    def load_images(self):
        self.nyu_depth_v2_labeled_file = (self.path/"nyu_depth_v2_labeled.mat")
        self.split_file = (self.path/"split.mat")
        self.mapping40_file = (self.path/"classMapping40.mat")
        if not self.nyu_depth_v2_labeled_file.exists(): download(self.nyu_depth_v2_labeled_file, NYU_V2_LABELED_MAT_URL)
        if not self.split_file.exists(): download(self.split_file, NYU_V2_SPLIT_MAT_URL)
        if not self.mapping40_file.exists(): download(self.mapping40_file, NYU_V2_MAPPING_40_URL)
        return np.hstack(loadmat(self.split_file)['trainNdxs' if self.split == 'train' else 'testNdxs']) - 1

    def h5_loader(self, path):
        h5f = h5py.File(path, "r")
        rgb = np.array(h5f['rgb'])
        rgb = np.transpose(rgb, (1, 2, 0))
        depth = np.array(h5f['depth'])
        return rgb, depth

    def mat_loader(self, index):
        data = h5py.File(self.nyu_depth_v2_labeled_file, "r")  
        rgb = data['images'][index]
        depth = data['depths'][index]
        labels = data['labels'][index]
        
        rgb = np.transpose(rgb, (2, 1, 0))
        depth = np.transpose(depth, (1,0))
        labels = np.transpose(labels, (1,0))
        labels_40 = self.mapping40[labels]
        # mask = labels_40 == 19 Mirrors
        return rgb, depth

if __name__ == "__main__":
    import visualize
    nyu = NYUDataset("G:/data/nyudepthv2", split="val", use_mat=True, mirrors_only=True)
    for item in nyu:
        visualize.show_item(item)
    
    
    