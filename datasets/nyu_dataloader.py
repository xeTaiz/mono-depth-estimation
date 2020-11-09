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

def my_hook(t):
    last_b = [0]
    def update_to(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return update_to

def download_split(filename):
    # start download
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading NYU split.mat: {}".format(filename.as_posix())) as t:
        urllib.request.urlretrieve("http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat", filename = filename, reporthook = my_hook(t), data = None)

def download_nyu_depth_v2_labeled(filename):
    # start download
    with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading NYU nyu_depth_v2_labeled.mat: {}".format(filename.as_posix())) as t:
        urllib.request.urlretrieve("http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat", filename = filename, reporthook = my_hook(t), data = None)

def h5_loader(path, data):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth

def mat_loader(index, data):
    data = h5py.File(data, "r")    
    rgb = data['images'][index]
    depth = data['depths'][index]
    rgb = np.transpose(rgb, (2, 1, 0))
    depth = np.transpose(depth, (1,0))
    data.close()
    return rgb, depth

class NYUDataset(BaseDataset):
    def __init__(self, path, output_size=(228, 304), resize=250, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)
        self.output_size = output_size
        self.resize = resize
        self.nyu_depth_v2_labeled_file = None
        if 'train' in self.split:
            self.loader = h5_loader
            self.path = Path(path)/'train'
            self.images = [path.as_posix() for path in self.path.glob("**/*") if path.name.endswith('.h5')]
            if '12k' in self.split:
                self.images = self.images[0:12000]
        elif self.split in ['val', 'test']:
            self.path = Path(path)
            self.loader = mat_loader
            self.images = self.load_images()
        assert len(self.images) > 0, "Found 0 images in subfolders of: " + path + "\n"
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

    def get_raw(self, index):
        path = self.images[index]
        rgb, depth = self.loader(path, self.nyu_depth_v2_labeled_file)
        return rgb, depth

    def load_images(self):
        self.nyu_depth_v2_labeled_file = (self.path/"nyu_depth_v2_labeled.mat")
        self.split_file = (self.path/"split.mat")
        if not self.nyu_depth_v2_labeled_file.exists(): download_nyu_depth_v2_labeled(self.nyu_depth_v2_labeled_file)
        if not self.split_file.exists(): download_split(self.split_file)
        return np.hstack(loadmat(self.split_file)['trainNdxs' if self.split == 'train' else 'testNdxs']) - 1

if __name__ == "__main__":
    import visualize
    nyu = NYUDataset("G:/data/nyudepthv2", split="val")
    item = nyu.__getitem__(0)
    visualize.show_item(item)