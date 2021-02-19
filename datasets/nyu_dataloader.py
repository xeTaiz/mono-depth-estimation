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
import json
import tarfile

DATASET_TYPES = ['labeled', 'no_mirror', 'mirror', 'mirror_corrected', 'sparse_2_dense']

NYU_V2_SPLIT_MAT_URL = 'http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat'
NYU_V2_MAPPING_40_URL = 'https://github.com/ankurhanda/nyuv2-meta-data/raw/master/classMapping40.mat'
NYU_V2_LABELED_MAT_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'
NYU_V2_SPARSE2DENSE_URL = 'http://datasets.lids.mit.edu/sparse-to-dense/data/nyudepthv2.tar.gz'
NYU_V2_CORRECTED_MAT_URL = 'https://cloudstore.uni-ulm.de/s/mRwWiLCCjsC6Rkf/download'

VAL_MIRROR_IDX = [25, 26, 76, 77, 86, 102, 131, 161, 162, 171, 172, 194, 195, 196, 199, 259, 266, 267, 268, 269, 271, 272, 273, 276, 277, 282, 283, 285, 286, 287, 290, 292, 294, 299, 302, 303, 305, 306, 308, 310, 313, 314, 323, 391, 401, 423, 427, 435, 440, 445, 457, 458, 487, 496, 505, 579, 583, 585, 586, 606, 609, 612, 613, 619]
TRAIN_MIRROR_IDX = [18, 20, 21, 91, 103, 104, 128, 130, 136, 139, 142, 143, 144, 145, 208, 209, 264, 269, 305, 306, 307, 308, 309, 311, 313, 317, 381, 382, 384, 386, 387, 388, 389, 391, 392, 394, 395, 396, 398, 400, 402, 404, 405, 406, 409, 412, 413, 414, 415, 416, 418, 420, 421, 423, 425, 426, 428, 439, 441, 473, 501, 532, 559, 566, 569, 574, 587, 588, 600, 608, 613, 615, 639, 640, 665, 666, 705, 706, 743, 756, 767, 768, 769, 774, 775, 780, 781, 782, 784]
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

def get_nyu_dataset(args, split, output_size, resize):
    return NYUDataset(args.path, split=split, output_size=output_size, resize=resize, dataset_type=args.type)

class NYUDataset(BaseDataset):
    def __init__(self, path, output_size=(228, 304), resize=250, n_images=-1, dataset_type=None, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)
        assert dataset_type in DATASET_TYPES, "unknow NYU data set: [{}] available: []".format(dataset_type, DATASET_TYPES)
        self.output_size = output_size
        self.resize = resize
        self.nyu_depth_v2_labeled_file = None
        self.exclude_mirrors = dataset_type == 'no_mirror'
        self.mirrors_only = dataset_type in ['mirror', 'mirror_corrected']
        self.use_corrected_depth = 'corrected' in dataset_type and not self.split == "train"
        self.use_mat = not dataset_type == 'sparse_2_dense'

        print("Use mat: ", self.use_mat)
        print("Use corrected depth: ", self.use_corrected_depth)
        
        if not self.use_mat:
            self.path = Path(path)/('train' if 'train' in self.split else 'val')
            if not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                download(self.path.parent/"nyudepthv2.tar.gz", NYU_V2_SPARSE2DENSE_URL)
                with tarfile.open(self.path.parent/"nyudepthv2.tar.gz", "r") as targz:
                    targz.extractall(self.path.parent)                
            self.images = [path.as_posix() for path in self.path.glob("**/*") if path.name.endswith('.h5')]
        else:
            self.path = Path(path)
            self.images = self.load_images()
            self.mapping40 = np.insert(loadmat(self.mapping40_file)['mapClass'][0], 0, 0)
        assert len(self.images) > 0, "Found 0 images in subfolders of: " + path + "\n"
        if self.exclude_mirrors: self.images = self.images[[idx for idx in np.arange(0, len(self.images)) if not idx in (TRAIN_MIRROR_IDX if self.split == "train" else VAL_MIRROR_IDX)]]
        if self.mirrors_only: self.images = self.images[[idx for idx in np.arange(0, len(self.images)) if idx in (TRAIN_MIRROR_IDX if self.split == "train" else VAL_MIRROR_IDX)]]
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
        self.nyu_depth_v2_labeled_file_corrected = (self.path/"nyu_depth_v2_labeled_corrected.mat")
        self.nyu_depth_v2_labeled_file = (self.path/"nyu_depth_v2_labeled.mat")
        self.split_file = (self.path/"split.mat")
        self.mapping40_file = (self.path/"classMapping40.mat")
        if self.use_corrected_depth and not self.nyu_depth_v2_labeled_file_corrected.exists(): download(self.nyu_depth_v2_labeled_file_corrected, NYU_V2_CORRECTED_MAT_URL)
        if not self.use_corrected_depth and not self.nyu_depth_v2_labeled_file.exists(): download(self.nyu_depth_v2_labeled_file, NYU_V2_LABELED_MAT_URL)
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
        if self.use_corrected_depth:
            data = h5py.File(self.nyu_depth_v2_labeled_file_corrected, "r")  
            depth = data['depths_corrected'][index]
            if np.max(depth) == 0: depth = data['depths'][index]   
        else:
            data = h5py.File(self.nyu_depth_v2_labeled_file, "r")  
            depth = data['depths'][index]   
        rgb = data['images'][index]     
        rgb = np.transpose(rgb, (2, 1, 0))
        depth = np.transpose(depth, (1,0))

        if self.mirrors_only:
            labels = data['labels'][index]
            labels = np.transpose(labels, (1,0))
            labels_40 = self.mapping40[labels]
            mask = labels_40 == 19 #Mirrors
            depth[~mask] = 0.0
        
        return rgb, depth

    @staticmethod
    def add_dataset_specific_args(parent_parser):
        parser = parent_parser.add_parser('nyu')
        BaseDataset.add_dataset_specific_args(parser)
        parser.add_argument('--type', type=str, default='labeled', help="available: {}".format(DATASET_TYPES))

if __name__ == "__main__":
    f = "mirrors.json"
    t = "val"
    nyu = NYUDataset("F:/data/nyudepthv2", split=t, dataset_type="official", use_corrected_depth=True)
    for item in nyu:
        visualize.show_item(item)
    """
    for _ in nyu:p
        pass
    if Path(f).exists():
        data = json.load(open(f, "r"))
    else:
        data = {}
    data["{}_indices".format(t)] = np.array(nyu.indices).astype(int).tolist()
    print(data)
    with open(f, "w") as json_file:
        json.dump(data, json_file)
    
    """
    