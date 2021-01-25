from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from datasets.dataset import BaseDataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch
from tqdm import tqdm
import urllib.request
from scipy.io import loadmat
import cv2
import visualize
import json

MIRROR_IDX = [25, 26, 76, 77, 86, 102, 131, 161, 162, 171, 172, 194, 195, 196, 199, 259, 266, 267, 268, 269, 271, 272, 273, 276, 277, 282, 283, 285, 286, 287, 290, 292, 294, 299, 302, 303, 305, 306, 308, 310, 313, 314, 323, 391, 401, 423, 427, 435, 440, 445, 457, 458, 487, 496, 505, 579, 583, 585, 586, 606, 609, 612, 613, 619]

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

def h5_loader(path, data, mapping):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth

def correct_depth(depth, mask, points):
    p0 = np.array(points[0:2])
    p1 = np.array(points[2:4])
    p2 = np.array(points[4:6])

    p0[[0,1]] = p0[[1,0]]
    p1[[0,1]] = p1[[1,0]]
    p2[[0,1]] = p2[[1,0]]

    d0 = np.append(p0, depth[p0[0], p0[1]])
    d1 = np.append(p1, depth[p1[0], p1[1]])
    d2 = np.append(p2, depth[p2[0], p2[1]])

    a = d0 - d1
    b = d2 - d1
    v = d1
    
    depth_corrected = np.copy(depth)

    (y_axis, x_axis) = np.where(mask==True)
    all_pixels = [[y,x] for (y,x) in zip(y_axis, x_axis)]
    
    all_pixels = np.array(all_pixels)
    b_div = b[1]/b[0]
    
    top = all_pixels[:, 1] - v[1] - all_pixels[:, 0]*b_div + b_div*v[0]
    bottom = a[1] - a[0]*b_div
    
    s = top / bottom
    t = (all_pixels[:, 0] - v[0] - a[0]*s)/b[0]
    correct_depth = v[2] + a[2]*s + b[2]*t
    depth_corrected[all_pixels[:, 0], all_pixels[:, 1]] = correct_depth
    return depth_corrected

def mat_loader(index, filename, mapping):
    data = h5py.File(filename, "r")  
    rgb = data['images'][index]
    depth = data["depths_corrected"][index]
    
    #depth = data['depths'][index]
    labels = data['labels'][index]
    #with open("points.json", "r") as json_file:
    #    point_data = json.load(json_file)
    rgb = np.transpose(rgb, (2, 1, 0))
    depth = np.transpose(depth, (1,0))
    labels = np.transpose(labels, (1,0))
    labels_40 = mapping[labels]
    mask = labels_40 == 19
    #depth[~mask] = 0.0
    
    mask_ = np.zeros_like(mask).astype(np.float32)
    
    if Path("{}_1.png".format(index)).is_file() and Path("{}_2.png".format(index)).is_file():
        filenames = ["{}_1.png".format(index), "{}_2.png".format(index)]
    else:
        filenames = ["{}.png".format(index)]
    for i,fn in enumerate(filenames):
        mask = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        mask = cv2.dilate(mask,np.ones((5,5),np.uint8),iterations = 1)
        _, mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        mask = (mask.astype(np.float32) / 255)
        mask_ += mask
        """
        if str(index) in point_data:
            points = point_data[str(index)]
            if isinstance(points[0], list):
                points = points[i] 
            depth_corrected = correct_depth(np.array(depth), mask, points)
            rgb = cv2.circle(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), tuple(points[0:2]), 10, (255,0,0), 2)
            rgb = cv2.circle(rgb, tuple(points[2:4]), 10, (255,0,0), 2)
            rgb = cv2.circle(rgb, tuple(points[4:6]), 10, (255,0,0), 2)
            visualize.show_corrected(rgb, depth, depth_corrected, mask, index, save=True)
        else:
            depth_corrected = np.copy(depth)
        depth = depth_corrected
    #depth[~mask] = 0.0
    data.close()
    
    data = h5py.File(filename, "a")
    if not "depths_corrected" in data:
        d_shape = data["depths"].shape 
        data.create_dataset("depths_corrected", d_shape)
    data["depths_corrected"][index] = np.transpose(depth_corrected, (1,0))
    data.close()
    """
    mask = mask_ > 0
    #depth[~mask] = 0.0
    return rgb, depth

class NYUDataset(BaseDataset):
    def __init__(self, path, output_size=(228, 304), resize=250, use_mat=False, n_images=-1, exclude_mirrors=False, mirrors_only=False, *args, **kwargs):
        super(NYUDataset, self).__init__(*args, **kwargs)
        self.output_size = output_size
        self.resize = resize
        self.nyu_depth_v2_labeled_file = None
        self.mapping40 = None
        use_mat = exclude_mirrors or mirrors_only
        if not use_mat:
            self.loader = h5_loader
            self.path = Path(path)/('train' if 'train' in self.split else 'val')
            self.images = [path.as_posix() for path in self.path.glob("**/*") if path.name.endswith('.h5')]            
        else:
            self.path = Path(path)
            self.loader = mat_loader
            self.images = self.load_images()
            self.mapping40 = np.insert(loadmat(self.path/'classMapping40.mat')['mapClass'][0], 0, 0)
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
        rgb, depth = self.loader(path, self.nyu_depth_v2_labeled_file, self.mapping40)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.stylization(bgr, sigma_s=60, sigma_r=0.3)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, depth

    def load_images(self):
        self.nyu_depth_v2_labeled_file = (self.path/"nyu_depth_v2_labeled.mat")
        self.split_file = (self.path/"split.mat")
        if not self.nyu_depth_v2_labeled_file.exists(): download_nyu_depth_v2_labeled(self.nyu_depth_v2_labeled_file)
        if not self.split_file.exists(): download_split(self.split_file)
        return np.hstack(loadmat(self.split_file)['trainNdxs' if self.split == 'train' else 'testNdxs']) - 1

if __name__ == "__main__":
    import visualize
    nyu = NYUDataset("G:/data/nyudepthv2", split="val", use_mat=True, mirrors_only=True)
    for item in nyu:
        visualize.show_item(item)
    
    
    