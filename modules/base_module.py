import torch
import pytorch_lightning as pl
from datasets.dataset import ConcatDataset
from datasets.nyu_dataloader import NYUDataset
from datasets.floorplan3d_dataloader import Floorplan3DDataset, DatasetType
from datasets.structured3d_dataset import Structured3DDataset
from metrics import MetricLogger
import visualize
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

class BaseModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = self.get_dataset(self.hparams.path, 'train', self.hparams.dataset)
        self.val_dataset = self.get_dataset(self.hparams.path, 'val', self.hparams.eval_dataset)
        self.test_dataset = self.get_dataset(self.hparams.path, 'test', self.hparams.test_dataset)
        self.train_dataset.transform = self.train_preprocess
        self.val_dataset.transform = self.val_preprocess
        self.test_dataset.transform = self.test_preprocess
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                    batch_size=self.hparams.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=self.hparams.worker, 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=self.hparams.worker, 
                                                    pin_memory=True) 
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=self.hparams.worker, 
                                                    pin_memory=True)                                     
        print("=> creating Model")
        self.model = self.setup_model()
        print("=> model created.")
        self.criterion = self.setup_criterion()
        self.metric_logger = MetricLogger(metrics=self.hparams.metrics)
        self.skip = len(self.val_loader) // 9

    def output_size(self):
        raise NotImplementedError()

    def resize(self):
        raise NotImplementedError()

    def setup_model(self):
        raise NotImplementedError()
    
    def setup_criterion(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

    def train_preprocess(self, rgb, depth):
        s = np.random.uniform(1, 1.5)
        depth = depth / s

        rgb = transforms.ToPILImage()(rgb)
        depth = transforms.ToPILImage()(depth)
        # color jitter
        rgb = transforms.ColorJitter(0.4, 0.4, 0.4)(rgb)
        # Resize
        resize = transforms.Resize(self.resize())
        rgb = resize(rgb)
        depth = resize(depth)
        # Random Rotation
        angle = np.random.uniform(-5,5)
        rgb = TF.rotate(rgb, angle)
        depth = TF.rotate(depth, angle)
        # Resize
        resize = transforms.Resize(int(self.resize() * s))
        rgb = resize(rgb)
        depth = resize(depth)
        # Center crop
        crop = transforms.CenterCrop(self.output_size())
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

    def val_preprocess(self, rgb, depth):
        rgb = transforms.ToPILImage()(rgb)
        depth = transforms.ToPILImage()(depth)
        # Resize
        resize = transforms.Resize(self.resize())
        rgb = resize(rgb)
        depth = resize(depth)
        # Center crop
        crop = transforms.CenterCrop(self.output_size())
        rgb = crop(rgb)
        depth = crop(depth)
        # Transform to tensor
        rgb = TF.to_tensor(np.array(rgb))
        depth = TF.to_tensor(np.array(depth))
        return rgb, depth

    def test_preprocess(self, rgb, depth):
        return self.val_preprocess(rgb, depth)

    def save_visualization(self, x, y, y_hat, batch_idx):
        if batch_idx == 0:
            self.img_merge = visualize.merge_into_row(x, y, y_hat)
        elif (batch_idx < 8 * self.skip) and (batch_idx % self.skip == 0):
            row = visualize.merge_into_row(x, y, y_hat)
            self.img_merge = visualize.add_row(self.img_merge, row)
        elif batch_idx == 8 * self.skip:
            filename = "{}/{}/version_{}/epoch{}.jpg".format(self.logger.save_dir, self.logger.name, self.logger.version, self.current_epoch)
            visualize.save_image(self.img_merge, filename)

    def get_dataset(self, path, split, dataset, use_mat=True, n_images=-1, mirrors_only=False, exclude_mirrors=False):
        path = path.split('+')
        if dataset == 'nyu':
            return NYUDataset(path[0], split=split, output_size=self.output_size(), resize=self.resize(), use_mat=use_mat, n_images=n_images, mirrors_only=mirrors_only, exclude_mirrors=exclude_mirrors)
        elif dataset == 'noreflection':
            return Floorplan3DDataset(path[0], split=split, datast_type=DatasetType.NO_REFLECTION, output_size=self.output_size(), resize=self.resize(), n_images=n_images)
        elif dataset == 'isotropic':
            return Floorplan3DDataset(path[0], split=split, datast_type=DatasetType.ISOTROPIC_MATERIAL, output_size=self.output_size(), resize=self.resize(), n_images=n_images)
        elif dataset == 'mirror':
            return Floorplan3DDataset(path[0], split=split, datast_type=DatasetType.ISOTROPIC_PLANAR_SURFACES, output_size=self.output_size(), resize=self.resize(), n_images=n_images)
        elif dataset == 'structured3d':
            return Structured3DDataset(path[0], split=split, dataset_type='perspective', output_size=self.output_size(), resize=self.resize())
        elif '+' in dataset:
            datasets = [self.get_dataset(p, split, d, use_mat=use_mat, n_images=n_images, mirrors_only=mirrors_only, exclude_mirrors=exclude_mirrors) for p, d in zip(path, dataset.split('+'))]
            return ConcatDataset(datasets)
        else:
            raise ValueError('unknown dataset {}'.format(dataset))

    @staticmethod
    def add_model_specific_args(parser):
        raise NotImplementedError()