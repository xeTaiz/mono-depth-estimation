import torch
import pytorch_lightning as pl
from datasets.dataset import ConcatDataset
from datasets.nyu_dataloader import NYUDataset, get_nyu_dataset
from datasets.floorplan3d_dataloader import Floorplan3DDataset, DatasetType, get_floorplan3d_dataset
from datasets.structured3d_dataset import Structured3DDataset, get_structured3d_dataset
from metrics import MetricLogger
import visualize
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

class BaseModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.globals = hparams
        self.hparams = hparams.method
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_dataset(hparams)
        self.train_dataset.transform = self.train_preprocess               
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                    batch_size=self.hparams.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=hparams.globals.worker, 
                                                    pin_memory=True)
        self.val_dataset.transform = self.val_preprocess 
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=hparams.globals.worker, 
                                                    pin_memory=True) 
        if self.test_dataset: 
            self.test_dataset.transform = self.test_preprocess                                                
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                        batch_size=1, 
                                                        shuffle=False, 
                                                        num_workers=hparams.globals.worker, 
                                                        pin_memory=True)
        else: self.test_loader = None                                 
        print("=> creating Model")
        self.model = self.setup_model()
        print("=> model created.")
        self.criterion = self.setup_criterion()
        self.metric_logger = MetricLogger(metrics=hparams.globals.metrics)
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

        if isinstance(rgb, np.ndarray):
            rgb = transforms.ToPILImage()(rgb)
        if isinstance(depth, np.ndarray):
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
        if isinstance(rgb, np.ndarray):
            rgb = transforms.ToPILImage()(rgb)
        if isinstance(depth, np.ndarray):
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

    def get_dataset(self, args):
        training_dataset = []
        validation_dataset = []
        test_dataset = []
        if hasattr(args, "nyu"):
            if args.nyu.training:   training_dataset.append(  get_nyu_dataset(args.nyu, 'train', self.output_size(), self.resize()))
            if args.nyu.validation: validation_dataset.append(get_nyu_dataset(args.nyu, 'val', self.output_size(), self.resize()))
            if args.nyu.test:       test_dataset.append(      get_nyu_dataset(args.nyu, 'test', self.output_size(), self.resize()))
        if hasattr(args, "floorplan3d"):
            if args.floorplan3d.training:   training_dataset.append(  get_floorplan3d_dataset(args.floorplan3d, 'train', self.output_size(), self.resize()))
            if args.floorplan3d.validation: validation_dataset.append(get_floorplan3d_dataset(args.floorplan3d, 'val', self.output_size(), self.resize()))
            if args.floorplan3d.test:       test_dataset.append(      get_floorplan3d_dataset(args.floorplan3d, 'test', self.output_size(), self.resize()))
        if hasattr(args, "structured3d"):
            if args.structured3d.training:   training_dataset.append(  get_structured3d_dataset(args.structured3d, 'train', self.output_size(), self.resize()))
            if args.structured3d.validation: validation_dataset.append(get_structured3d_dataset(args.structured3d, 'val', self.output_size(), self.resize()))
            if args.structured3d.test:       test_dataset.append(      get_structured3d_dataset(args.structured3d, 'test', self.output_size(), self.resize()))

        if len(training_dataset) > 1:   training_dataset = [ConcatDataset(training_dataset)]
        if len(validation_dataset) > 1: validation_dataset = [ConcatDataset(validation_dataset)]
        if len(test_dataset) > 1:       test_dataset = [ConcatDataset(test_dataset)]

        assert len(training_dataset) > 0, "No training dataset specified!"
        assert len(validation_dataset) > 0, "No validation dataset specified!"

        training_dataset = training_dataset[0]
        validation_dataset = validation_dataset[0]
        test_dataset = test_dataset[0] if test_dataset else None
        return training_dataset, validation_dataset, test_dataset

    @staticmethod
    def add_default_args(parser, name, learning_rate, batch_size, ckpt=None):
        parser.add_argument('--name', default=name, type=str, help="Method for training.")
        parser.add_argument('--learning_rate', default=learning_rate, type=float, help='Learning Rate')
        parser.add_argument('--batch_size',    default=batch_size,     type=int,   help='Batch Size')
        parser.add_argument('--ckpt',    default=ckpt,     type=str,   help='Load checkpoint')

    @staticmethod
    def add_model_specific_args(parser):
        raise NotImplementedError()